# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import time
import torch
import shutil
import logging
import numpy as np
import os
from progress.bar import Bar

from lib.core.config_model_data import CONFIG_DATA_PRETRAINED
from lib.utils.utils import move_dict_to_device, AverageMeter

from lib.utils.eval_utils import (
    compute_accel,
    compute_error_accel,
    compute_error_verts,
    batch_compute_similarity_transform_torch,
)

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
            self,
            data_loaders,
            dataset_nums,
            model,
            u_loss,
            u_optimizer,
            u_lr_scheduler,
            epoch,
            num_iters_per_epoch,
            performance_type='min',
            device='cuda',
            writer=None,
            output_dir=None
    ):
        self.dataset_nums = dataset_nums

        if dataset_nums == 4:
            self.h36m_data_loader, self.mpii_data_loader, self.pw3d_data_loader, self.val_data_loader = data_loaders
            self.h36m_iter, self.mpii_iter,  self.pw3d_iter = \
                iter(self.h36m_data_loader), iter(self.mpii_data_loader), iter(self.pw3d_data_loader)
        else:
            self.h36m_data_loader, self.mpii_data_loader, self.val_data_loader = data_loaders
            self.h36m_iter, self.mpii_iter = iter(self.h36m_data_loader), iter(self.mpii_data_loader)

        # Models and Optimizers
        self.model = model
        self.u_loss = u_loss
        self.u_optimizer = u_optimizer
        self.u_lr_scheduler = u_lr_scheduler

        # Training Parameters
        self.exchange = True
        self.end_epoch = epoch
        self.device = device
        self.performance_type = performance_type
        self.writer = writer
        self.output_dir = output_dir
        self.train_global_step = 0
        self.valid_global_step = 0
        self.epoch = 0
        self.exchange = False
        self.best_performance = float('inf') if performance_type == 'min' else -float('inf')
        self.num_iters_per_epoch = num_iters_per_epoch
        # metrics are related to 3d joint and vertex
        self.evaluation_accumulators = dict.fromkeys(['pred_j3d', 'target_j3d', 'target_theta', 'pred_verts'])

        # if self.writer is None:
        #     from torch.utils.tensorboard import SummaryWriter
        #     self.writer = SummaryWriter(log_dir=self.logdir)

        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def train(self):
        # record loss
        losses = AverageMeter()

        # record time
        timer = {
            'data': 0,
            'forward': 0,
            'loss': 0,
            'backward': 0,
            'batch': 0,
        }

        # mode train
        self.model.train()
        start = time.time()
        summary_string = ''
        bar = Bar(f'Epoch {self.epoch + 1}/{self.end_epoch}', fill='#', max=self.num_iters_per_epoch)

        for i in range(self.num_iters_per_epoch):
            # data ready
            target_h36m = None
            target_mpii = None
            # without 3dpw train
            target_pw3d = None
            if self.h36m_iter:
                try:
                    target_h36m = next(self.h36m_iter)
                except StopIteration:
                    self.h36m_iter = iter(self.h36m_data_loader)
                    target_h36m = next(self.h36m_iter)
                move_dict_to_device(target_h36m, self.device)

            if self.mpii_iter:
                try:
                    target_mpii = next(self.mpii_iter)
                except StopIteration:
                    self.mpii_iter = iter(self.mpii_data_loader)
                    target_mpii = next(self.mpii_iter)
                move_dict_to_device(target_mpii, self.device)

            if self.pw3d_iter:
                try:
                    target_pw3d = next(self.pw3d_iter)
                except StopIteration:
                    self.pw3d_iter = iter(self.pw3d_data_loader)
                    target_pw3d = next(self.pw3d_iter)
                move_dict_to_device(target_pw3d, self.device)

            inp = torch.cat((target_h36m['features'], target_mpii['features'], target_pw3d['features']), dim=0).to(self.device)  #
            timer['data'] = time.time() - start
            start = time.time()

            # forward
            preds = self.model(inp)
            timer['forward'] = time.time() - start
            start = time.time()

            # loss
            loss, loss_dict = self.u_loss(
                preds,
                target_h36m,
                target_mpii,
                target_pw3d,
            )

            timer['loss'] = time.time() - start
            start = time.time()

            # backward
            self.u_optimizer.zero_grad()
            loss.backward()
            self.u_optimizer.step()
            losses.update(loss.item(), inp.size(0))

            timer['backward'] = time.time() - start

            timer['batch'] = timer['data'] + timer['forward'] + timer['loss'] + timer['backward']
            start = time.time()

            summary_string = f'({i + 1}/{self.num_iters_per_epoch}) | Total: {bar.elapsed_td} | ' \
                             f'ETA: {bar.eta_td:} | loss: {losses.avg:.4f}'

            for k, v in loss_dict.items():
                summary_string += f' | {k}: {v:.2f}'
                self.writer.add_scalar('train_loss/'+k, v, global_step=self.train_global_step)

            for k, v in timer.items():
                summary_string += f' | {k}: {v:.2f}'

            self.writer.add_scalar('train_loss/loss', loss.item(), global_step=self.train_global_step)

            self.train_global_step += 1
            bar.suffix = summary_string
            bar.next()

            if torch.isnan(loss):
                exit('Nan value in loss, exiting!...')

    # val
    def validate(self):

        self.model.eval()

        start = time.time()
        summary_string = ''
        bar = Bar('Validation', fill='#', max=len(self.val_data_loader))

        if self.evaluation_accumulators is not None:
            for k, v in self.evaluation_accumulators.items():
                self.evaluation_accumulators[k] = []

        J_regressor = torch.from_numpy(np.load(os.path.join(CONFIG_DATA_PRETRAINED, 'J_regressor_h36m.npy'))).float()

        for i, target in enumerate(self.val_data_loader):
            move_dict_to_device(target, self.device)

            with torch.no_grad():

                inp = target['features']
                preds = self.model(inp, J_regressor=J_regressor, is_train=False)

                n_kp = preds[-1]['kp_3d'].shape[-2]
                pred_j3d = preds[-1]['kp_3d'].reshape(-1, n_kp, 3).cpu().numpy()   # 32*8, 14, 3

                target_j3d = target['kp_3d'].reshape(-1, n_kp, 3).cpu().numpy()     # 32*8, 14, 3

                pred_verts = preds[-1]['verts'].reshape(-1, 6890, 3).cpu().numpy()
                target_theta = target['theta'].reshape(-1, 85).cpu().numpy()

                self.evaluation_accumulators['pred_j3d'].append(pred_j3d)
                self.evaluation_accumulators['target_j3d'].append(target_j3d)
                self.evaluation_accumulators['pred_verts'].append(pred_verts)
                self.evaluation_accumulators['target_theta'].append(target_theta)

            batch_time = time.time() - start

            summary_string = f'({i + 1}/{len(self.val_data_loader)}) | batch: {batch_time * 10.0:.4}ms | ' \
                             f'Total: {bar.elapsed_td} | ETA: {bar.eta_td:}'

            self.valid_global_step += 1
            bar.suffix = summary_string
            bar.next()
            if i >= 300:
                break

        bar.finish()
        logger.info(summary_string)

    def fit(self):

        for epoch in range(0, self.end_epoch):

            self.epoch = epoch  # current epoch
            self.train()
            self.validate()
            performance = self.evaluate()
            if self.u_lr_scheduler is not None:
               self.u_lr_scheduler.step(performance)

            # log the learning rate
            for param_group in self.u_optimizer.param_groups:
                print(f'Learning rate {param_group["lr"]}')
                self.writer.add_scalar('lr/gen_lr', param_group['lr'], global_step=self.epoch)

            logger.info(f'Epoch {epoch+1} performance: {performance:.4f}')

            self.save_model(performance, epoch)

        self.writer.close()

    def evaluate(self):

        for k, v in self.evaluation_accumulators.items():
            self.evaluation_accumulators[k] = np.vstack(v)

        pred_j3ds = self.evaluation_accumulators['pred_j3d']
        target_j3ds = self.evaluation_accumulators['target_j3d']

        pred_j3ds = torch.from_numpy(pred_j3ds).float()
        target_j3ds = torch.from_numpy(target_j3ds).float()

        print(f'Evaluating on {pred_j3ds.shape[0]} number of poses...')
        pred_pelvis = (pred_j3ds[:, [2], :] + pred_j3ds[:, [3], :]) / 2.0
        target_pelvis = (target_j3ds[:, [2], :] + target_j3ds[:, [3], :]) / 2.0

        pred_j3ds -= pred_pelvis
        target_j3ds -= target_pelvis
        # Absolute error (MPJPE)
        errors = torch.sqrt(((pred_j3ds - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        S1_hat = batch_compute_similarity_transform_torch(pred_j3ds, target_j3ds)
        errors_pa = torch.sqrt(((S1_hat - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        pred_verts = self.evaluation_accumulators['pred_verts']
        target_theta = self.evaluation_accumulators['target_theta']

        # errors = torch.sqrt(((pred_j3ds - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1)
        # S1_hat = batch_compute_similarity_transform_torch(pred_j3ds, target_j3ds)
        # errors_pa = torch.sqrt(((S1_hat - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1)
        # pred_verts = self.evaluation_accumulators['pred_verts']
        # target_theta = self.evaluation_accumulators['target_theta']

        m2mm = 1000

        pve = np.mean(compute_error_verts(target_theta=target_theta, pred_verts=pred_verts)) * m2mm

        """
        # modify dim for acc calculate
        # pred_j3ds = pred_j3ds.reshape(-1, 4, 3, test14, 3)[:, :, [1]]
        pred_j3ds = pred_j3ds.reshape(-1, 4, test14, 3)
        pred_j3ds = pred_j3ds.reshape(-1, 8, 4, test14, 3)
        pred_j3ds = pred_j3ds.permute(0, 2, 1, 3, 4)
        pred_j3ds = pred_j3ds.reshape(-1, test14, 3)
        # target_j3ds = target_j3ds.reshape(-1, 4, 3, test14, 3)[:, :, [1]]
        target_j3ds = target_j3ds.reshape(-1, 4, test14, 3)
        target_j3ds = target_j3ds.reshape(-1, 8, 4, test14, 3)
        target_j3ds = target_j3ds.permute(0, 2, 1, 3, 4)
        target_j3ds = target_j3ds.reshape(-1, test14, 3)
        """

        accel = np.mean(compute_accel(pred_j3ds)) * m2mm
        accel_err = np.mean(compute_error_accel(joints_pred=pred_j3ds, joints_gt=target_j3ds)) * m2mm
        mpjpe = np.mean(errors) * m2mm
        pa_mpjpe = np.mean(errors_pa) * m2mm

        eval_dict = {
            'mpjpe': mpjpe,
            'pa-mpjpe': pa_mpjpe,
            'accel': accel,
            'pve': pve,
            'accel_err': accel_err
        }

        log_str = f'Epoch {self.epoch + 1}, '
        log_str += ' '.join([f'{k.upper()}: {v:.4f},' for k, v in eval_dict.items()])
        logger.info(log_str)

        for k, v in eval_dict.items():
            self.writer.add_scalar(f'error/{k}', v, global_step=self.epoch)

        return pa_mpjpe

    def save_model(self, performance, epoch):
        save_dict = {
            'epoch': epoch,
            'gen_state_dict': self.model.state_dict(),
            'performance': performance,
            'gen_optimizer': self.u_optimizer.state_dict()
        }

        filename = os.path.join(self.output_dir, 'checkpoint.pth.tar')
        torch.save(save_dict, filename)

        if self.performance_type == 'min':
            is_best = performance < self.best_performance
        else:
            is_best = performance > self.best_performance

        if is_best:
            logger.info('Best performance achived, saving it!')
            self.best_performance = performance
            shutil.copyfile(filename, os.path.join(self.output_dir, 'model_best2.pth.tar'))

            with open(os.path.join(self.output_dir, 'best.txt'), 'w') as f:
                f.write(str(float(performance)))
