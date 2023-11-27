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

import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'  # egl platform is not supported, use osmesa
import torch
import pprint
import random
import numpy as np
import torch.backends.cudnn as cudnn

from torch.utils.tensorboard import SummaryWriter
from lib.core.loss import ULoss
from lib.core.trainer import Trainer
from lib.core.config_bilevel import parse_args
from lib.utils.utils import prepare_output_dir
from lib.models import BiLevel
from lib.dataset.loader_data import get_data_loaders
from lib.utils.utils import create_logger, get_optimizer


def main(cfg):
    if cfg.SEED_VALUE >= 0:
        print(f'Seed value for the experiment {cfg.SEED_VALUE}')
        os.environ['PYTHONHASHSEED'] = str(cfg.SEED_VALUE)
        random.seed(cfg.SEED_VALUE)
        torch.manual_seed(cfg.SEED_VALUE)
        np.random.seed(cfg.SEED_VALUE)
    logger = create_logger(cfg.LOGDIR, phase='train')
    logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')
    logger.info(pprint.pformat(cfg))  # print cfg

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.DETERMINISTIC  # True
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC  # False
    cudnn.enabled = cfg.CUDNN.ENABLED  # True

    writer = SummaryWriter(log_dir=cfg.OUTPUT)
    writer.add_text('config', pprint.pformat(cfg), 0)

    # Data Loader
    data_loaders = get_data_loaders(cfg)

    # Loss Setting
    u_loss = ULoss(
        kp2d_loss_weight=cfg.LOSS.POSE_LOSS_WEIGHTS[0],
        kp3d_loss_weight=cfg.LOSS.POSE_LOSS_WEIGHTS[1],
        pose_loss_weight=cfg.LOSS.POSE_LOSS_WEIGHTS[2],
        shape_loss_weight=cfg.LOSS.POSE_LOSS_WEIGHTS[3],
        acc_loss_weight=cfg.LOSS.POSE_LOSS_WEIGHTS[4],
        device=cfg.DEVICE
    )

    model = BiLevel(
        clip_dim=cfg.MODEL.CLIP_ENCODER.DIM,
        clip_depth=cfg.MODEL.CLIP_ENCODER.DEPTH,
        clip_heads=cfg.MODEL.CLIP_ENCODER.HEADS,
        clip_dim_head=cfg.MODEL.CLIP_ENCODER.DIM_HEAD,
        clip_mlp_dim=cfg.MODEL.CLIP_ENCODER.MLP_DIM,
        clip_dropout=cfg.MODEL.CLIP_ENCODER.DROPOUT,
        clip_len=cfg.TRAIN.CLIP,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        cross_dim=cfg.MODEL.CROSS_ENCODER.DIM,
        cross_depth=cfg.MODEL.CROSS_ENCODER.DEPTH,
        cross_heads=cfg.MODEL.CROSS_ENCODER.HEADS,
        cross_dim_head=cfg.MODEL.CROSS_ENCODER.DIM_HEAD,
        cross_mlp_dim=cfg.MODEL.CROSS_ENCODER.MLP_DIM,
        cross_dropout=cfg.MODEL.CROSS_ENCODER.DROPOUT
    ).to(cfg.DEVICE)


    # if cfg.TRAIN.PRETRAINED != '' and os.path.isfile(cfg.TRAIN.PRETRAINED):
    #     checkpoint = torch.load(cfg.TRAIN.PRETRAINED)
    #     best_performance = checkpoint['performance']
    #     model.load_state_dict(checkpoint['gen_state_dict'])
    #     print(f'==> Loaded pretrained model from {cfg.TRAIN.PRETRAINED}...')
    #     print(f'Performance on 3DPW test set {best_performance}')
    # else:
    #     print(f'{cfg.TRAIN.PRETRAINED} is not a pretrained model!!!!')

    u_optimizer = get_optimizer(
        model=model,
        optim_type=cfg.TRAIN.GEN_OPTIM,
        lr=cfg.TRAIN.GEN_LR,
        weight_decay=cfg.TRAIN.GEN_WD,
        momentum=cfg.TRAIN.GEN_MOMENTUM,
    )

    u_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        u_optimizer,
        mode='min',
        factor=0.1,
        patience=cfg.TRAIN.LR_PATIENCE,
        verbose=True,
    )

    # Train Setting
    Trainer(
        data_loaders=data_loaders,
        dataset_nums=int(len(cfg.TRAIN.DATASETS_3D) + len(cfg.TRAIN.DATASET_EVAL)),
        model=model,
        u_loss=u_loss,
        u_optimizer=u_optimizer,
        u_lr_scheduler=u_lr_scheduler,
        epoch=cfg.TRAIN.EPOCHS,
        num_iters_per_epoch=cfg.TRAIN.NUM_ITERS_PER_EPOCH,
        performance_type='min',
        device=cfg.DEVICE,
        writer=writer,
        output_dir=cfg.OUTPUT
    ).fit()


if __name__ == '__main__':
    cfg, cfg_file = parse_args()
    cfg = prepare_output_dir(cfg, cfg_file)

    main(cfg)
