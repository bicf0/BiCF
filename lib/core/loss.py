# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from lib.utils.geometry import batch_rodrigues


class ULoss(nn.Module):
    def __init__(
            self,
            kp2d_loss_weight,
            kp3d_loss_weight,
            pose_loss_weight,
            shape_loss_weight,
            acc_loss_weight,
            device='cuda'
    ):
        super(ULoss, self).__init__()

        self.kp2d_loss_weight = kp2d_loss_weight
        self.kp3d_loss_weight = kp3d_loss_weight
        self.pose_loss_weight = pose_loss_weight
        self.shape_loss_weight = shape_loss_weight
        self.acc_loss_weight = acc_loss_weight

        self.device = device

        self.theta_loss = nn.L1Loss().to(self.device)
        self.kp_loss = nn.L1Loss(reduction='none').to(self.device)
        self.consistent_loss = nn.L1Loss().to(self.device)

    def forward(
            self,
            preds,
            data_h36m,
            data_mpii,
            data_pw3d,
    ):

        reduce = lambda x: x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])
        flatten = lambda x: x.reshape(-1)

        # joint_weights = preds[-1]['joint_weights']      # 14, 3

        # label
        real_kp2d = torch.cat([data_h36m['kp_2d'], data_mpii['kp_2d'], data_pw3d['kp_2d']], dim=0)  # 32, 16, 49, 2
        real_kp2d = reduce(real_kp2d)

        real_kp3d = torch.cat([data_h36m['kp_3d'], data_mpii['kp_3d'], data_pw3d['kp_3d']], dim=0)  # 32, 32, 49, 3

        real_theta = torch.cat([data_h36m['theta'], data_mpii['theta'], data_pw3d['theta']], dim=0)     # 32, 32, 85
        real_theta = reduce(real_theta)                                                                 # 1024, 85

        # w_3d and w_smpl is used for filter the samples without 3d or smpl labels
        w_3d = torch.cat([data_h36m['w_3d'].type(torch.bool), data_mpii['w_3d'].type(torch.bool), data_pw3d['w_3d'].type(torch.bool)], dim=0)   # 32, 32
        w_3d = reduce(w_3d)                             # 1024
        w_3d = flatten(w_3d)                          # 1024,

        w_smpl = torch.cat([data_h36m['w_smpl'].type(torch.bool), data_mpii['w_smpl'].type(torch.bool), data_pw3d['w_smpl'].type(torch.bool)], dim=0)
        w_smpl = reduce(w_smpl)
        w_smpl = flatten(w_smpl)                      # 1024,

        real_theta = real_theta[w_smpl]
        real_shape, real_pose = real_theta[:, 75:], real_theta[:, 3:75]

        # prediction
        preds = preds[-1]
        pre_kp2d = preds['kp_2d']
        pre_kp2d = reduce(pre_kp2d)

        pre_kp3d = preds['kp_3d']

        pre_theta = preds['theta']            # 64, 16, 85
        pre_theta = reduce(pre_theta)

        loss_dict = {}

        pre_theta = pre_theta[w_smpl]
        pre_shape, pre_pose = pre_theta[:, 75:], pre_theta[:, 3:75]

        # loss calculate
        result_kp2d = self.kp2d_criterion(pre_kp2d, real_kp2d, openpose_weight=1., gt_weight=1., kp2d_loss_weight=self.kp2d_loss_weight)
        result_kp3d = self.kp3d_criterion(reduce(real_kp3d)[w_3d], reduce(pre_kp3d)[w_3d], self.kp3d_loss_weight)
        result_acc = self.acc_loss(real_kp3d, pre_kp3d, self.acc_loss_weight)

        loss_dict['loss_kp2d'] = result_kp2d
        loss_dict['loss_kp3d'] = result_kp3d
        loss_dict['loss_acc'] = result_acc

        if pre_theta.shape[0] > 0:
            result_pose, result_shape = self.smpl_criterion(pre_pose, pre_shape, real_pose, real_shape, self.pose_loss_weight, self.shape_loss_weight)
            loss_dict['loss_pose'] = result_pose
            loss_dict['loss_shape'] = result_shape

        loss = torch.stack(list(loss_dict.values())).sum()

        return loss, loss_dict

    # kp_2d loss
    def kp2d_criterion(self, pre_kp2d, gt_kp2d, openpose_weight, gt_weight, kp2d_loss_weight):
        """
        Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        The available keypoints are different for each dataset.
        """
        conf = gt_kp2d[:, :, -1].unsqueeze(-1).clone()
        conf[:, :25] *= openpose_weight
        conf[:, 25:] *= gt_weight
        loss = (conf * self.kp_loss(pre_kp2d, gt_kp2d[:, :, :-1])).mean()
        return loss * kp2d_loss_weight

    # kp_3d loss
    def kp3d_criterion(self, pre_kp3d, gt_kp3d, kp3d_loss_weight):
        """
        Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence.
        """
        pre_kp3d = pre_kp3d[:, 25:39, :]
        gt_kp3d = gt_kp3d[:, 25:39, :]

        pre_kp3d = pre_kp3d
        if len(gt_kp3d) > 0:
            gt_pelvis = (gt_kp3d[:, 2, :] + gt_kp3d[:, 3, :]) / 2
            gt_kp3d = gt_kp3d - gt_pelvis[:, None, :]
            pred_pelvis = (pre_kp3d[:, 2, :] + pre_kp3d[:, 3, :]) / 2
            pre_kp3d = pre_kp3d - pred_pelvis[:, None, :]                       # 64*16, 14, 3
            return self.kp_loss(pre_kp3d, gt_kp3d).mean() * kp3d_loss_weight
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device) * kp3d_loss_weight

    # smpl loss
    def smpl_criterion(self, pre_pose, pre_beta, gt_pose, gt_beta, pose_loss_weight, shape_loss_weigh):
        pre_pose = batch_rodrigues(pre_pose.reshape(-1, 3)).reshape(-1, 24, 3, 3)
        gt_pose = batch_rodrigues(gt_pose.reshape(-1, 3)).reshape(-1, 24, 3, 3)
        pre_beta = pre_beta
        gt_beta = gt_beta
        if len(pre_pose) > 0:
            loss_pose = self.theta_loss(pre_pose, gt_pose)
            loss_beta = self.theta_loss(pre_beta, gt_beta)
        else:
            loss_pose = torch.FloatTensor(1).fill_(0.).to(self.device)
            loss_beta = torch.FloatTensor(1).fill_(0.).to(self.device)
        return loss_pose * pose_loss_weight, loss_beta * shape_loss_weigh

    def consistent_criterion(self, pre_beta, consistent_loss_weight):
        # 64*16, 10
        pre_beta = pre_beta.reshape(64, 16, 10)  # 64, 16, 10
        pre = pre_beta[:, :-1]
        pre = pre.reshape(-1, pre_beta.shape[-1])
        gt = pre_beta[:, 1:]
        gt = gt.reshape(-1, pre_beta.shape[-1])
        # pre_beta = pre_beta.permute(0, 2, 1, 3)
        # pre_beta = pre_beta.reshape(-1, self.split_rate, 10)
        return self.consistent_loss(pre, gt) * consistent_loss_weight

    def acc_loss(self, joints_gt, joints_pred, acc_loss_weight):
        # dui qi, shi yong 14
        # 32, 8, 49(14), 3
        joints_gt = joints_gt[:, :, 25:39, :]
        gt_pelvis = (joints_gt[:, :, [2], :] + joints_gt[:, :, [3], :]) / 2
        joints_gt -= gt_pelvis
        accel_gt = joints_gt[:, :-2] - 2 * joints_gt[:, 1:-1] + joints_gt[:, 2:]
        accel_gt = accel_gt.reshape(-1, 14, 3)

        joints_pred = joints_pred[:, :, 25:39, :]
        pre_pelvis = (joints_pred[:, :, [2], :] + joints_pred[:, :, [3], :]) / 2
        joints_pred -= pre_pelvis
        accel_pred = joints_pred[:, :-2] - 2 * joints_pred[:, 1:-1] + joints_pred[:, 2:]
        accel_pred = accel_pred.reshape(-1, 14, 3)
        normed = torch.linalg.norm(accel_pred - accel_gt, axis=2)
        new_vis = torch.ones(len(normed), dtype=torch.bool)

        return torch.mean(torch.mean(normed[new_vis], dim=1), dim=0) * acc_loss_weight
