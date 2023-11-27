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

from torch.utils.data import ConcatDataset, DataLoader

from lib.dataset import *


def get_data_loaders(cfg):
    if cfg.TRAIN.OVERLAP:
        overlap = ((cfg.DATASET.CLIP-1)/float(cfg.DATASET.CLIP))
        # overlap = 0.9
    else:
        overlap = 0

    def get_2d_datasets(dataset_names):
        datasets = []
        for dataset_name in dataset_names:
            db = eval(dataset_name)(seqlen=cfg.DATASET.CLIP, overlap=overlap, debug=cfg.DEBUG)
            datasets.append(db)
        return ConcatDataset(datasets)

    def get_3d_datasets(dataset_names):
        datasets = []

        for dataset_name in dataset_names:
            db = eval(dataset_name)(set='train', seqlen=cfg.DATASET.CLIP, overlap=overlap, debug=cfg.DEBUG)
            datasets.append(db)
        return ConcatDataset(datasets)

    # ===== 2D keypoint datasets =====
    train_2d_dataset_names = cfg.TRAIN.DATASETS_2D
    train_2d_db = get_2d_datasets(train_2d_dataset_names)

    data_2d_batch_size = int(cfg.TRAIN.BATCH_SIZE * cfg.TRAIN.DATA_2D_RATIO)    # 19
    data_3d_batch_size = cfg.TRAIN.BATCH_SIZE - data_2d_batch_size

    train_2d_loader = DataLoader(
        dataset=train_2d_db,
        batch_size=data_2d_batch_size,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
    )

    # ===== 3D keypoint datasets =====
    train_3d_dataset_names = cfg.TRAIN.DATASETS_3D
    train_3d_db = get_3d_datasets(train_3d_dataset_names)

    train_3d_loader = DataLoader(
        dataset=train_3d_db,
        # batch_size=data_3d_batch_size,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        drop_last=True
    )

    # ===== Evaluation dataset =====
    overlap_v = ((cfg.DATASET.CLIP - 1) / float(cfg.DATASET.CLIP))
    valid_db = eval(cfg.TRAIN.DATASET_EVAL)(set='val', seqlen=cfg.DATASET.CLIP, overlap=overlap_v, debug=cfg.DEBUG)

    valid_loader = DataLoader(
        dataset=valid_db,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        drop_last=True
    )

    return None, train_3d_loader, valid_loader
