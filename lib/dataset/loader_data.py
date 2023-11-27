# -*- coding:utf-8 -*-
from torch.utils.data import DataLoader
from lib.dataset import *


def get_data_loaders(cfg):
    # if cfg.TRAIN.OVERLAP:
    #     overlap = ((cfg.TRAIN.CLIP - 1) / float(cfg.TRAIN.CLIP))
    # else:
    overlap = 0     # make input output equal

    train_dataset_names = cfg.TRAIN.DATASETS_3D
    train_dataset_weight = cfg.TRAIN.DATASET_WEIGHTS
    val_dataset_name = cfg.TRAIN.DATASET_EVAL

    train_val_data_loader = []

    for dataset_name in train_dataset_names:
        if dataset_name == 'h36m':
            weight = train_dataset_weight[train_dataset_names.index(dataset_name)]
            db = H36mTrain(dataset_name=dataset_name, dataset_weight=weight, seq_len=cfg.TRAIN.CLIP, overlap=overlap)
            db_loader = DataLoader(
                dataset=db,
                # batch_size=int(cfg.TRAIN.BATCH_SIZE*weight),
                batch_size=12,
                shuffle=True,
                num_workers=cfg.NUM_WORKERS,
                drop_last=True
            )
            train_val_data_loader.append(db_loader)

        elif dataset_name == 'mpi_inf_3dhp':
            weight = train_dataset_weight[train_dataset_names.index(dataset_name)]
            db = Mpii3dTrain(dataset_name=dataset_name, dataset_weight=weight, seq_len=cfg.TRAIN.CLIP, overlap=overlap)
            db_loader = DataLoader(
                dataset=db,
                # batch_size=int(cfg.TRAIN.BATCH_SIZE * weight),
                batch_size=20,
                shuffle=True,
                num_workers=cfg.NUM_WORKERS,
                drop_last=True
            )
            train_val_data_loader.append(db_loader)
        else:
            weight = train_dataset_weight[train_dataset_names.index(dataset_name)]
            db = Mpii3dTrain(dataset_name=dataset_name, dataset_weight=weight, seq_len=cfg.TRAIN.CLIP, overlap=0)
            db_loader = DataLoader(
                dataset=db,
                # batch_size=int(cfg.TRAIN.BATCH_SIZE * weight),
                batch_size=32,
                shuffle=True,
                num_workers=cfg.NUM_WORKERS,
                drop_last=True
            )
            train_val_data_loader.append(db_loader)

    if val_dataset_name[0] == 'h36m':
        db_val = H36mVal(dataset_name='h36m', seq_len=cfg.TRAIN.CLIP, overlap=0)
        db_loader = DataLoader(
            dataset=db_val,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.NUM_WORKERS,
            drop_last=True
        )
        train_val_data_loader.append(db_loader)

    elif val_dataset_name[0] == 'mpi_inf_3dhp':
        db_val = Mpii3dVal(dataset_name='mpi_inf_3dhp', seq_len=cfg.TRAIN.CLIP, overlap=0)
        db_loader = DataLoader(
            dataset=db_val,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.NUM_WORKERS,
            drop_last=True
        )
        train_val_data_loader.append(db_loader)

    else:
        db_val = ThreedpwVal(dataset_name='3dpw', seq_len=cfg.TRAIN.CLIP, overlap=0)
        db_loader = DataLoader(
            dataset=db_val,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            # batch_size=32,
            shuffle=False,
            num_workers=cfg.NUM_WORKERS,
            drop_last=True
        )
        train_val_data_loader.append(db_loader)

    return train_val_data_loader

