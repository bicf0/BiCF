# -*- coding: utf-8 -*-

import argparse
from yacs.config import CfgNode as CN

cfg = CN()

# out setting
# cfg.OUTPUT_DIR = 'results'
cfg.EXP_NAME = 'bilevel_3dpw'
cfg.LOGDIR = '/data1/wupeng/myProject/BiCF/logs'
cfg.DEVICE = 'cuda'
cfg.OUTPUT = '/data1/wupeng/myProject/BiCF/exp_out/exp1_'

cfg.DEBUG = False
cfg.NUM_WORKERS = 8
cfg.DEBUG_FREQ = 1000
cfg.SEED_VALUE = 2
# cfg.GPUS = 0,

cfg.CUDNN = CN()
cfg.CUDNN.BENCHMARK = True
cfg.CUDNN.DETERMINISTIC = False
cfg.CUDNN.ENABLED = True

# dataset setting
cfg.TRAIN = CN()
cfg.TRAIN.DATASETS_3D = ['h36m', 'mpi_inf_3dhp', '3dpw']
cfg.TRAIN.DATASET_EVAL = ['3dpw']
cfg.TRAIN.DATASET_WEIGHTS = [0.2, 0.3, 0.5]
cfg.TRAIN.OVERLAP = False
cfg.TRAIN.CLIP = 32
# train setting
cfg.TRAIN.BATCH_SIZE = 20
cfg.TRAIN.EPOCHS = 60
# cfg.TRAIN.PRETRAINED_REGRESSOR = '/data/wupeng/myProject/BiCF/BiCF/data/config_data/smpl_regressor_pretrained'
cfg.TRAIN.NUM_ITERS_PER_EPOCH = 500
cfg.TRAIN.LR_PATIENCE = 5
cfg.TRAIN.LR_RATIO = 10

# generator optimizer
cfg.TRAIN.GEN_OPTIM = 'Adam'
cfg.TRAIN.GEN_LR = 1e-4
cfg.TRAIN.GEN_WD = 1e-4
cfg.TRAIN.GEN_MOMENTUM = 0.9

# pretrained model
cfg.TRAIN.PRETRAINED = ''


# loss
# pose_level loss weights
# 2d_k, 3d_k, smpl:pose(theta), smpl:shape(beta), acc
cfg.LOSS = CN()
cfg.LOSS.NORMALIZE = False
cfg.LOSS.POSE_LOSS_WEIGHTS = [60.0, 300.0, 60.0, 60.0, 60.0]
# shape_level loss weights
# 2d_k,3d_k, smpl:pose(theta), smpl:shape(beta), same_shape
# cfg.LOSS.SHAPE_LOSS_WEIGHTS = [0.0, 0.0, 0.0, 0.0, 0.0]

# model setting
cfg.MODEL = CN()

cfg.MODEL.CLIP_ENCODER = CN()
cfg.MODEL.CLIP_ENCODER.DIM = 2048
cfg.MODEL.CLIP_ENCODER.DEPTH = 4
cfg.MODEL.CLIP_ENCODER.HEADS = 8
cfg.MODEL.CLIP_ENCODER.DIM_HEAD = 256
cfg.MODEL.CLIP_ENCODER.MLP_DIM = 512
cfg.MODEL.CLIP_ENCODER.DROPOUT = 0.

cfg.MODEL.CROSS_ENCODER = CN()
cfg.MODEL.CROSS_ENCODER.DIM = 2048
cfg.MODEL.CROSS_ENCODER.DEPTH = 4
cfg.MODEL.CROSS_ENCODER.HEADS = 8
cfg.MODEL.CROSS_ENCODER.DIM_HEAD = 256
cfg.MODEL.CROSS_ENCODER.MLP_DIM = 512
cfg.MODEL.CROSS_ENCODER.DROPOUT = 0.

# cfg.MODEL.TEMPORAL_ENCODER = CN()
# cfg.MODEL.TEMPORAL_ENCODER.DIM = 2048
# cfg.MODEL.TEMPORAL_ENCODER.DEPTH = 4
# cfg.MODEL.TEMPORAL_ENCODER.HEADS = 8
# cfg.MODEL.TEMPORAL_ENCODER.DIM_HEAD = 256
# cfg.MODEL.TEMPORAL_ENCODER.MLP_DIM = 512
# cfg.MODEL.TEMPORAL_ENCODER.DROPOUT = 0.

# cfg.MODEL.U_ENCODER = CN()
# cfg.MODEL.U_ENCODER.DIM = 2048
# cfg.MODEL.U_ENCODER.DEPTH = 4
# cfg.MODEL.U_ENCODER.HEADS = 8
# cfg.MODEL.U_ENCODER.DIM_HEAD = 256
# cfg.MODEL.U_ENCODER.MLP_DIM = 512
# cfg.MODEL.U_ENCODER.DROPOUT = 0.
#
# cfg.MODEL.L_ENCODER = CN()
# cfg.MODEL.L_ENCODER.DIM = 2048
# cfg.MODEL.L_ENCODER.DEPTH = 4
# cfg.MODEL.L_ENCODER.HEADS = 8
# cfg.MODEL.L_ENCODER.DIM_HEAD = 256
# cfg.MODEL.L_ENCODER.MLP_DIM = 512
# cfg.MODEL.L_ENCODER.DROPOUT = 0.


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()


def update_cfg(cfg_file):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_file)
    return cfg.clone()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path')

    args = parser.parse_args()
    print(args, end='\n\n')

    cfg_file = args.cfg
    if args.cfg is not None:
        cfg = update_cfg(args.cfg)
    else:
        cfg = get_cfg_defaults()

    return cfg, cfg_file
