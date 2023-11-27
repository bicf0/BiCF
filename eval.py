import os
import torch

from lib.dataset import ThreedpwVal
from lib.models import TCMR, MPSnet, BiLevel
from lib.core.evaluate import Evaluator
from lib.core.config_eval import parse_args
from torch.utils.data import DataLoader


def main(cfg):
    print(f'...{cfg.MODEL.NAME} Evaluating on {cfg.DATASET.EVAL} test set...')

    model_name = cfg.MODEL.NAME

    model = None
    if model_name == 'Bilevel':
        model = BiLevel(
            clip_dim=cfg.MODEL.BICF.CLIP_ENCODER.DIM,
            clip_depth=cfg.MODEL.BICF.CLIP_ENCODER.DEPTH,
            clip_heads=cfg.MODEL.BICF.CLIP_ENCODER.HEADS,
            clip_dim_head=cfg.MODEL.BICF.CLIP_ENCODER.DIM_HEAD,
            clip_mlp_dim=cfg.MODEL.BICF.CLIP_ENCODER.MLP_DIM,
            clip_dropout=cfg.MODEL.BICF.CLIP_ENCODER.DROPOUT,
            clip_len=cfg.DATASET.CLIP,
            batch_size=cfg.DATASET.BATCH_SIZE,
            cross_dim=cfg.MODEL.BICF.CROSS_ENCODER.DIM,
            cross_depth=cfg.MODEL.BICF.CROSS_ENCODER.DEPTH,
            cross_heads=cfg.MODEL.BICF.CROSS_ENCODER.HEADS,
            cross_dim_head=cfg.MODEL.BICF.CROSS_ENCODER.DIM_HEAD,
            cross_mlp_dim=cfg.MODEL.BICF.CROSS_ENCODER.MLP_DIM,
            cross_dropout=cfg.MODEL.BICF.CROSS_ENCODER.DROPOUT
        ).to(cfg.DEVICE)

    elif model_name == 'tcmr':
        model = TCMR(
            n_layers=cfg.MODEL.TCMR.TGRU.NUM_LAYERS,
            batch_size=cfg.DATASET.BATCH_SIZE,
            seqlen=cfg.DATASET.CLIP,
            hidden_size=cfg.MODEL.TCMR.TGRU.HIDDEN_SIZE,
            pretrained=cfg.DATASET.PRETRAINED_REGRESSOR
        ).to(cfg.DEVICE)

    elif model_name == 'mpsnet':
        model = MPSnet(
            seqlen=cfg.DATASET.CLIP,
            n_layers=cfg.MODEL.MPSNET.TGRU.NUM_LAYERS,
            hidden_size=cfg.MODEL.MPSNET.TGRU.HIDDEN_SIZE
        ).to(cfg.DEVICE)

    if cfg.DATASET.PRETRAINED != '' and os.path.isfile(cfg.DATASET.PRETRAINED):
        checkpoint = torch.load(cfg.DATASET.PRETRAINED)
        best_performance = checkpoint['performance']
        model.load_state_dict(checkpoint['gen_state_dict'], strict=False)
        print(f'==> Loaded pretrained model from {cfg.DATASET.PRETRAINED}...')
        print(f'Performance on 3DPW test set {best_performance}')
    else:
        print(f'{cfg.DATASET.PRETRAINED} is not a pretrained model!!!!')
        exit()

    test_db = None
    if model_name == 'bicf':
        test_db = ThreedpwVal(dataset_name='3dpw', seq_len=cfg.DATASET.CLIP, overlap=0)
    elif model_name == 'tcmr' or model_name == 'mpsnet':
        test_db = ThreedpwVal(dataset_name='3dpw', seq_len=cfg.DATASET.CLIP, overlap=(cfg.DATASET.CLIP - 1) / float(cfg.DATASET.CLIP))

    test_loader = DataLoader(
        dataset=test_db,
        batch_size=cfg.DATASET.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        drop_last=True
    )

    Evaluator(
        model=model,
        device=cfg.DEVICE,
        test_loader=test_loader,
        clip_len=cfg.DATASET.CLIP,
        # split_rate=cfg.DATASET.CLIP_SPIT_RATE,
        batch_size=cfg.DATASET.BATCH_SIZE,
    ).run()


if __name__ == '__main__':
    cfg, cfg_file = parse_args()
    main(cfg)
