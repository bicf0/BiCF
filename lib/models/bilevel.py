# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn

from lib.core.config_model_data import CONFIG_DATA_PRETRAINED
from lib.models.spin import Regressor
from lib.models.transformer import Transformer, Fusion


class BiLevel(nn.Module):
    def __init__(
            self,
            clip_dim,
            clip_depth,
            clip_heads,
            clip_dim_head,
            clip_mlp_dim,
            clip_dropout,
            clip_len,
            batch_size,
            cross_dim,
            cross_depth,
            cross_heads,
            cross_dim_head,
            cross_mlp_dim,
            cross_dropout
    ):
        super(BiLevel, self).__init__()

        self.clip_len = clip_len
        self.batch_size = batch_size

        self.clip_encoder = Transformer(
            dim=clip_dim,
            depth=clip_depth,
            heads=clip_heads,
            dim_head=clip_dim_head,
            mlp_dim=clip_mlp_dim,
            dropout=clip_dropout
        )
        self.clip_encoder_positional = nn.Parameter(torch.zeros(1, self.clip_len, clip_dim))

        "RNN replace Transformer"
        # self.clip_encoder = nn.GRU(
        #     input_size=2048,
        #     hidden_size=2048,
        #     bidirectional=False,
        #     num_layers=2
        # )

        self.regressor = Regressor()

        self.cross_encoder = Transformer(
            dim=cross_dim,
            depth=cross_depth,
            heads=cross_heads,
            dim_head=cross_dim_head,
            mlp_dim=cross_mlp_dim,
            dropout=cross_dropout
        )
        self.cross_encoder_positional = nn.Parameter(torch.zeros(1, self.batch_size, cross_dim))

    def forward(self, x, J_regressor=None, is_train=True):
        # x: 64, 26, 2048
        dim = x.shape[-1]

        clip_x = self.clip_encoder(x + self.clip_encoder_positional)    # 64, 16, 2048
        # clip_x, _ = self.clip_encoder(x.permute(1, 0, 2))
        clip_x = clip_x + x

        inter_x = clip_x.permute(1, 0, 2)                               # 16, 64, 2048
        cross_x = self.cross_encoder(inter_x + self.cross_encoder_positional)   # 16, 64, 2048
        cross_x = cross_x + inter_x
        cross_x = cross_x.permute(1, 0, 2)                              # 64, 16, 2048

        feature = cross_x.reshape(-1, dim)

        if is_train:
            output = self.regressor(feature, J_regressor=J_regressor)

            for s in output:
                s['theta'] = s['theta'].reshape(self.batch_size, self.clip_len, -1)             # (b, 8, 85)
                s['verts'] = s['verts'].reshape(self.batch_size, self.clip_len, -1, 3)          # (b, 8, 6890, 3)
                s['kp_2d'] = s['kp_2d'].reshape(self.batch_size, self.clip_len, -1, 2)          # (b, 8, 49, 2)
                s['kp_3d'] = s['kp_3d'].reshape(self.batch_size, self.clip_len, -1, 3)          # (b, 8, 49, 3)
                s['rotmat'] = s['rotmat'].reshape(self.batch_size, self.clip_len, -1, 3, 3)     # (b, 8, 24, 3, 3)

            return output

        else:
            output = self.regressor(feature, J_regressor=J_regressor)

            for s in output:
                s['theta'] = s['theta'].reshape(self.batch_size, self.clip_len, -1)  # (b, 8, 85)
                s['verts'] = s['verts'].reshape(self.batch_size, self.clip_len, -1, 3)  # (b, 8, 6890, 3)
                s['kp_2d'] = s['kp_2d'].reshape(self.batch_size, self.clip_len, -1, 2)  # (b, 8, 49, 2)
                s['kp_3d'] = s['kp_3d'].reshape(self.batch_size, self.clip_len, -1, 3)  # (b, 8, 49, 3)
                s['rotmat'] = s['rotmat'].reshape(self.batch_size, self.clip_len, -1, 3, 3)  # (b, 8, 24, 3, 3)

            return output





