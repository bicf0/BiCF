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

import torch

from torch import nn, einsum
from einops import rearrange


# Norm
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# MLP(feedforward)
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# Attention

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super(Attention, self).__init__()
        # head 从 dim 维度划分，因此可以输入的图片数量任意
        inner_dim = dim_head * heads
        # True
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        # 最后一维做Softmax
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

        # self.alpha = nn.Parameter(torch.zeros(16))
        # self.fuse = nn.Linear(2, 1)
        self.guide_attend = nn.Softmax(dim=-1)

    def forward(self, x):
        b, n, c, h = *x.shape, self.heads  # （batch_size, number, D=dim, h)
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # (b, n(65), dim*3) ---> 3 * (b, n, dim)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # q, k, v   (b, h, n, dim_head(64))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # (b, h, n, n) (32, 8, 16, 16)
        attn = self.attend(dots)
        # # guide
        # x_ = x.reshape(b, n, h, -1).permute(0, 2, 1, 3)      # b, h, n, d
        # guide = einsum('b h i d, b h j d -> b h i j', x_, x_) * self.scale
        # guide_attn = self.guide_attend(guide)
        #
        # attn = attn + guide_attn

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# Transformer Block
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., state='o'):
        super(Transformer, self).__init__()
        self.state = state
        self.layers = nn.ModuleList([])
        # Transformer Encoder
        for d in range(depth):
            self.layers.append(nn.ModuleList([
                # single transformer block
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
            ]))

    def forward(self, x):
        # x_ = x.clone()
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return x


# feature fusion
class Fusion(nn.Module):
    def __init__(self, in_dim, nums, non_linearity='tanh'):
        super(Fusion, self).__init__()

        if non_linearity == 'relu':
            activation = nn.ReLU()
        else:
            activation = nn.Tanh()

        self.fc = nn.Linear(in_dim, 256)  # 2048 -> 256
        # self.relu = nn.ReLU()
        self.attention = nn.Sequential(
            nn.Linear(256 * nums, 256),
            activation,
            nn.Linear(256, 256),
            activation,
            nn.Linear(256, nums),
            activation
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_ = x.clone()
        batch = x.shape[0]
        x = self.fc(x)
        x = x.view(batch, -1)

        scores = self.attention(x)
        scores = self.softmax(scores)

        x = torch.mul(x_, scores[:, :, None])
        x = torch.sum(x, dim=1)

        return x
