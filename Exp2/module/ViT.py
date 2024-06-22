# -*- encoding: utf-8 -*-
"""
File ViT.py
Created on 2024/5/6 22:37
Copyright (c) 2024/5/6
@author: 
"""
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from Exp2.module.transformer import Transformer


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class LayerNorm(nn.Module):
    def __init__(self, dim, bias = False):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1)) if bias else None

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g + default(self.b, 0)
class ViT(nn.Module):
    def __init__(self, img_size = 224, patch_size=16, embed_dim=1024, depth=24,
                 num_heads=16, num_classes=10, img_channels = 3, dropout=0, pool = "cls"):
        super().__init__()
        assert exists(img_size)
        assert (img_size % patch_size) == 0

        # patch数目
        num_patches_height_width = img_size // patch_size

        # patch的特征维度
        patch_dim = img_channels * (patch_size ** 2)

        self.to_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = patch_size, p2 = patch_size),
            nn.Conv2d(patch_dim, embed_dim, 1),
            LayerNorm(embed_dim)
        )

        # 初始化为0
        self.pos_emb = nn.Parameter(torch.zeros(1, embed_dim, num_patches_height_width * num_patches_height_width + 1))
        # 随机初始化
        self.cls_token = nn.Parameter(torch.randn(1, embed_dim, 1))

        self.transformer = Transformer(
            dim = embed_dim,
            depth = depth,
            dropout = dropout,
            heads = num_heads
        )

        self.pool = pool
        # 分类
        self.cls_head = nn.Linear(embed_dim, num_classes)

        # 去patch
        # self.to_patches = nn.Sequential(
        #     LayerNorm(dim),
        #     nn.Conv2d(dim, output_patch_dim, 1),
        #     Rearrange('b (c p1 p2) h w -> b c (h p1) (w p2)', p1 = patch_size, p2 = patch_size),
        # )

    def forward(self, x):
        x = self.to_tokens(x)
        b, c, h, w = x.shape
        x = rearrange(x, "b c h w -> b c (h w)")
        # 跟cls token进行拼接
        cls_tokens = repeat(self.cls_token, "b c 1 -> (repeat b) c 1", repeat=b)
        x = torch.cat([cls_tokens, x], dim = 2)
        # print(x.shape, self.pos_emb.shape)

        x = x + self.pos_emb

        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == "mean" else x[:, :, 0]
        return self.cls_head(x)


def vit_base_patch16_224(num_classes = 10):
    # model = ViT(img_size=48,
    #           patch_size=8,
    #           embed_dim=768,
    #           depth=12,
    #           num_heads=12,
    #           num_classes=num_classes)
    model = ViT(img_size=224,
              patch_size=16,
              embed_dim=256,
              depth=8,
              num_heads=8,
              num_classes=num_classes)
    return model


def vit_base_patch32_224(num_classes = 10):
    model = ViT(img_size=224,
              patch_size=32,
              embed_dim=768,
              depth=12,
              num_heads=12,
              num_classes=num_classes)
    return model

def vit_large_patch16_224(num_classes = 10):
    model = ViT(img_size=224,
              patch_size=16,
              embed_dim=1024,
              depth=24,
              num_heads=16,
              num_classes=num_classes)
    return model
