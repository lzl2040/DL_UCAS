# -*- encoding: utf-8 -*-
"""
File transformer.py
Created on 2024/5/6 22:48
Copyright (c) 2024/5/6
@author: 
"""
import torch
import torch.nn as nn
from einops import rearrange
from torch import einsum

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


def FeedForward(dim, mult = 4, dropout = 0):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Conv2d(dim, inner_dim, 1),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Conv2d(inner_dim, dim, 1),
        nn.Dropout(dropout)
    )

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        out = self.fn(x, *args, **kwargs)
        # print(out.shape, x.shape)
        return out + x

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dropout = 0):
        super().__init__()
        dim_head = dim // heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        # 在进行attention之前进行归一化，LLM中也有after norm
        self.prenorm = LayerNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)
        # 使用drop out随机
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # b, c, h, w = x.shape
        b, c, n = x.shape
        x = x.unsqueeze(-1)

        x = self.prenorm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        # print(qkv.shape)
        # q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
        q, k, v = map(lambda t: rearrange(t.squeeze(-1), 'b (h c) n -> b h c n', h = self.heads), qkv)
        # print(q.shape, k.shape)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        # print(out.shape)

        # out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        out = rearrange(out, 'b h n d -> b (h d) n')
        # out = self.to_out(out)
        out = self.to_out(out.unsqueeze(-1)).squeeze(-1)
        # print(out.shape)
        return self.dropout(out)

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        depth = 1,
        dropout = 0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(Attention(dim, heads = heads, dropout=dropout)),
                Residual(FeedForward(dim))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = x.unsqueeze(-1)
            x = ff(x)
            x = x.squeeze(-1)
        return x
