# -*- encoding: utf-8 -*-
"""
File cnn.py
Created on 2024/4/5 23:16
Copyright (c) 2024/4/5
@author: 
"""
import torch.nn as nn
from einops import rearrange
from loguru import logger
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self, in_c, out_c, k_size = 3, stride = 1, pad = 0, act_mode="relu"):
        # k_size=3 stride=2 padding=1  --> 分辨率变为原来一半
        # k_size=3 stride=1 padding=1  --> 分辨率不变一半
        super().__init__()
        # 卷积层
        self.conv = nn.Conv2d(in_channels = in_c, out_channels=out_c, kernel_size=k_size, stride=stride, padding=pad)
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2)
        # 激活函数
        if act_mode == "sigmoid":
            self.act = nn.Sigmoid()
        elif act_mode == "relu":
            self.act = nn.ReLU()
        elif act_mode == "tanh":
            self.act = nn.Tanh()
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return self.act(x)
class CNN(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        # 两层卷积
        self.conv1 = ConvLayer(in_c = 1, out_c = 16, k_size = 5)     #output: 16 12 12
        self.conv2 = ConvLayer(in_c=16, out_c = 32, k_size = 5)      #output: 32 6 6
        # 全连接层
        self.fc1 = nn.Linear(in_features=32 * 4 * 4, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        # 输出类别数
        self.out = nn.Linear(in_features=128, out_features=class_num)

    def forward(self, images):
        x = self.conv1(images)
        x = self.conv2(x)
        # 平铺
        # B, C, H, W = x.shape
        flatten_x = rearrange(x, "b c h w -> b (c h w)")
        # print(flatten_x.shape)
        x = F.relu(self.fc1(flatten_x))
        x = F.relu(self.fc2(x))
        pred = self.out(x)
        return pred