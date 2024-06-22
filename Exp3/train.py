# -*- encoding: utf-8 -*-
"""
File train.py
Created on 2024/5/15 11:35
Copyright (c) 2024/5/15
@author: 
"""
import argparse
import os

import numpy as np
import torch
from timm.utils import AverageMeter
from torch.utils.data import DataLoader

from tokenizer import Tokenizer
from generate_poem import generate_poem_with_lstm, generate_poem_with_transformer
from datasets import PoemDataset
from module.Poem import PoemModel_LSTM, PoemModel_Trasformer
from loguru import logger
import torch.optim as optim
import torch.nn as nn
from hyper_para import get_args
import random

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

# 保证结果可重复
torch.manual_seed(123243)
# 检查CUDA是否可用
if torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info("CUDA is available! Training on GPU.")
else:
    device = torch.device("cpu")
    logger.info("CUDA is not available. Training on CPU.")


def print_poem(data, pos, ix2word):
    p_len = data.shape[1]
    for i in range(p_len):
        id = data[pos, i]
        # 填充符的位置 <START> 开始标识 <EOP> 结束标识 </s>填充符
        # 诗数据的内容：</s> </s> <START> 诗的内容 <EOP>
        if id < 8290:
            print(ix2word[id], end = "")

def construct_loader(args):
    poem_data = PoemDataset(args)
    dataloader = DataLoader(poem_data,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers)
    return dataloader, poem_data.word2ix, poem_data.ix2word

def train(model, data_loader, vocab_size, args):
    logger.info("Start training")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=5e-5)
    criterion = nn.CrossEntropyLoss()
    global_iter = 0
    loss_avg = AverageMeter()
    tokenizer = Tokenizer(args)
    for epoch in range(1, args.epoches + 1):
        logger.info(f"Epoch:{epoch}")
        cur_iter = 0
        for data in data_loader:
            # data shape: B N, N为序列长度
            data = data.long().transpose(1, 0).to(device)
            input_data = data[:-1, :]
            target = data[1:, :]
            # output shape: BN V_S, V_S表示词典大小
            if args.model_type == "lstm":
                output, hidden = model(input_data)
            elif args.model_type == "transformer":
                src_mask = model.generate_square_subsequent_mask(input_data.shape[0])
                src_pad_mask = input_data == len(word2ix) - 1
                src_pad_mask = src_pad_mask.permute(1, 0).contiguous()
                memory, output = model(input_data, src_mask.to(device), src_pad_mask.to(device))
            mask = target != word2ix['</s>']
            # print(word2ix['</s>'])
            target = target[mask]  # 去掉前缀的空格
            logit = output.flatten(0, 1)[mask.reshape(-1)]
            # loss = criterion(output.reshape(-1, vocab_size), target.reshape(-1))
            # print(logit.shape, target.shape)
            loss = criterion(logit, target)
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            global_iter += 1
            cur_iter += 1
            if global_iter % args.log_interval == 0:
                loss_avg.update(loss.item(), data.shape[0])
                logger.info(f"Epoch:{epoch}/{args.epoches} Iter:{cur_iter}/{len(data_loader)} "
                            f"Loss:{loss.item():.4f}({loss_avg.avg:.4f})")
        # 保存模型
        if epoch % args.save_interval == 0:
            save_path = os.path.join(args.save_path, f"{args.model_type}_Epoch_{epoch}.pth")
            state_dict = model.state_dict()
            torch.save(state_dict, save_path)
            logger.info(f"Save model to {save_path}")
            # 验证模型
        if args.model_type == "lstm":
            results = generate_poem_with_lstm(model=model, start_words="海上升明月",
                                              ix2word=ix2word, word2ix=word2ix,
                                              device=device, max_len=20)
        elif args.model_type == "transformer":
            results = generate_poem_with_transformer(model=model, start_words="海上升明月",
                                              ix2word=ix2word, word2ix=word2ix,
                                              device=device, max_len=20)
        logger.info(f"Generate poem:{results}")
        # print(output.shape)

if __name__ == '__main__':
    args = get_args()
    os.makedirs(args.save_path, exist_ok=True)
    dataloader, word2ix, ix2word = construct_loader(args)
    if args.model_type == "lstm":
        model = PoemModel_LSTM(len(word2ix), args.l_embed_dim, args.hidden_dim).to(device)
    elif args.model_type == "transformer":
        model = PoemModel_Trasformer(len(word2ix), args.t_embed_dim,
                                     num_encoder_layers=args.encoder_layer_num,
                                     dim_ffn=args.dim_ffn).to(device)
    train(model, data_loader=dataloader, vocab_size = len(word2ix), args = args)
