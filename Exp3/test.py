# -*- encoding: utf-8 -*-
"""
File test.py
Created on 2024/5/15 21:42
Copyright (c) 2024/5/15
@author: 
"""
import torch as t
from generate_poem import *

from module.Poem import PoemModel_LSTM
from hyper_para import get_args
import numpy as np

# 保证结果可重复
torch.manual_seed(123243)
# 检查CUDA是否可用
if torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info("CUDA is available! Training on GPU.")
else:
    device = torch.device("cpu")
    logger.info("CUDA is not available. Training on CPU.")


def userTest():
    args = get_args()
    print("正在初始化......")
    poem_data = PoemDataset(args)
    ix2word = poem_data.ix2word
    word2ix = poem_data.word2ix
    if args.model_type == "lstm":
        model = PoemModel_LSTM(len(ix2word), args.l_embed_dim, args.hidden_dim).to(device)
    elif args.model_type == "transformer":
        model = PoemModel_Trasformer(len(poem_data.word2ix), args.t_embed_dim,
                                     num_encoder_layers=args.encoder_layer_num,
                                     # num_decoder_layers=args.decoder_layer_num,
                                     dim_ffn=args.dim_ffn).to(device)
    model.load_state_dict(torch.load(args.ckt_path))
    print("初始化完成！\n")
    while True:
        print("欢迎使用唐诗生成器，\n"
              "输入1 进入首句生成模式\n"
              "输入2 进入藏头诗生成模式\n")
        mode = int(input())
        if mode == 1:
            print("请输入您想要的诗歌首句，可以是五言或七言")
            start_words = str(input())
            if args.model_type == "lstm":
                gen_poetry = ''.join(generate_poem_with_lstm(model, start_words, ix2word, word2ix, device=device, max_len=30))
            elif args.model_type == "transformer":
                gen_poetry = ''.join(
                    generate_poem_with_transformer(model, start_words, word2ix, ix2word, device=device, max_len=20))
            print("生成的诗句如下：%s\n" % (gen_poetry))
        elif mode == 2:
            print("请输入您想要的诗歌藏头部分，不超过16个字，最好是偶数")
            start_words = str(input())
            if args.model_type == "lstm":
                gen_poetry = ''.join(gen_acrostic_with_lstm(model, start_words, ix2word, word2ix, device=device, max_len=30))
            elif args.model_type == "transformer":
                print("暂时未完成")
            print("生成的诗句如下：%s\n" % (gen_poetry))


if __name__ == '__main__':
    userTest()