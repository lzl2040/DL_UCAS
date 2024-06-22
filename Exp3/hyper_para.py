# -*- encoding: utf-8 -*-
"""
File hyper_para.py
Created on 2024/5/16 16:49
Copyright (c) 2024/5/16
@author: 
"""
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="defination of some parameters")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--data_path", type=str, default="data/tang.npz")
    parser.add_argument("--epoches", type=int, default=100, help="epoches for training")
    parser.add_argument("--log-interval", type=int, default=5, help="the interval(iter) of logging")
    parser.add_argument("--save-interval", type=int, default=10, help="the interval(epoch) of saving the model")
    parser.add_argument("--save-path", type=str, default="/data2/gaoxingyu/Deep_Learning/saves/Exp3",
                        help="the path to save the weights")
    # 训练参数
    parser.add_argument("--batch-size", type=int, default=128, help="batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate 1e-3")
    # LSTM模型
    parser.add_argument("--l_embed_dim", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=64)
    # Transformer模型
    parser.add_argument("--t_embed_dim", type=int, default=512)
    parser.add_argument("--encoder_layer_num", type=int, default=4)
    parser.add_argument("--decoder_layer_num", type=int, default=3)
    parser.add_argument("--dim_ffn", type=int, default=1024)

    parser.add_argument("--model_type", type=str, default="transformer", help="transformer, lstm")
    # 模型加载
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--ckt_path", type=str, default= "/data2/gaoxingyu/Deep_Learning/saves/Exp3/lstm_Epoch_20.pth")
    args = parser.parse_args()
    return args