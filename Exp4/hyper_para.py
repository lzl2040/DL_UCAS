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
    parser.add_argument("--epochs", type=int, default=50, help="epoches for training")
    parser.add_argument("--log-interval", type=int, default=5, help="the interval(iter) of logging")
    parser.add_argument("--save-interval", type=int, default=5, help="the interval(epoch) of saving the model")
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--save-path", type=str, default="/data2/gaoxingyu/Deep_Learning/saves/Exp4",
                        help="the path to save the weights")
    # 训练参数
    parser.add_argument("--batch-size", type=int, default=128, help="batch size for training")
    # parser.add_argument("--val_bs", ty)
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate 1e-3")
    parser.add_argument("--max_steps", type=int, default=50 * 782)
    parser.add_argument("--warmup_step", type=int, default=5 * 782)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    # Transformer模型
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--encoder_layer_num", type=int, default=4)
    parser.add_argument("--decoder_layer_num", type=int, default=3)
    parser.add_argument("--dim_ffn", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--head_num", type=int, default=8)

    parser.add_argument("--model_type", type=str, default="lstm", help="transformer, lstm")
    parser.add_argument("--exp_name", type=str, default="transformer_trans")
    # 模型加载
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--ckt_path", type=str, default= "/data2/gaoxingyu/Deep_Learning/saves/Exp4/transformer_trans/epoch_30.pth")
    # 数据集
    parser.add_argument("--weight_path", type=str, default="/data2/gaoxingyu/Deep_Learning/saves/Exp4/")
    parser.add_argument("--max_length", type=int, default=72)
    parser.add_argument("--pad_id", type=int, default=2)
    args = parser.parse_args()
    return args