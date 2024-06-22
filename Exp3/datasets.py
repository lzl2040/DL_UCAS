# -*- encoding: utf-8 -*-
"""
File datasets.py
Created on 2024/5/15 12:20
Copyright (c) 2024/5/15
@author: 
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import random
def print_poem(data, pos, ix2word):
    p_len = data.shape[1]
    for i in range(p_len):
        id = data[pos, i]
        # 填充符的位置 <START> 开始标识 <EOP> 结束标识 </s>填充符
        # 诗数据的内容：</s> </s> <START> 诗的内容 <EOP>
        if id < 8290:
            print(ix2word[id], end = "")

class PoemDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        poem = np.load(args.data_path, allow_pickle=True)
        # 字典类型：id: 对应的字，如 1: "你"
        self.ix2word = poem['ix2word'].item()
        # 字典类型(8293个)： 某个字： 对应的id 如："你":1
        # 字典里面的填充符:8290 : '<EOP>', 8291: '<START>', 8292: '</s>'
        self.word2ix = poem['word2ix'].item()
        # 填充符的id
        self.start_id = self.word2ix["<START>"]
        self.pad_id= self.word2ix["</s>"]
        self.end_id = self.word2ix["<EOP>"]
        # [57580, 125]，存储的是唐诗每个字对应的id
        self.data = poem['data']
        # print_poem(data, 1, self.ix2word)

    def reverse_data(self, poem):
        # 因为原始数据是 </s>在前面，不符合我们的习惯，所以把它放在后面
        # 找到<START>的位置
        ind = np.argwhere(poem == self.start_id).item()
        # 诗句的主体
        new_poem = poem[ind : len(poem)]
        # </s>
        pad = poem[0 : ind]
        # 返回拼接的
        return np.hstack((new_poem, pad))


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, id):
        poem = self.data[id]
        poem = self.reverse_data(poem)
        # input_poem = poem[:-1]
        # target_poem = poem[1:]
        # return input_poem, target_poem
        return poem

if __name__ == '__main__':
    print(123)
