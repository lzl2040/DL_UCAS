# -*- encoding: utf-8 -*-
"""
File tokenizer.py
Created on 2024/5/17 11:22
Copyright (c) 2024/5/17
@author: 
"""
import numpy as np


class Tokenizer:
    def __init__(self, args):
        super().__init__()
        poem = np.load(args.data_path, allow_pickle=True)
        # 字典类型：id: 对应的字，如 1: "你"
        self.ix2word = poem['ix2word'].item()
        # 字典类型(8293个)： 某个字： 对应的id 如："你":1
        # 字典里面的填充符:8290 : '<EOP>', 8291: '<START>', 8292: '</s>'
        self.word2ix = poem['word2ix'].item()

        self.start_id = self.word2ix['<START>']
        self.end_id = self.word2ix['<EOP>']
        self.pad_id = self.word2ix['</s>']

    def encode(self, tokens):
        token_ids = []
        for token in tokens:
            token_ids.append(self.word2ix[token])
        return token_ids

    def decode(self, token_ids):
        tokens = []
        for idx in token_ids:
            # 跳过起始、结束标记
            if idx != self.start_id and idx != self.end_id:
                tokens.append(self.ix2word[idx])
        return tokens