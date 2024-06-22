# -*- encoding: utf-8 -*-
"""
File utils.py
Created on 2024/6/19 19:10
Copyright (c) 2024/6/19
@author: 
"""
import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def subsequent_mask(size):
    """
    由于transformer的Decoder是采用的自回归的机制，预测t的时候只能看见1到t-1部分的词，不能看见t+1之后的词
    所以需要将t+1后面的词给mask掉
    本函数根据size产生对应的mask矩阵
    :param size:
    :return:
    """
    attention_shape = (1, size, size)
    # 产生一个上三角矩阵(除了上三角位置的数据保留其他所有位置的数置为0)，k=1表示对角线也为0
    subsequent_mask = np.triu(np.ones(attention_shape), k=1).astype('uint8')
    # 通过==0的操作，下三角全置为1
    return torch.from_numpy(subsequent_mask) == 0

def make_std_mask(tgt, pad):
    "Create a mask to hide padding and future words."
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & torch.tensor(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask

def compute_bleu(translate, reference, references_lens):
    """
    计算翻译句子的的BLEU值
    :param translate: transformer翻译的句子
    :param reference: 标准译文
    :return: BLEU值
    """
    # 定义平滑函数
    translate = translate.tolist()
    reference = reference.tolist()
    smooth = SmoothingFunction()
    references_lens = references_lens.tolist()
    blue_score = []
    for translate_sentence, reference_sentence, references_len in zip(translate, reference, references_lens):
        if 1 in translate_sentence:
            index = translate_sentence.index(1)
        else:
            index = len(translate_sentence)
        blue_score.append(sentence_bleu([reference_sentence[:references_len]], translate_sentence[:index], weights=(0.3, 0.4, 0.3, 0.0), smoothing_function=smooth.method1))
    return blue_score