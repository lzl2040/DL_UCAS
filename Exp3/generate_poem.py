# -*- encoding: utf-8 -*-
"""
File generate_poem.py
Created on 2024/5/15 21:49
Copyright (c) 2024/5/15
@author: 
"""
import argparse

import torch

from tokenizer import Tokenizer
from datasets import PoemDataset
from module.Poem import PoemModel_LSTM, PoemModel_Trasformer
from loguru import logger
from hyper_para import get_args
import random

# 保证结果可重复
torch.manual_seed(123243)
# 检查CUDA是否可用
if torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info("CUDA is available! Training on GPU.")
else:
    device = torch.device("cpu")
    logger.info("CUDA is not available. Training on CPU.")

def generate_poem_with_lstm(model, start_words, ix2word, word2ix, device, max_len, prefix_words=None):
    results = list(start_words)
    start_words_len = len(start_words)
    # 第一个词语是<START>
    input = torch.Tensor([word2ix['<START>']]).view(1, 1).long().to(device)
    hidden = None

    # 若有风格前缀，则先用风格前缀生成hidden
    if prefix_words:
        # 第一个input是<START>，后面就是prefix中的汉字
        # 第一个hidden是None，后面就是前面生成的hidden
        for word in prefix_words:
            output, hidden = model(input, hidden)
            input = input.data.new([word2ix[word]]).view(1, 1)

    # 开始真正生成诗句，如果没有使用风格前缀，则hidden = None，input = <START>
    # 否则，input就是风格前缀的最后一个词语，hidden也是生成出来的
    for i in range(max_len):
        output, hidden = model(input, hidden)
        # print(output.shape)
        # 如果还在诗句内部，输入就是诗句的字，不取出结果，只为了得到
        # 最后的hidden
        if i < start_words_len:
            w = results[i]
            input = input.data.new([word2ix[w]]).view(1, 1)
            # print(f"{i} < L: word:{w}")
        # 否则将output作为下一个input进行
        else:
            # print(output.data[0].topk(1))
            top_index = output.data[0].topk(1)[1][0].item()
            w = ix2word[top_index]
            # print(f"{i} >= L:top_index:{top_index} word:{w}")
            results.append(w)
            input = input.data.new([top_index]).view(1, 1)
        if w == '<EOP>':
            del results[-1]
            break
    return results

# 生成藏头诗
def gen_acrostic_with_lstm(model, start_words, ix2word, word2ix, device, max_len, prefix_words=None):
    result = []
    start_words_len = len(start_words)
    input = (torch.Tensor([word2ix['<START>']]).view(1, 1).long()).to(device)
    # 指示已经生成了几句藏头诗
    index = 0
    pre_word = '<START>'
    hidden = None

    # 存在风格前缀，则生成hidden
    if prefix_words:
        for word in prefix_words:
            output, hidden = model(input, hidden)
            input = (input.data.new([word2ix[word]])).view(1, 1)

    # 开始生成诗句
    for i in range(max_len):
        output, hidden = model(input, hidden)
        top_index = output.data[0].topk(1)[1][0].item()
        w = ix2word[top_index]
        # print(f"top index:{top_index} word:{w} pre_word:{pre_word}")
        # 说明上个字是句末
        if pre_word in {'。', '，', '?', '！', '<START>'}:
            if index == start_words_len:
                break
            else:
                w = start_words[index]
                index += 1
                # print(w,word2ix[w])
                input = (input.data.new([word2ix[w]])).view(1, 1)
        else:
            input = (input.data.new([top_index])).view(1, 1)
        result.append(w)
        pre_word = w
    return result


def generate_poem_with_transformer(model, start_words, word2ix, ix2word, device, max_len=100):
    model.eval()

    src = [word2ix[word] for word in start_words]
    res = [word2ix['<START>']] + src

    for _ in range(max_len):
        src = torch.tensor(res).to(device)[:, None]
        src_mask = model.generate_square_subsequent_mask(src.shape[0])
        src_pad_mask = src == len(word2ix) - 1
        src_pad_mask = src_pad_mask.permute(1, 0).contiguous()
        memory, logits = model(src, src_mask.cuda(), src_pad_mask.cuda())

        next_word = logits[-1, 0].argmax().item()
        if next_word == word2ix['<EOP>']:
            break
        res.append(next_word)

        if next_word == word2ix['<EOP>']:
            break
    res = [ix2word[_] for _ in res]
    model.train()
    return res


if __name__ == '__main__':
    weight_path = "/data2/gaoxingyu/Deep_Learning/saves/Exp3/transformer_Epoch_2.pth"
    state_dict = torch.load(weight_path)
    args = get_args()
    poem_data = PoemDataset(args)
    # model = PoemModel_LSTM(len(poem_data.word2ix), args.embed_dim, args.hidden_dim).to(device)
    model = PoemModel_Trasformer(len(poem_data.word2ix), args.embed_dim,
                                 num_encoder_layers=args.encoder_layer_num,
                                 # num_decoder_layers=args.decoder_layer_num,
                                 dim_ffn=args.dim_ffn).to(device)
    model.load_state_dict(state_dict)
    logger.info("model loaded")
    # 验证模型
    # results = generate_poem_with_lstm(model=model, start_words="床前明月光",
    #                                   ix2word=poem_data.ix2word, word2ix=poem_data.word2ix,
    #                                   device=device, max_len=20)
    tokenizer = Tokenizer(args)
    results = generate_poem_with_transformer(model=model, start_words="海上升明月",
                                             ix2word=poem_data.ix2word, word2ix=poem_data.word2ix,
                                             device=device, max_len=20)
    logger.info(f"Generate poem:{results}")
