# -*- encoding: utf-8 -*-
"""
File prepare.py
Created on 2024/6/15 20:21
Copyright (c) 2024/6/15
@author: 
"""
import os.path

import torch
# 使用huggingface的分词器
from tokenizers import Tokenizer
# 用于构建词典
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm

def get_row_count(filepath):
    count = 0
    for _ in open(filepath, encoding='utf-8'):
        count += 1
    return count

def en_tokenizer(tokenizer, line):
    """
        定义英文分词器
    """
    # 使用bert进行分词，并获取tokens。add_special_tokens是指不要在结果中增加‘<bos>’和`<eos>`等特殊字符
    return tokenizer.encode(line, add_special_tokens=False).tokens

def yield_en_tokens(tokenizer, en_filepath, row_count):
    file = open(en_filepath, encoding='utf-8')
    print("-------开始构建英文词典-----------")
    for line in tqdm(file, desc="构建英文词典", total=row_count):
        yield en_tokenizer(tokenizer, line)
    file.close()


def zh_tokenizer(line):
    """
    定义中文分词器
    :param line: 中文句子，例如：机器学习
    :return: 分词结果，例如['机','器','学','习']
    """
    return list(line.strip().replace(" ", ""))

def yield_zh_tokens(zh_filepath, row_count):
    file = open(zh_filepath, encoding='utf-8')
    for line in tqdm(file, desc="构建中文词典", total=row_count):
        yield zh_tokenizer(line)
    file.close()

def create_vocab(file_path, save_path, is_en = True, use_cache = True):
    row_count = get_row_count(file_path)
    if use_cache and os.path.exists(save_path):
        vocab = torch.load(save_path, map_location="cpu")
        return vocab
    if is_en:
        # 返回数字和文本对应的字典
        vocab = build_vocab_from_iterator(
            # 传入一个可迭代的token列表。例如[['i', 'am', ...], ['machine', 'learning', ...], ...]
            yield_en_tokens(tokenizer, file_path, row_count),
            # 最小频率为2，一个单词最少出现两次才会被收录到词典
            min_freq = 2,
            # 在词典的最开始加上这些特殊token，即1-4个token是它们
            specials=["<s>", "</s>", "<pad>", "<unk>"],
        )
    else:
        # 返回数字和文本对应的字典
        vocab = build_vocab_from_iterator(
            yield_zh_tokens(file_path, row_count),
            min_freq=1,
            specials=["<s>", "</s>", "<pad>", "<unk>"],
        )
    # 设置默认index，后面文本转index时，如果找不到，就会用该index填充
    vocab.set_default_index(vocab["<unk>"])
    # 保存缓存文件
    torch.save(vocab, save_path)
    return vocab

def load_tokens(save_path, file_path, vocab, desc, lang):
    """
    加载tokens，即将文本转换成index。
    :return: 返回构造好的tokens。例如：[[6, 8, 93, 12, ..], [62, 891, ...], ...]
    """

    # 定义缓存文件存储路径
    cache_file =  os.path.join(save_path, "tokens_list.{}.pt".format(lang))
    # 如果使用缓存，且缓存文件存在，则直接加载
    if os.path.exists(cache_file):
        print(f"正在加载缓存文件{cache_file}, 请稍后...")
        return torch.load(cache_file, map_location="cpu")

    # 从0开始构建，定义tokens_list用于存储结果
    tokens_list = []
    row_count = get_row_count(file_path)
    # 打开文件
    with open(file_path, encoding='utf-8') as file:
        # 逐行读取
        for line in tqdm(file, desc=desc, total=row_count):
            # 进行分词
            if lang == "en":
                tokens = en_tokenizer(tokenizer, line)
            else:
                tokens = zh_tokenizer(line)
            # 将文本分词结果通过词典转成index
            tokens = vocab(tokens)
            # append到结果中
            tokens_list.append(tokens)
    torch.save(tokens_list, cache_file)

    return tokens_list

def load_tokens_from_val_data(en_vocab, zh_vocab, val_file, save_path):
    row_count = get_row_count(val_file)
    count = 0
    pre = 0
    # 打开文件
    zh_tokens_list = []
    en_tokens_list = []
    with open(val_file, encoding='utf-8') as file:
        for line in tqdm(file, desc="读取验证集", total=row_count):
            if count % 3 == 0:
                # 中文
                tokens = zh_tokenizer(line)
                tokens_id = zh_vocab(tokens)
                zh_tokens_list.append(tokens_id)
                pre = count
            if count - pre == 2:
                print(line)
                # 英文
                tokens = en_tokenizer(tokenizer, line)
                tokens_id = en_vocab(tokens)
                en_tokens_list.append(tokens_id)
            count += 1
    print(f"English lines:{len(en_tokens_list)}")
    print(f"Chinese lines:{len(zh_tokens_list)}")
    en_tokens_file = os.path.join(save_path, "val_en_data.pt")
    torch.save(en_tokens_list, en_tokens_file)
    zh_tokens_file = os.path.join(save_path, "val_zh_data.pt")
    torch.save(zh_tokens_list, zh_tokens_file)
    print("save!")

def prepare_val_data():
    # tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    save_root = "/data2/gaoxingyu/Deep_Learning/saves/Exp4/"
    data_root = "/data2/gaoxingyu/Deep_Learning/data/sample-submission-version/"
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    en_vocab_file = os.path.join(save_root, "vocab_en.pt")
    en_filepath = os.path.join(data_root, "TM-training-set", "english.txt")
    en_vocab = create_vocab(en_filepath, en_vocab_file, is_en=True, use_cache=True)
    zh_vocab_file = os.path.join(save_root, "vocab_zh.pt")
    zh_filepath = os.path.join(data_root, "TM-training-set", "chinese.txt")
    zh_vocab = create_vocab(zh_filepath, zh_vocab_file, is_en=False, use_cache=True)
    val_file = os.path.join(data_root, "Dev-set", "Niu.dev.txt")
    load_tokens_from_val_data(en_vocab, zh_vocab, val_file, save_root)

if __name__ == '__main__':
    # 加载bert分词器，不区分大小写
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    # save_root = "/data2/gaoxingyu/Deep_Learning/saves/Exp4/"
    # data_root = "/data2/gaoxingyu/Deep_Learning/data/sample-submission-version/"
    # tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    # type = "zh"
    # use_cache = True
    # if type == "en":
    #     # 英文文件路径
    #     en_vocab_file = os.path.join(save_root, "vocab_en.pt")
    #     en_filepath = os.path.join(data_root, "TM-training-set", "english.txt")
    #     en_vocab = create_vocab(en_filepath, en_vocab_file, is_en=True, use_cache=use_cache)
    #     print("英文词典大小:", len(en_vocab))
    #     print(dict((i, en_vocab.lookup_token(i)) for i in range(10)))
    #     load_tokens(save_root, en_filepath, en_vocab, "英文token", "en")
    # # 中文文件路径
    # else:
    #     zh_vocab_file = os.path.join(save_root, "vocab_zh.pt")
    #     zh_filepath = os.path.join(data_root, "TM-training-set", "chinese.txt")
    #     zh_vocab = create_vocab(zh_filepath, zh_vocab_file, is_en=False, use_cache=use_cache)
    #     print("中文词典大小:", len(zh_vocab))
    #     print(dict((i, zh_vocab.lookup_token(i)) for i in range(10)))
    #     tokens_list = load_tokens(save_root, zh_filepath, zh_vocab, "中文token", "zh")
    #     print(len(tokens_list))

    prepare_val_data()
