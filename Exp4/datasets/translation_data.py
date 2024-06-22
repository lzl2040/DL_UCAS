# -*- encoding: utf-8 -*-
"""
File translation_data.py
Created on 2024/6/15 20:19
Copyright (c) 2024/6/15
@author: 
"""
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import pad, log_softmax

# from Exp4.datasets.prepare import en_tokenizer, zh_tokenizer


# def collate_fn(batch):
#     """
#     将dataset的数据进一步处理，并组成一个batch。
#     :param batch: 一个batch的数据，例如：
#                   [([6, 8, 93, 12, ..], [62, 891, ...]),
#                   ....
#                   ...]
#     :return: 填充后的且等长的数据，包括src, tgt, tgt_y, n_tokens
#              其中src为原句子，即要被翻译的句子
#              tgt为目标句子：翻译后的句子，但不包含最后一个token
#              tgt_y为label：翻译后的句子，但不包含第一个token，即<bos>
#              n_tokens：tgt_y中的token数，<pad>不计算在内。
#     """
#
#     # 定义'<bos>'的index，在词典中为0，所以这里也是0
#     bs_id = torch.tensor([0])
#     # 定义'<eos>'的index
#     eos_id = torch.tensor([1])
#     # 定义<pad>的index
#     pad_id = 2
#
#     # 用于存储处理后的src和tgt
#     src_list, tgt_list = [], []
#
#     # 循环遍历句子对
#     for (_src, _tgt, max_length) in batch:
#         """
#         _src: 英语句子，例如：`I love you`对应的index
#         _tgt: 中文句子，例如：`我 爱 你`对应的index
#         """
#         # 将<bos>，句子index和<eos>拼到一块
#         processed_src = torch.cat([bs_id, torch.tensor(_src, dtype=torch.int64), eos_id], dim = 0)
#         processed_tgt = torch.cat([bs_id, torch.tensor(_tgt, dtype=torch.int64), eos_id,], dim = 0)
#
#         # 将长度不足的句子进行填充到max_padding的长度的，然后增添到list中
#         # 使用pad_id填充processed_src的左右两边，如果原先是[a,b]，pad第二个参数中第一个值表示左边填充数目，第二个表示右边填充数目
#         src_list.append(pad(processed_src, (0, max_length - len(processed_src)), value=pad_id))
#         tgt_list.append(pad(processed_tgt, (0, max_length - len(processed_tgt)), value=pad_id))
#
#     # 将多个src句子堆叠到一起
#     src = torch.stack(src_list)
#     tgt = torch.stack(tgt_list)
#
#     # tgt_y是目标句子去掉第一个token，因为实际还是是预测下一个token
#     tgt_y = tgt[:, 1:]
#     # tgt是目标句子去掉最后一个token
#     tgt = tgt[:, :-1]
#
#     # 计算本次batch要预测的token数
#     n_tokens = (tgt_y != 2).sum()
#     # 返回batch后的结果
#     return src, tgt, tgt_y, n_tokens

class TranslationDataset(Dataset):
    def __init__(self, weight_path, max_len, mode, target = "en_to_zh"):
        super().__init__()
        # en_vocab_path = os.path.join(vocab_path, "vocab_en.pt")
        # self.en_vocab = torch.load(en_vocab_path)
        # zh_vocab_path = os.path.join(vocab_path, "vocab_zh.pt")
        # self.zh_vocab = torch.load(zh_vocab_path)
        self.target = target
        en_tokens_path = os.path.join(weight_path, f"{mode}_tokens_list.en.pt")
        self.en_tokens = torch.load(en_tokens_path, map_location="cpu")
        zh_tokens_path = os.path.join(weight_path, f"{mode}_tokens_list.zh.pt")
        self.zh_tokens = torch.load(zh_tokens_path, map_location="cpu")
        assert len(self.en_tokens) == len(self.zh_tokens)
        self.max_len = max_len
        # start id
        self.bs_id = 0
        # 定义'<eos>'的index
        self.eos_id = 1
        # 为了保证句子等长，使用pad来填充
        self.pad_id = 2

    def process_token_id(self, src, tgt):
        bs_id = torch.tensor([0])
        eos_id = torch.tensor([1])
        # 拼接到两端，形成 <s> content </s>
        src = torch.cat([bs_id, torch.tensor(src, dtype=torch.int64), eos_id], dim = 0)
        tgt = torch.cat([bs_id, torch.tensor(tgt, dtype=torch.int64), eos_id, ], dim=0)

        if src.shape[0] < self.max_len:
            src = pad(src, (0, self.max_len - len(src)), value=self.pad_id)
            # tgt = pad(tgt, (0, self.max_len - len(tgt)), value=self.pad_id)
            # print(f"Less: {src.shape}, {tgt.shape}")
        if tgt.shape[0] < self.max_len:
            tgt = pad(tgt, (0, self.max_len - len(tgt)), value=self.pad_id)
        if src.shape[0] >= self.max_len and tgt.shape[0] >= self.max_len:
            # print(f"{src.shape}, {tgt.shape}")
            src = src[:self.max_len]
            tgt = tgt[:self.max_len]
            # print(f"More: {src.shape}, {tgt.shape}")

        # 预测下一个词，所以从第2个token开始
        tgt_y = tgt[1:]
        # 
        tgt = tgt[:-1]
        n_tokens = (tgt_y != 2).sum()
        return src, tgt, tgt_y, n_tokens

    def __getitem__(self, idx):
        if self.target == "en_to_zh":
            src, tgt, tgt_y, n_tokens = self.process_token_id(self.en_tokens[idx], self.zh_tokens[idx])
            return src, tgt, tgt_y, n_tokens
        else:
            src, tgt, tgt_y, n_tokens = self.process_token_id(self.zh_tokens[idx], self.en_tokens[idx])
            return src, tgt, tgt_y, n_tokens

    def __len__(self):
        return len(self.zh_tokens)


if __name__ == '__main__':
    weight_path = "/data2/gaoxingyu/Deep_Learning/saves/Exp4/"
    en_vocab_path = os.path.join(weight_path, "vocab_en.pt")
    zh_vocab_path = os.path.join(weight_path, "vocab_zh.pt")
    dataset = TranslationDataset(weight_path, max_len=72)
    src, tgt, tgt_y, n_tokens = dataset.__getitem__(0)
    en_vocab = torch.load(en_vocab_path, map_location="cpu")
    zh_vocab = torch.load(zh_vocab_path, map_location="cpu")
    print(len(en_vocab), len(zh_vocab))
    print(len(src), len(tgt))
    print("en:")
    for e_id in src:
        print(en_vocab.lookup_token(e_id), end=" ")
    print("\nzh:")
    for z_id in tgt:
        print(zh_vocab.lookup_token(z_id), end="")
    print("\n")

    # 保证结果可重复
    torch.manual_seed(123243)
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # logger.info("CUDA is available! Training on GPU.")
    else:
        device = torch.device("cpu")
        # logger.info("CUDA is not available. Training on CPU.")

    max_length = 72
    batch_size = 128
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"data size:{len(train_loader)}")
    for src, tgt, tgt_y, n_tokens in train_loader:
        # src, tgt, tgt_y, n_tokens = next(iter(train_loader))
        src, tgt, tgt_y = src.to(device), tgt.to(device), tgt_y.to(device)
        # src, tgt, max_len = next(iter(train_loader))
        # src, tgt =  src.to(device), tgt.to(device)
        print("src.size:", src.size())
        print("tgt.size:", tgt.size())
        print("tgt_y.size:", tgt_y.size())
        print("n_tokens:", n_tokens)
