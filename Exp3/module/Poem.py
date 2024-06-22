# -*- encoding: utf-8 -*-
"""
File Poem.py
Created on 2024/5/15 11:58
Copyright (c) 2024/5/15
@author: 
"""
import math
import torch
import torch.nn as nn
from loguru import logger
from torch.nn import TransformerEncoderLayer, TransformerEncoder


class PoemModel_LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(PoemModel_LSTM, self).__init__()

        self.hidden_dim = hidden_dim
        # 每一个字都转化为一个一维向量，输出维度（词典大小×字向量维度）
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=1,
                            batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, input, hidden=None):
        # embeds: B N C, N表示序列长度
        embeds = self.embeddings(input)
        # logger.info(f"embed shape:{embeds.shape}")

        # [batch, seq_len] => [batch, seq_len, embed_dim]
        batch_size, seq_len = input.size()
        if hidden is None:
            h_0 = input.data.new(1, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(1, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        output, hidden = self.lstm(embeds, (h_0, c_0))
        output = self.linear(output)
        output = output.reshape(batch_size * seq_len, -1)
        return output, hidden


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, maxlen=200):
        super().__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2).float() * math.log(100) / emb_size)
        pos = torch.arange(0, maxlen).float().reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0),:])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class PoemModel_Trasformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_encoder_layers, dim_ffn, dropout=0.1):
        super().__init__()
        # 每一个字都转化为一个一维向量，输出维度（词典大小×字向量维度）
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 定义Transformer
        encoder_layer = TransformerEncoderLayer(d_model=embedding_dim, nhead=8, dim_feedforward=dim_ffn)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        # 定义位置编码器
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout=dropout)
        # 线性层输出需要和原始词典的字符编号范围对应
        self.predictor = nn.Linear(embedding_dim, vocab_size)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask, src_pad_mask):

        # 词嵌入 B L C
        src = self.embedding(src)
        # 增加位置信息
        src = self.positional_encoding(src)
        # [125, 32, 64], [32, 125], [125, 125]
        # print(f"after pos: src:{src.shape} padding mask:{src_pad_mask.shape} src_mask:{src_mask.shape}")

        memory = self.transformer_encoder(src, src_mask, src_pad_mask)
        logit = self.predictor(memory)
        return memory, logit

    def get_key_padding_mask(self, tokens, pad = 8292):
        # pad </s> id: 8292
        key_padding_mask = torch.zeros(tokens.size())
        key_padding_mask[tokens == pad] = float('-inf')
        return key_padding_mask

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)  # 生成下三角矩阵（下三角全为True，其余位False）
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
