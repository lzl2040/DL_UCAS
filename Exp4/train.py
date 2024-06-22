# -*- encoding: utf-8 -*-
"""
File train.py
Created on 2024/6/15 20:19
Copyright (c) 2024/6/15
@author: 
"""
import os

import warmup_scheduler
from tokenizers import Tokenizer
from torch import nn, log_softmax
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import make_std_mask, compute_bleu
from datasets.prepare import en_tokenizer
from datasets.translation_data import TranslationDataset
from module.translation_model import TranslateEn2Zh
from hyper_para import get_args
import torch
from loguru import logger
import wandb
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

# 分词器
tokenizer = Tokenizer.from_pretrained("bert-base-uncased")

def translate(src, args):
    """
    :param src: 英文句子，例如 "I like machine learning."
    :return: 翻译后的句子，例如：”我喜欢机器学习“
    """

    # 将与原句子分词后，通过词典转为index，然后增加<bos>和<eos>
    src = torch.tensor([0] + en_vocab(en_tokenizer(tokenizer, src)) + [1]).unsqueeze(0).to(device)
    # 首次tgt为<bos>
    tgt = torch.tensor([[0]]).to(device)
    # 一个一个词预测，直到预测为<eos>，或者达到句子最大长度
    for i in range(args.max_length):
        # 进行transformer计算
        out = model(src, tgt)
        # 预测结果，因为只需要看最后一个词，所以取`out[:, -1]`
        predict = model.predictor(out[:, -1])
        # 找出最大值的index
        y = torch.argmax(predict, dim=1)
        # 和之前的预测结果拼接到一起
        tgt = torch.concat([tgt, y.unsqueeze(0)], dim=1)
        # 如果为<eos>，说明预测结束，跳出循环
        if y == 1:
            break
    # 将预测tokens拼起来
    tgt = ''.join(zh_vocab.lookup_tokens(tgt.squeeze().tolist())).replace("<s>", "").replace("</s>", "")
    return tgt

if __name__ == '__main__':
    args = get_args()
    os.makedirs(args.save_path, exist_ok=True)

    wandb.init(
        # Set the project where this run will be logged
        project="Translation",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"experiment_2",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": args.lr,
            "architecture": "Transformer",
            "dataset": "translation",
            "epochs": args.epochs,
        })

    # 加载vocab
    en_vocab_path = os.path.join(args.weight_path, "vocab_en.pt")
    zh_vocab_path = os.path.join(args.weight_path, "vocab_zh.pt")
    en_vocab = torch.load(en_vocab_path, map_location="cpu")
    zh_vocab = torch.load(zh_vocab_path, map_location="cpu")
    model = TranslateEn2Zh(args, enVocabLen=len(en_vocab), zhVocabLen=len(zh_vocab)).to(device)

    # dataset
    train_dataset = TranslationDataset(args.weight_path, max_len=args.max_length, mode="train")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = TranslationDataset(args.weight_path, max_len=args.max_length, mode="val")
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    # src, tgt, tgt_y, n_tokens = next(iter(train_loader))
    # src, tgt, tgt_y = src.to(device), tgt.to(device), tgt_y.to(device)
    # print(model(src, tgt).size())
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps, eta_min=args.min_lr)
    scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1., total_epoch=args.warmup_step,
                                                        after_scheduler=base_scheduler)
    criterion = nn.NLLLoss()
    # 训练
    padding_idx = 2
    step = 0
    model.train()
    for epoch in range(args.epochs):
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        cur_step = 0
        for index, data in enumerate(train_loader):
            # 生成数据
            src, tgt, tgt_y, n_tokens = data
            src, tgt, tgt_y = src.to(device), tgt.to(device), tgt_y.to(device)
            # 清空梯度
            optimizer.zero_grad()
            src_mask = (src != args.pad_id).unsqueeze(-2)
            dst_mask = make_std_mask(tgt, args.pad_id)
            out = model(english_seq=src, english_mask=src_mask, chinese_seq=tgt, chinese_mask=dst_mask)
            output = model.generator(out)

            output = output.view(-1, output.size(-1))
            # output.shape=(batch_size * seq_len, zhVocabLen)
            tgt_y = tgt_y.view(-1)
            # chineseSeqY.shape=(baych_size * seq_len, )
            loss = criterion(output, tgt_y)

            # 计算梯度
            loss.backward()
            # 更新参数
            optimizer.step()
            scheduler.step()

            loop.set_description("Epoch {}/{}".format(epoch, args.epochs))
            loop.set_postfix(loss=loss.item())
            loop.update(1)

            step += 1
            cur_step += 1

            del src
            del tgt
            del tgt_y

            if step % args.log_interval == 0:
                # logger.info(f"Epoch:{epoch}/{args.epochs} Step:{cur_step}/{len(train_loader)} Loss:{loss.item():.4f}")
                wandb.log({"loss": loss})
                wandb.log({"Lr": optimizer.param_groups[0]['lr']})

        if epoch % args.save_interval == 0:
            save_path = os.path.join(args.save_path, args.exp_name)
            os.makedirs(save_path, exist_ok=True)
            file_path = os.path.join(save_path, f"epoch_{epoch}.pth")
            # 保存ckt
            state_dict = {
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "weight": model.state_dict(),
            }
            torch.save(state_dict, file_path)
            logger.info(f"Save model to {file_path}")

        if epoch % args.val_interval == 0:
            # 验证
            with torch.no_grad():
                blue_socres = []
                for valid_index, (englishSeq, chineseSeq, chineseSeqY, chineseSeqY_lens) in enumerate(
                        tqdm(val_loader)):
                    # englishSeq.shape=(batch_size, paded_seq_len)
                    # chineseSeq.shape=(batch_size, paded_seq_len)
                    # chineseSeqY.shape=(batch_size, padded_seq_len)
                    englishSeq = englishSeq.to(device)
                    chineseSeq = chineseSeq.to(device)
                    chineseSeqY = chineseSeqY.to(device)
                    src_mask = (englishSeq != args.pad_id).unsqueeze(-2)
                    # src_mask.shape=(batch_size, 1, seq_len)
                    memory = model.encode(englishSeq, src_mask)
                    # memory.shape=(batch_size, seq_len, embedding_dim)
                    translate = torch.ones(args.batch_size, 1).fill_(0).type_as(englishSeq.data)
                    # translate_ = chineseSeqY[:, 0]
                    # ys.shape=(1, 1)
                    for i in range(args.max_length):
                        translate_mask = make_std_mask(translate, args.pad_id)
                        # print(memory.shape, src_mask.shape, translate.shape, translate_mask.shape)
                        out = model.decode(memory, src_mask, translate, translate_mask)
                        prob = model.generator(out[:, -1])
                        _, next_word = torch.max(prob, dim=1)
                        next_word = next_word.unsqueeze(1)
                        translate = torch.cat([translate, next_word], dim=1)
                        # translate_ = chineseSeqY[:, :]
                    blue_socres += compute_bleu(translate, chineseSeqY, chineseSeqY_lens)
                    if (valid_index + 1) % 1 == 0:
                        reference_sentence = chineseSeqY[0].tolist()
                        translate_sentence = translate[0].tolist()
                        englishSeq_sentence = englishSeq[0].tolist()
                        reference_sentence_len = chineseSeqY_lens.tolist()[0]
                        if 1 in translate_sentence:
                            index = translate_sentence.index(1)
                        else:
                            index = len(translate_sentence)
                        print("原文: {}".format(" ".join([en_vocab.lookup_token(x) for x in englishSeq_sentence])))
                        print("机翻译文: {}".format("".join([zh_vocab.lookup_token(x) for x in translate_sentence[:index]])))
                        print("参考译文: {}".format(
                            "".join([zh_vocab.lookup_token(x) for x in reference_sentence[:reference_sentence_len]])))
                        print("原文: {}".format(" ".join([en_vocab.lookup_token(x) for x in englishSeq_sentence])))
                        print("机翻译文: {}".format("".join([zh_vocab.lookup_token(x) for x in translate_sentence[:index]])))
                        print("参考译文: {}".format(
                            "".join([zh_vocab.lookup_token(x) for x in reference_sentence[:reference_sentence_len]])))
                    #
                epoch_bleu = np.sum(blue_socres) / len(blue_socres)
                epoch_bleu *= 100
                wandb.log({"BLEU": epoch_bleu})

    wandb.finish()


