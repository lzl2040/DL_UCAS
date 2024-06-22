# -*- encoding: utf-8 -*-
"""
File train.py
Created on 2024/5/6 16:07
Copyright (c) 2024/5/6
@author: 
"""
import os
import sys

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from timm.utils import AverageMeter
from torch.utils.data import DataLoader
from loguru import logger
import argparse
import pytorch_warmup as warmup
# https://libraries.io/pypi/warmup-scheduler
import warmup_scheduler
from torchvision.transforms import transforms

# 解决ModuleNotFoundError: No module named 'Exp2
# sys.path.append("/home/gaoxingyu/Project/Deep_Learning/")
from module.ViT import ViT, vit_base_patch16_224

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 保证结果可重复
torch.manual_seed(123243)
# 检查CUDA是否可用
if torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info("CUDA is available! Training on GPU.")
else:
    device = torch.device("cpu")
    logger.info("CUDA is not available. Training on CPU.")

def get_args_parser():
    parser = argparse.ArgumentParser(description="defination of some parameters")
    parser.add_argument("--batch-size", type=int, default=128, help="batch size for training")
    parser.add_argument("--test-bs", type=int, default=256, help="batch size for testing")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate 1e-3")
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--img-size", type=int, default=224, help="crop image size")
    parser.add_argument("--size", type=int, default=256, help="image size in test")
    parser.add_argument("--epoches", type=int, default=150, help="epoches for training")
    parser.add_argument("--max_steps", type=int, default = 200 * 391)
    parser.add_argument("--warmup_step", type=int, default=5 * 391)
    parser.add_argument("--log-interval", type=int, default=5, help="the interval(iter) of logging")
    parser.add_argument("--save-interval", type=int, default=5, help="the interval(epoch) of saving the model")
    parser.add_argument("--save-path", type=str, default="/data2/gaoxingyu/Deep_Learning/saves", help="the path to save the weights")
    parser.add_argument("--test_interval", type=int, default=2, help="the interval(epoch) of testing the model")
    # 优化器
    parser.add_argument("--opt-type", type=str, default="adamw", help="the type of the optimizer")
    parser.add_argument("--steps", type=list, default=[24000], help="the steps when learning rate decay")
    parser.add_argument("--gamma", type=float, default=0.1, help="LR * gamma at every decay step")
    # 模型的参数
    parser.add_argument("--classes", type=int, default=10, help="classes num in the dataset")
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--img_channels", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dim_head", type=int, default=32)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--pool", type=str, default="cls")

    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--ckt_path", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default="vit_b_img_224_patch_16")

    args = parser.parse_args()
    return args

def construct_loader(args):
    # 数据增强
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.458, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(args.size),
        transforms.RandomCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.458, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载数据集
    cifar10_train = torchvision.datasets.CIFAR10(
        root='/data2/gaoxingyu/dataset',
        train=True,
        download=False,
        transform=train_transforms
    )
    cifar10_test = torchvision.datasets.CIFAR10(
        root='/data2/gaoxingyu/dataset',
        train=False,
        download=False,
        transform=test_transforms
    )
    # 数据集加载
    train_loader = DataLoader(cifar10_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(cifar10_test, batch_size=args.test_bs, shuffle=False, num_workers=2)
    return train_loader, test_loader

def train(args, train_loader, test_loader, model):
    # 从中断处加载
    start_epoch = 1
    total_iter = 0
    best_acc = 0
    if args.resume:
        state_dict = torch.load(args.resume, map_location="cuda:0")
        vit_model.load_state_dict(state_dict)
        logger.info(f"resume from {args.resume}")
        start_epoch = 81
        best_acc = 84.651
    logger.info("Start training...")
    # 损失函数：交叉熵损失
    criterion = nn.CrossEntropyLoss()
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=5e-5)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.steps, args.gamma)
    base_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps, eta_min=args.min_lr)
    scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1., total_epoch=args.warmup_step, after_scheduler=base_scheduler)
    for epoch in range(start_epoch, args.epoches + 1):
        logger.info(f"Epoch:{epoch}")
        # 记录损失
        loss_avg = AverageMeter()
        acc_avg = AverageMeter()
        total_num = 0
        correct = 0
        iter = 0

        for image, label in train_loader:
            image, label = image.to(device), label.to(device)
            out = model(image)
            optimizer.zero_grad()
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            scheduler.step()
            # 更新损失
            loss_avg.update(loss.item(), args.batch_size)
            # 计算准确率
            _, pred = out.max(-1)
            total_num += image.shape[0]
            correct += pred.eq(label).sum().item()
            acc_rate = (correct / total_num) * 100
            acc_avg.update(acc_rate, 1)
            iter += 1
            total_iter += 1
            if iter % args.log_interval == 0:
                logger.info(f"Epoch:{epoch}({args.epoches}) Iter:{iter}({len(train_loader)}) "
                            f"Lr:{optimizer.param_groups[0]['lr']} Loss:{loss_avg.val:.6f}({loss_avg.avg:.6f}) "
                            f"Acc:{acc_avg.val:.2f}({acc_avg.avg:.2f})")
        if epoch % args.test_interval == 0:
            test_acc = test(args, test_loader, model)
            logger.info("####" * 6)
            logger.info(f"Epoch:{epoch} Test acc:{test_acc:.3f}")
            logger.info("####" * 6)
            if best_acc < test_acc:
                best_acc = test_acc
                save_path = os.path.join(args.save_path, args.exp_name, f"{args.opt_type}_epoch_{epoch}_acc_{best_acc:.3f}.pth")
                # 保存ckt
                state_dict = {
                    "epoch" : epoch,
                    "optimizer" : optimizer.state_dict(),
                    "scheduler" : scheduler.state_dict(),
                    "weight" : model.state_dict(),
                    "best_acc" : best_acc,
                    "iter" : total_iter
                }
                torch.save(state_dict, save_path)
                logger.info(f"Save model to {save_path}")
        model.train()

def test(args, test_loader, model):
    if args.ckt_path != None:
        # 加载权重
        state_dict = torch.load(args.ckt_path, map_location="cuda:0")
        model.load_state_dict(state_dict)
        logger.info(f"load weights from {args.ckt_path}")
    logger.info("Start testing")
    model.eval()
    # 损失函数：交叉熵损失
    criterion = nn.CrossEntropyLoss()
    # 记录损失
    loss_avg = AverageMeter()
    acc_avg = AverageMeter()
    total_num = 0
    correct = 0
    iter = 0
    with torch.no_grad():
        for image, label in test_loader:
            image, label = image.to(device), label.to(device)
            out = model(image)
            loss = criterion(out, label)
            loss_avg.update(loss.item(), args.test_bs)
            _, pred = out.max(-1)
            total_num += image.shape[0]
            correct += pred.eq(label).sum().item()
            acc_rate = (correct / total_num) * 100
            acc_avg.update(acc_rate, 1)
            iter += 1
            logger.info(f"Iter: {iter}/{len(test_loader)} Loss:{loss_avg.val:.6f}({loss_avg.avg:.6f}) "
                        f"Acc:{acc_avg.val:.2f}({acc_avg.avg:.2f})")
    return acc_avg.avg

if __name__ == '__main__':
    # 相关参数的定义
    args = get_args_parser()
    save_path = os.path.join(args.save_path, args.exp_name)
    os.makedirs(args.save_path, exist_ok=True)
    train_loader, test_loader = construct_loader(args)
    vit_model = vit_base_patch16_224(num_classes=args.classes).to(device)
    if args.mode == "train":
        train(args, train_loader, test_loader, vit_model)
    else:
        test(args, test_loader, vit_model)