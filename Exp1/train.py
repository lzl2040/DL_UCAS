# -*- encoding: utf-8 -*-
"""
File train.py.py
Created on 2024/4/5 23:16
Copyright (c) 2024/4/5
@author: 
"""
import os.path

from timm.utils import AverageMeter
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import argparse
import torch
import torch.optim as optim
from loguru import logger

from Exp1.cnn import CNN

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
    parser.add_argument("--batch-size", type=int, default=4, help="batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--epoch", type=int, default=40, help="epoches for training")
    parser.add_argument("--log-interval", type=int, default=50, help="the interval(iter) of logging")
    parser.add_argument("--save-interval", type=int, default=2, help="the interval(epoch) of saving the model")
    parser.add_argument("--save-path", type=str, default="saves", help="the path to save the weights")
    parser.add_argument("--opt-type", type=str, default="sgd", help="the type of the optimizer")
    # parser.add_argument("--test_interval", type=int, default=5, help="the interval(epoch) of testing the model")
    parser.add_argument("--classes", type=int, default=10, help="classes num in the dataset")
    args = parser.parse_args()
    return args

def construct_loader(args):
    # 训练数据的预处理，含数据增强
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # 加载训练数据集
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    # 加载测试数据集
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1)
    logger.info(f"Train dataset size:{len(train_loader)}")
    logger.info(f"Test dataset size:{len(test_loader)}")
    return train_loader, test_loader

def train_and_test(args, model, train_loader, test_loader):
    # 创建优化器
    # weight decay表示权重衰减
    if args.opt_type == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.01)
    elif args.opt_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.opt_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.01)
    # 损失函数：交叉熵损失
    criterion = nn.CrossEntropyLoss()
    iter = 0
    # 记录损失
    loss_avg = AverageMeter()
    # 最好的准确率
    best_acc = 0
    logger.info("Start training...")
    for epoch in range(1, args.epoch + 1):
        logger.info(f"Training epoch:{epoch}")
        for image, label in train_loader:
            # img: B C(1) H(28) W(28)
            # label: B
            # image = image.to(device)
            # label = label.to(device)
            # 梯度设为0
            optimizer.zero_grad()
            # 得到预测结果
            pred = model(image)
            loss = criterion(pred, label)
            # 计算梯度
            loss.backward()
            # 更新参数
            optimizer.step()
            loss_avg.update(loss.item(), args.batch_size)
            # 输出日志
            if iter % args.log_interval == 0:
                logger.info(f"Epoch:{epoch} Iter:{iter} Loss:{loss_avg.avg}")
            iter += 1
        # 保存模型
        if epoch % args.save_interval == 0:
            # 测试
            acc = test(model, test_loader)
            logger.info(f"Epoch:{epoch} Acc:{acc}%")
            if best_acc < acc:
                best_acc = acc
                # 选择精度最好的保存
                save_path = os.path.join(args.save_path, f"{args.opt_type}_epoch_{epoch}_acc_{best_acc:.2f}.pth")
                torch.save(model.state_dict(), save_path)
                logger.info(f"Save model to {save_path}")
    logger.info("Training over!")

def test(model, data_loader):
    logger.info("Start testing...")
    # 预测正确的数目
    correct_num = 0
    # 总的图片数目
    total_num = 0
    with torch.no_grad():
        for image, label in data_loader:
            out = model(image)
            pred = torch.argmax(out)
            correct_num += torch.sum(pred == label)
            total_num += image.shape[0]
    logger.info("Test over!")
    return (correct_num / total_num) * 100

if __name__ == '__main__':
    # 相关参数的定义
    args = get_args_parser()
    os.makedirs(args.save_path, exist_ok=True)
    # 加载数据集
    train_loader, test_loader = construct_loader(args)
    # 创建网络
    cnn_net = CNN(class_num=args.classes)
    # 训练
    train_and_test(args, cnn_net, train_loader, test_loader)


