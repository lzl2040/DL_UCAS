# 基于ViT的cifar-10图像分类
## 运行
训练使用如下命令：
```
python train.py
```

使用了warm up的策略调整学习率，即最开始学习率从一个最小值上升到一个最大值， 然后再随着迭代次数逐步下降。调整的参数为：
- lr：最大的学习率
- warmup_step：经过多少步达到最大学习率

测试的时候使用命令：
```js
python train.py --mode test --ckt_path 权重地址
```