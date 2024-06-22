# 机器翻译
参考代码：[MT](https://github.com/sunhanwu/MT/)
## 数据预处理
预处理的代码在[prepare.py](datasets/prepare.py)中
### 构建词典
在Bert的tokenizer的基础上使用```build_vocab_from_iterator```构建词典，去除频率少的词，最终得到```vocab_zh.pt```和```vocab_en.pt```

### 训练集
使用上面生成的词典将数据以token id的形式存储，分别存储在```tokens_list.en.pt```和```tokens_list.zh.pt```中。

### 验证集
使用```prepare_val_data```函数，即可将文本转换为一系列的token id，并存储在文件中，方便下次使用。

## 训练
注意，你需要修改数据集的路径和保存文件的路径。

后台训练：
```js
nohup python -u  train.py  > nohup.txt 2>&1 &
```
前台运行：
```js
python train.py
```
## 测试
测试也在train.py文件中，在训练过程中会每隔一定的epoch对模型进行验证。

