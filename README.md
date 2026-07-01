# deep-learning-network-lab

一个用来记录深度学习入门过程的实验仓库，重点是从零理解神经网络 demo 的搭建方式：数据加载、模型定义、训练循环、验证指标、模型保存和测试评估。

当前 demo 使用 PyTorch 实现一个 LeNet 风格的卷积神经网络，并在 FashionMNIST 数据集上训练和测试。

## 当前内容

| 文件 | 作用 |
| --- | --- |
| `model.py` | 定义 LeNet 模型结构和设备选择逻辑 |
| `model_train.py` | 下载 FashionMNIST，划分训练/验证集，训练模型并保存权重 |
| `model_test.py` | 加载已训练权重，在测试集上计算准确率 |
| `plot.py` | 可视化 FashionMNIST 的一批训练样本 |
| `main.py` | 检查当前环境是否支持 Apple MPS |
| `requirements.txt` | 项目依赖 |

## 环境准备

建议使用虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 运行方式

查看模型结构：

```bash
python model.py
```

可视化训练样本：

```bash
python plot.py
```

训练模型：

```bash
python lenet_train.py
```

训练完成后会生成：

```text
model.pth
```

测试模型：

```bash
python model_test.py
```

## 学习重点

这个仓库目前主要用于理解这些概念：

- `Dataset` / `DataLoader` 的基本用法
- 训练集、验证集、测试集的职责区别
- `model.train()` 和 `model.eval()` 的区别
- `torch.no_grad()` 在验证/测试阶段的作用
- `CrossEntropyLoss` 和分类任务输出维度的关系
- batch 级 loss / accuracy 如何累计成 epoch 指标
- `state_dict()` 保存和加载模型参数的方式

## 当前模型

当前模型是一个简化版 LeNet：

```text
Conv2d -> Sigmoid -> AvgPool2d
Conv2d -> Sigmoid -> AvgPool2d
Flatten
Linear -> Linear -> Linear
```

输入为 FashionMNIST 的单通道 `28x28` 图片，输出为 10 个类别的 logits。

## 后续计划

后面可以继续补这些实验：

- 使用 ReLU 替换 Sigmoid，对比训练效果
- 加入 BatchNorm / Dropout，观察 train/eval 模式差异
- 抽象 `Trainer`，拆分 `train_one_epoch()` 和 `evaluate()`
- 增加固定随机种子，让实验更容易复现
- 对比 SGD、Adam 和学习率调度器
- 尝试 CIFAR-10、MNIST、自己的小图片数据集

## 代码风格

本仓库倾向于给 Python 函数补全类型注解，让 demo 代码尽量保持清楚、可读、方便 IDE 检查。
