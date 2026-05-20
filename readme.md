# 此项目用于完成机器学习作业，顺便学习 torch 的构架

## 项目结构

```
TensorLearning/
├── main1.py                  # 西瓜二分类 demo（mt 框架入门示例）
├── iris/                     # 鸢尾花分类数据集
│   ├── iris.csv              # 带表头的 CSV（150 条 / 4 特征 / 3 类）
│   └── iris.data             # UCI 原始格式
├── mt/                       # 自研 NumPy 深度学习框架
│   └── core/
│       ├── base.py           # 计算图核心：Node、Session、数学运算
│       ├── constant.py       # 全局运行时 & 注册表
│       ├── initialize.py     # 权重初始化（Xavier、He、正态、均匀）
│       ├── optm.py           # 反向传播（BFS） + SGD / Momentum / Adam
│       ├── util.py           # Register 类、one-hot、彩色打印
│       ├── function/
│       │   ├── activate.py   # 激活函数：sigmoid、tanh、relu、softmax 等
│       │   ├── gradiend.py   # 所有操作的梯度函数（自动微分核心）
│       │   └── measure.py    # 损失函数：CrossEntropy、MSE
│       └── net/
│           └── linear.py     # Linear（全连接）层
└── minist/                   # MNIST 手写数字识别（PyTorch）
    ├── download.py           # 下载 MNIST 并保存为 JPG
    ├── model.py              # 三种模型 + 训练循环
    ├── analysis.py           # 模型评估与可视化
    ├── plt_train_data.py     # 绘制训练曲线
    ├── 训练结果.md           # 训练日志
    ├── mnist_jpg/            # 下载的 MNIST 图片（~70K 张）
    └── md/                   # 训练好的模型权重（.pth）
```

## mt — 自研 NumPy 自动微分框架

基于纯 Python + NumPy 实现的静态计算图与自动梯度反向传播库，类似 TensorFlow 1.x / PyTorch 底层原理。

**核心特性：**
- **计算图**：Node → Operation → Session 前向执行
- **自动微分**：BFS 反向传播，链式法则梯度累加
- **优化器**：SGD、Momentum、Adam
- **激活函数**：sigmoid、tanh、relu、leaky_relu、elu、softmax
- **损失函数**：CrossEntropy、MSE
- **权重初始化**：正态、Xavier、He
- **全连接层**：Linear（支持 bias 和激活）

## minist — MNIST 手写数字识别（PyTorch）

基于 PyTorch 实现三种模型架构，完成 MNIST 10 分类任务。

**模型：**
| 模型 | 架构 | 测试准确率 |
|------|------|-----------|
| ConvNet | 7 层卷积 + 2 层全连接 | 98.90% |
| PatchNet | Patch 分块 + 1D 卷积 | 98.82% |
| ResNet | 残差连接卷积网络 | 98.92% |

## Iris — 鸢尾花分类（待完成）

使用经典 Iris 数据集（150 条样本，3 类，4 个特征），基于 `mt` 框架实现分类任务。

## main1.py — 西瓜分类

西瓜书（周志华《机器学习》）中的二分类示例，使用 `mt` 框架构建 2 层神经网络，对 17 条西瓜数据进行好瓜/坏瓜分类。
