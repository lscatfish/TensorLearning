#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
"""
MNIST 下载与图片导出工具

功能：
    1. 通过 torchvision.datasets.MNIST 下载 / 加载 MNIST 数据集（训练集 60K、测试集 10K）；
    2. 将数据集中的灰度图片以 JPEG 格式批量导出到 mnist_jpg/ 目录；
    3. 文件名格式：{training|test}_{序号}_{标签}.jpg，供 model.py 按文件名解析标签。

依赖：torchvision、Pillow、tqdm
"""

import os

from PIL import Image
from torchvision.datasets import MNIST
from tqdm import tqdm


def _save_split(dataset, save_dir: str, prefix: str) -> None:
    """把某个划分（训练/测试）的全部图片保存为 JPG。

    :param dataset: torchvision 数据集对象
    :param save_dir: 图片输出目录
    :param prefix: 文件名前缀，training / test
    """
    with tqdm(total=len(dataset), ncols=150, desc=f"saving {prefix}") as bar:
        for idx, (img, label) in enumerate(dataset):
            # PIL Image 对象直接以 JPG 格式落盘
            filepath = os.path.join(save_dir, f"{prefix}_{idx}_{label}.jpg")
            img.save(filepath)
            bar.update(1)


if __name__ == "__main__":
    # 图片保存路径（相对当前工作目录）
    save_dir = "mnist_jpg"
    os.makedirs(save_dir, exist_ok=True)

    # 下载 / 加载 MNIST 数据（原始数据缓存到 mnist/ 目录）
    train_dataset = MNIST(root="mnist", train=True, download=True)
    test_dataset = MNIST(root="mnist", train=False, download=True)

    # 导出训练集与测试集图片
    _save_split(train_dataset, save_dir, prefix="training")
    _save_split(test_dataset, save_dir, prefix="test")
