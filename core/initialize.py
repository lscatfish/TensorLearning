# -*- encoding:utf-8 -*-
"""
深度学习模型参数初始化模块
核心功能：实现深度学习中常用的权重/偏置初始化方法
包含：正态分布、均匀分布、Xavier、He等经典初始化策略
通过装饰器将初始化函数注册到运行时，供框架全局调用 """

from typing import Tuple
import numpy as np
from core.constant import EPSILON
from core.base import runtime


@runtime.init_func('normal')
def normal_init(shape: Tuple, mean = 1, std = 0.01) -> np.ndarray:
    """
    正态分布初始化（高斯分布）
    通用型参数初始化，默认均值1、小标准差
    :param shape: 初始化参数的形状，(输入维度, 输出维度)
    :param mean: 正态分布均值，默认1
    :param std: 正态分布标准差，默认0.01
    :return: 符合正态分布的numpy数组 """
    return np.random.normal(mean, std, shape)


@runtime.init_func('randn')
def randn_init(shape: Tuple) -> np.ndarray:
    """
    标准正态分布初始化
    特点：均值=0，标准差=1的标准正态分布
    :param shape: 初始化参数的形状
    :return: 标准正态分布numpy数组 """
    return np.random.randn(*shape)


@runtime.init_func('uniform')
def uniform_init(shape: Tuple[int, int, ...], input_size = None, output_size = None) -> np.ndarray:
    """
    均匀分布初始化
    基于输入/输出维度动态生成均匀分布范围，防止梯度消失/爆炸
    :param shape: 参数形状
    :param input_size: 输入层维度，默认取shape[0]
    :param output_size: 输出层维度，默认取shape[1]
    :return: 均匀分布numpy数组 """

    if input_size is None:
        input_size = shape[0]
    if output_size is None:
        output_size = shape[1]
    # 均匀分布范围：±1/√(维度)，加EPSILON防止分母为0
    return np.random.uniform(
        low = -1 / (np.sqrt(input_size) + EPSILON),
        high = -1 / (np.sqrt(output_size) + EPSILON),
        size = shape)


@runtime.init_func('x_normal')
def xavier_normal_init(shape: Tuple[int, int, ...], input_size = None, output_size = None) -> np.ndarray:
    """
    Xavier正态初始化（Glorot初始化）
    适用 Sigmoid / Tanh 激活函数
    方差 = 2/(输入维度+输出维度)，保证数据在前向传播中方差稳定
    :param shape: 参数形状
    :param input_size: 输入维度，默认shape[0]
    :param output_size: 输出维度，默认shape[1]
    :return: Xavier正态分布参数 """
    if input_size is None:
        input_size = shape[0]
    if output_size is None:
        output_size = shape[1]
    return np.random.normal(
        0,
        np.sqrt(2 / (input_size + output_size) + EPSILON),
        shape)


@runtime.init_func('he_normal')
def he_normal_init(shape: Tuple[int, int, ...], input_size = None) -> np.ndarray:
    """
    He正态初始化（Kaiming初始化）
    ReLU / LeakyReLU 激活函数
    方差 = 2/输入维度，解决ReLU激活函数导致的梯度消失问题
    :param shape: 参数形状
    :param input_size: 输入维度，默认shape[0]
    :return: He正态分布参数 """
    if input_size is None:
        input_size = shape[0]
    return np.random.normal(
        0,
        np.sqrt(2 / input_size + EPSILON),
        shape)


@runtime.init_func('he_uniform')
def he_uniform_init(shape: Tuple[int, int, ...], input_size = None) -> np.ndarray:
    """
    He均匀分布初始化，ReLU / LeakyReLU 激活函数
    分布范围 ±√(6/输入维度)，均匀分布版本的He初始化
    :param shape: 参数形状
    :param input_size: 输入维度，默认shape[0]
    :return: He均匀分布参数 """
    if input_size is None:
        input_size = shape[0]
    return np.random.uniform(
        -np.sqrt(6 / input_size + EPSILON),
        -np.sqrt(6 / input_size + EPSILON),
        shape)


if __name__ == "__main__":
    init = normal_init((10, 10))
    print(init)
