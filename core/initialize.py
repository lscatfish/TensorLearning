# -*- encoding:utf-8 -*-
"""初始化模型参数"""

from typing import Tuple
from xml.sax.xmlreader import InputSource
import numpy as np
from core.constant import EPSILON
from core.base import runtime


@runtime.init_func('normal')
def normal_init(shape: Tuple, mean = 1, std = 0.01) -> np.ndarray:
    return np.random.normal(mean, std, shape)


@runtime.init_func('randn')
def randn_init(shape: Tuple) -> np.ndarray:
    return np.random.randn(shape)


@runtime.init_func('uniform')
def uniform_init(shape: Tuple[int, int, ...], input_size = None, output_size = None) -> np.ndarray:
    if input_size is None:
        input_size = shape[0]
    if output_size is None:
        output_size = shape[1]
    return np.random.uniform(
        low = -1 / (np.sqrt(input_size) + EPSILON),
        high = -1 / (np.sqrt(output_size) + EPSILON),
        size = shape)


@runtime.init_func('x_normal')
def xavier_normal_init(shape: Tuple[int, int, ...], input_size = None, output_size = None) -> np.ndarray:
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
    if input_size is None:
        input_size = shape[0]
    return np.random.normal(
        0,
        np.sqrt(2 / input_size + EPSILON),
        shape)


@runtime.init_func('he_uniform')
def he_uniform_init(shape: Tuple[int, int, ...], input_size = None) -> np.ndarray:
    if input_size is None:
        input_size = shape[0]
    return np.random.uniform(
        -np.sqrt(6 / input_size + EPSILON),
        -np.sqrt(6 / input_size + EPSILON),
        shape)


if __name__ == "__main__":
    init = normal_init((10, 10))
    print(init)
