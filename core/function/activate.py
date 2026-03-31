# -*- encoding:utf-8 -*-
# 注册激活函数计算

from core.constant import runtime, Clip
from core.base import Operation, Node
import numpy as np


@runtime.activate_func("sigmoid")
class sigmoid(Operation):
    def __init__(self, x: Node, node_name: str = ""):
        super().__init__(input_nodes = [x], node_name = node_name)

    def compute(self, x_v: np.ndarray):
        return 1 / (1 + Clip.EXP(-1. * x_v))  # 防止计算溢出


@runtime.activate_func("tanh")
class tanh(Operation):
    def __init__(self, x: Node, node_name: str = ""):
        super().__init__(input_nodes = [x], node_name = node_name)

    def compute(self, x_v: np.ndarray):
        ex = Clip.EXP(x_v)
        dex = 1. / ex
        return (ex - dex) / (ex + dex)


@runtime.activate_func("relu")
class relu(Operation):
    def __init__(self, x: Node, node_name: str = ""):
        super().__init__(input_nodes = [x], node_name = node_name)

    def compute(self, x_v: np.ndarray):
        y = np.array(x_v)
        y[y < 0] = 0.
        return y


@runtime.activate_func("leaky_relu")
class leaky_relu(Operation):
    def __init__(self, x: Node, alpha: float = 1e-2, node_name: str = ""):
        super().__init__(input_nodes = [x], node_name = node_name)
        self.alpha = alpha

    def compute(self, x_v: np.ndarray):
        y = np.array(x_v)
        y[y < 0] *= self.alpha
        return y


@runtime.activate_func("elu")
class elu(Operation):
    def __init__(self, x: Node, alpha: float = 1e-2, node_name: str = ""):
        super().__init__(input_nodes = [x], node_name = node_name)
        self.alpha = alpha

    def compute(self, x_v: np.ndarray):
        y = np.array(x_v)
        y[y < 0] = self.alpha * (np.exp(y[y < 0]) - 1)
        return y


@runtime.activate_func("softmax")
class softmax(Operation):
    def __init__(self, x: Node, axis: int = None, node_name: str = ""):
        super().__init__(input_nodes = [x], node_name = node_name)
        self.axis = axis

    def compute(self, x_v: np.ndarray):
        ex = np.exp(x_v)
        reduce_shape = list(x_v.shape)
        reduce_shape[self.axis] = 1
        return ex / np.sum(ex, axis = self.axis).reshape(reduce_shape)
