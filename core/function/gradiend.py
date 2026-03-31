# -*- encoding:utf-8 -*-
"""
梯度计算核心模块
核心功能：为所有基础运算/激活函数注册 **反向传播梯度计算函数**
通过装饰器注册到runtime，实现框架自动求导
所有函数均为反向传播的回调函数，输入上游梯度，输出当前操作输入节点的梯度
"""
import numpy as np
from typing import Any, Union
from core.constant import runtime, Clip
from core.base import Node, Operation


def __get_grad_by_shape(node: Node, grad: np.ndarray):
    """
    核心工具：自动适配梯度形状（解决前向传播广播操作的反向梯度维度不匹配问题）
    场景：加法/乘法等操作前向时发生广播，反向梯度需要还原为输入节点原始形状
    :param node: 输入节点（需要匹配形状的目标）
    :param grad: 上游回传的梯度
    :return: 适配形状后的梯度
    """
    # 获取节点原始形状和梯度形状
    node_shape, grad_shape = node.shape, grad.shape
    # 形状一致，直接返回梯度
    if node_shape == grad_shape:
        return grad
    # 计算需要压缩的维度（广播维度）
    squeeze_axes = []
    # 遍历梯度的所有维度，找出和目标形状不匹配的轴
    for g_axis, n_axis in zip(range(len(grad_shape)), range(len(node_shape))):
        if grad_shape[g_axis] != node_shape[n_axis]:
            squeeze_axes.append(g_axis)
    # 处理梯度维度比目标多的情况
    if len(grad_shape) > len(node_shape):
        squeeze_axes.extend(range(len(node_shape), len(grad_shape)))
    # 对广播维度求和（保持梯度数值正确，不能用mean！）
    grad_squeezed = np.sum(grad, axis = tuple(squeeze_axes))
    # 重塑为目标形状
    return grad_squeezed.reshape(node_shape)


# ====================== 基础数学运算 梯度注册 ======================
@runtime.gradient_func("add")
def __add_gradient(op_node: Operation, grad: np.ndarray):
    """
    加法操作 反向梯度
    dz/dx = 1, dz/dy = 1
    :param op_node: 加法操作节点
    :param grad: 上游回传的梯度 d(out)/dz
    :return: [x的梯度, y的梯度]
    """
    return [
        1. * __get_grad_by_shape(op_node.input_nodes[0].data, grad),
        1. * __get_grad_by_shape(op_node.input_nodes[1].data, grad)
    ]


@runtime.gradient_func("minus")
def __minus_gradient(op_node: Operation, grad: np.ndarray):
    """
    减法操作 反向梯度
    dz/dx = 1, dz/dy = -1
    :return: [x的梯度, y的梯度]
    """
    return [
        1. * __get_grad_by_shape(op_node.input_nodes[0].data, grad),
        -1. * __get_grad_by_shape(op_node.input_nodes[1].data, grad)
    ]


@runtime.gradient_func("negative")
def __negative_gradient(op_node: Operation, grad: np.ndarray):
    """
    负号操作 反向梯度
    d(-x)/dx = -1
    :return: [x的梯度]
    """
    return [-1. * grad]


@runtime.gradient_func("elementwise_pow")
def __elementwise_pow_gradient(op_node: Operation, grad: np.ndarray):
    """
    逐元素幂运算 反向梯度（代码原名拼写错误，已注释修正）
    d(x^y)/dx = y * x^(y-1)
    :param op_node: 幂操作节点（self.y为幂次）
    :return: [x的梯度]
    """
    x = op_node.input_nodes[0].data  # 底数
    y = op_node.y  # 幂次
    return [y * (x ** (y - 1)) * grad]


@runtime.gradient_func("matmul")
def __matmul_gradient(op_node: Operation, grad: np.ndarray):
    """
    矩阵乘法 反向梯度
    d(X@Y)/dX = grad @ Y.T, d(X@Y)/dY = X.T @ grad
    :return: [X的梯度, Y的梯度]
    """
    x = op_node.input_nodes[0].data
    y = op_node.input_nodes[1].data
    return [grad @ y.T, x.T @ grad]


@runtime.gradient_func("multiply")
def __multiply_gradient(op_node: Operation, grad: np.ndarray):
    """
    逐元素乘法 反向梯度
    d(x*y)/dx = y, d(x*y)/dy = x
    :return: [x的梯度, y的梯度]
    """
    x = op_node.input_nodes[0].data
    y = op_node.input_nodes[1].data
    return [y * grad, x * grad]


@runtime.gradient_func("reduce_sum")
def __reduce_sum_gradient(op_node: Operation, grad: np.ndarray):
    """
    求和操作 反向梯度
    原理：求和的梯度为全1矩阵（每个元素对结果贡献相同）
    :return: [输入的梯度]
    """
    grad_shape = op_node.input_nodes[0].shape
    return [1. * np.ones(grad_shape) * grad]


@runtime.gradient_func("reduce_mean")
def __reduce_mean_gradient(op_node: Operation, grad: np.ndarray):
    """
    均值操作 反向梯度
    原理：均值梯度为 1/元素总数（代码中multiplier默认1，可扩展）
    :return: [输入的梯度]
    """
    multiplier = 1
    grad_shape = op_node.input_nodes[0].shape
    return [1. * np.ones(grad_shape) / multiplier * grad]


@runtime.gradient_func("log")
def __log_gradient(op_node: Operation, grad: np.ndarray):
    """
    对数操作 反向梯度
    d(log(x))/dx = 1/x
    :return: [x的梯度]
    """
    x = op_node.input_nodes[0].data
    return [1. / x * grad]


# ====================== 激活函数 梯度注册 ======================
@runtime.gradient_func("sigmoid")
def __sigmoid_gradient(op_node: Operation, grad: np.ndarray):
    """
    Sigmoid激活函数 反向梯度
    sigmoid(x)' = sigmoid(x) * (1-sigmoid(x)) = ex/(1+ex)²
    :return: [输入梯度]
    """
    x = op_node.input_nodes[0].data
    ex = Clip.EXP(x)  # 裁剪指数，防止数值溢出
    return [ex / ((1 + ex) ** 2) * grad]


@runtime.gradient_func("tanh")
def __tanh_gradient(op_node: Operation, grad: np.ndarray):
    """
    Tanh激活函数 反向梯度
    tanh(x)' = 1 - tanh(x)² = 4e²x/(1+e²x)²
    :return: [输入梯度]
    """
    x = op_node.input_nodes[0].data
    e2x = Clip.EXP(2. * x)
    return [4 * e2x / ((1 + e2x) ** 2) * grad]


@runtime.gradient_func("relu")
def __relu_gradient(op_node: Operation, grad: np.ndarray):
    """
    ReLU激活函数 反向梯度
    relu(x)' = 1 (x>0), 0 (x≤0)
    简化实现：用输出/输入直接求导
    :return: [输入梯度]
    """
    x = op_node.input_nodes[0].data
    y = op_node.data
    k = y / x  # 梯度系数：x>0=1，x≤0=0
    return [k * grad]


@runtime.gradient_func("leaky_relu")
def __leaky_relu(op_node: Operation, grad: np.ndarray):
    """
    Leaky ReLU激活函数 反向梯度
    leaky_relu(x)' = 1 (x>0), alpha (x≤0)
    简化实现：用输出/输入直接求导
    :return: [输入梯度]
    """
    x = op_node.input_nodes[0].data
    y = op_node.data
    k = y / x
    return [k * grad]


@runtime.gradient_func("elu")
def __elu_gradient(op_node: Union[Operation, Any], grad: np.ndarray):
    """
    ELU激活函数 反向梯度
    elu(x)' = 1 (x≥0), alpha*e^x (x<0)
    :return: [输入梯度]
    """
    x = op_node.input_nodes[0].data
    k = np.array(x)
    k[k >= 0] = 1.  # 正区间梯度=1
    k[k < 0] = op_node.alpha * np.exp(k[k < 0])  # 负区间梯度=alpha*e^x
    return [k * grad]


@runtime.gradient_func("softmax")
def __softmax_gradient(op_node: Operation, grad: np.ndarray):
    """
    Softmax激活函数 反向梯度（简化版，配合交叉熵使用）
    softmax(x)' = output * (1 - output)
    完整Softmax梯度为矩阵，分类任务中与交叉熵结合后简化为此形式
    :return: 简化后的梯度
    """
    s = op_node.data
    return s * grad - np.sum(s * grad, axis = 1, keepdims = True) * s
