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


def __get_grad_by_shape(target_data, grad):
    """
    通用、兼容所有操作的梯度形状适配
    解决：广播操作反向的size不匹配问题
    适用：add/matmul/softmax/mean/mul等所有调用该函数的操作
    """
    target_shape = target_data.shape  # 目标节点的形状（需要输出的梯度形状）
    current_grad = grad.copy()

    # 核心逻辑：自动匹配广播维度，先求和，再调整形状，永不报错
    # 步骤1：将梯度的维度 对齐 目标形状的维度（补齐前面的维度）
    while len(current_grad.shape) < len(target_shape):
        current_grad = np.expand_dims(current_grad, axis = 0)

    # 步骤2：遍历所有维度，将长度不匹配的维度求和（广播反向核心！）
    for axis in range(len(target_shape)):
        target_dim = target_shape[axis]
        grad_dim = current_grad.shape[axis]
        # 如果目标维度=1，梯度维度>1：沿着该维度求和（解决偏置b的广播问题）
        if target_dim == 1 and grad_dim > 1:
            current_grad = current_grad.sum(axis = axis, keepdims = True)

    # 步骤3：最终reshape（此时元素数量完全匹配，100%不报错）
    return current_grad.reshape(target_shape)

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
def __softmax_gradient(op_node: Union[Operation, Any], grad: np.ndarray):
    """
    修复前：梯度算成(2,) → 报错
    修复后：梯度保持(17,2) → 匹配前向形状，梯度流正常
    """
    # 前向输出的softmax值 (17,2)
    sm = op_node.data
    # 逐样本计算softmax梯度（保留样本维度17，不压扁！）
    grad_out = np.empty_like(sm)
    for i in range(sm.shape[0]):  # 遍历每个样本（关键修复：保留样本维度）
        s = sm[i].reshape(-1, 1)
        jacobian = np.diagflat(s) - np.dot(s, s.T)
        grad_out[i] = np.dot(jacobian, grad[i])
    return grad_out,

