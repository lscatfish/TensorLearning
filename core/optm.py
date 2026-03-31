# -*- encoding:utf-8 -*-
"""
优化器与反向传播模块

 backwards函数：通过BFS遍历计算图，实现 **自动反向传播求梯度**
 Optimizer基类：定义优化器通用接口
 优化器：实现梯度下降，更新可训练参数(Variable)
 优化器注册字典：统一管理所有优化器
"""
from typing import Dict

import numpy as np
import abc
from collections import deque

from core.base import Data, Operation, Placeholder, runtime, Variable, Node
from core.constant import EPSILON


def _backwards_(op_node: Operation) -> Dict[Operation | Node | Variable, float]:
    """
    反向传播函数：从损失节点开始，遍历整个计算图，计算所有节点的梯度
    BFS 反向遍历计算图
    :param op_node: 根节点（通常是损失函数节点）
    :return: grad_table 梯度表：{节点: 该节点的梯度值}
    """
    # 梯度表：存储每个计算节点对应的梯度值，初始损失节点梯度为1（d(loss)/d(loss)=1）
    grad_table = {op_node: 1.}
    # 已访问节点集合：防止重复计算，避免循环依赖
    visit_nodes = set()
    # BFS队列：按层级遍历计算图节点
    queue = deque()

    # 初始化：将损失节点加入已访问和队列
    visit_nodes.add(op_node)
    queue.append(op_node)

    # BFS循环遍历所有节点
    while len(queue) > 0:
        # 取出队列头部节点（先进先出）
        cur_node = queue.popleft()

        # 排除、根节点、占位符(Placeholder)、常量(Data)，这些节点不需要计算梯度
        if cur_node != op_node and not (isinstance(cur_node, Placeholder) or isinstance(cur_node, Data)):
            # 初始化当前节点的梯度为0
            grad_table[cur_node] = 0
            # 遍历当前节点的后继节点（依赖当前节点的操作）
            for next_node in cur_node.next_nodes:
                # 获取：后继节点的梯度（上游回传的梯度）
                grad_loss_wrt_next_node: np.ndarray = grad_table[next_node]
                # 获取：后继节点的类名（用于匹配注册的梯度函数）
                next_node_op_name: str = next_node.__class__.__name__
                # 从运行时注册器中获取：该操作对应的梯度计算函数
                gradient_func = runtime.gradient_func[next_node_op_name]
                # 调用梯度函数，计算当前节点相对于损失的梯度
                grad_loss_wrt_cur_node = gradient_func(next_node, grad_loss_wrt_next_node)

                # 累加梯度（处理多输出节点的梯度合并）
                if len(next_node.input_nodes) == 1:
                    # 单输入操作：直接累加梯度
                    grad_table[cur_node] += grad_loss_wrt_cur_node[0]
                else:
                    # 多输入操作：找到当前节点在输入中的位置，累加对应梯度
                    cur_node_in_next_node_index = next_node.input_nodes.index(cur_node)
                    grad_table[cur_node] += grad_loss_wrt_cur_node[cur_node_in_next_node_index]
        # 如果当前节点是操作节点(Operation)，继续遍历它的输入节点（前向传播的上游）
        if isinstance(cur_node, Operation):
            for input_node in cur_node.input_nodes:
                # 未访问过的节点：加入集合和队列
                if input_node not in visit_nodes:
                    visit_nodes.add(input_node)
                    queue.append(input_node)
    # 返回所有节点的梯度表
    return grad_table


class Optimizer(abc.ABC):
    """优化器基类：定义所有优化器必须实现的接口"""

    def __init__(self, lr_rate: float = 1e-3):
        """学习率"""
        self.learning_rate = lr_rate

    @abc.abstractmethod
    def backward(self, loss_node: Operation | Node):
        """
        反向传播, 最小化损失函数，子类必须实现
        计算梯度 -> 更新可训练参数
        :param loss_node: 损失函数节点
        """

    def zero_grad(self):
        """清除梯度"""
        if hasattr(runtime, 'grad_table'):
            runtime.grad_table = {}


class SGD(Optimizer):
    """
    SGD（随机梯度下降）优化器
    参数 = 参数 - 学习率 * 梯度
    """

    def __init__(self, learning_rate: float = 1e-3):
        super().__init__(lr_rate = learning_rate)

    def backward(self, loss_node: Operation | Node):
        """
        实现SGD参数更新
        调用backwards()计算所有节点梯度
        仅更新Variable（可训练参数）
        保存梯度表到运行时，方便后续查看
        """
        lr = self.learning_rate
        # 反向传播：获取所有节点的梯度
        grad_table = _backwards_(op_node = loss_node)

        # 遍历梯度表，仅更新可训练变量(Variable)
        for node in grad_table:
            if isinstance(node, Variable):
                # 获取当前变量的梯度
                grad = grad_table[node]
                # SGD核心更新公式：w = w - lr * dw
                node.data -= lr * grad

        # 将梯度表存入全局运行时，供调试/可视化使用
        runtime.grad_table = grad_table
        # 返回梯度表
        return grad_table


class Momentum(Optimizer):
    def __init__(self, learning_rate: float = 0.001, gamma = 0.7):
        super().__init__(learning_rate)
        # save gradient of each node
        self.node2v = {}
        self.gamma = gamma

    def backward(self, loss_node: Operation):
        lr = self.learning_rate
        grad_table = _backwards_(op_node = loss_node)
        for node in grad_table:
            if isinstance(node, Variable):
                grad = grad_table[node]
                if node not in self.node2v:
                    self.node2v[node] = lr * grad
                else:
                    self.node2v[node] = self.gamma * self.node2v[node] + lr * grad
                node.data -= self.node2v[node]

        runtime.grad_table = grad_table
        return grad_table


class Adam(Optimizer):
    def __init__(self, learning_rate: float = 0.001, beta1 = 0.9, beta2 = 0.999):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2

        self.prod_beta1 = 1
        self.prod_beta2 = 1
        self.node2m = {}  # history value
        self.node2v = {}  # accumulate value

    def backward(self, loss_node: Operation | Node):
        lr = self.learning_rate
        grad_table = _backwards_(op_node = loss_node)
        for node in grad_table:
            if isinstance(node, Variable):
                grad = grad_table[node]
                if node not in self.node2m:
                    self.node2m[node] = (1 - self.beta1) * grad
                else:
                    self.node2m[node] = self.beta1 * self.node2m[node] + (1 - self.beta1) * grad

                if node not in self.node2v:
                    self.node2v[node] = (1 - self.beta2) * grad * grad
                else:
                    self.node2v[node] = self.beta2 * self.node2v[node] + (1 - self.beta2) * grad * grad

                self.prod_beta1 *= self.beta1
                self.prod_beta2 *= self.beta2
                m_hat = self.node2m[node] / (1 - self.prod_beta1)
                v_hat = self.node2v[node] / (1 - self.prod_beta2)

                node.data -= lr * m_hat / (np.sqrt(v_hat + EPSILON))

        runtime.grad_table = grad_table
        return grad_table


# 一些优化器
optimizers = {
    "SGD"     : SGD,
    "Momentum": Momentum,  # 动量优化器
    # "AdaGrad" : AdaGrad,      # 自适应梯度优化器
    # "RMSProp" : RMSProp,      # RMSProp优化器
    "Adam"    : Adam  # Adam优化器
}
