# -*- encoding:utf-8 -*-
# 优化器
from typing import Dict

import numpy as np
import abc
from collections import deque
from core.base import Data, Operation, Placeholder, runtime, Variable, Node


def backwards(op_node: Operation) -> Dict[Operation | Node | Variable, float]:
    """
    前馈
    """
    grad_table = {op_node: 1.}  # 梯度表
    visit_nodes = set()  # 已经便利的node
    queue = deque()  # 访问队列
    visit_nodes.add(op_node)
    queue.append(op_node)

    while len(queue) > 0:  # 遍历所有
        cur_node = queue.popleft()  # 先遍历左节点(模型参数)
        if cur_node != op_node and not (isinstance(cur_node, Placeholder) or isinstance(cur_node, Data)):
            grad_table[cur_node] = 0
            for next_node in cur_node.next_nodes:
                grad_loss_wrt_next_node: np.ndarray = grad_table[next_node]
                next_node_op_name: str = next_node.__class__.__name__
                gradient_func = runtime.gradient_func[next_node_op_name]
                grad_loss_wrt_cur_node = gradient_func(next_node, grad_loss_wrt_next_node)

                if len(next_node.input_nodes) == 1:
                    grad_table[cur_node] += grad_loss_wrt_cur_node[0]
                else:
                    cur_node_in_next_node_index = next_node.input_nodes.index(cur_node)
                    grad_table[cur_node] += grad_loss_wrt_cur_node[cur_node_in_next_node_index]
        if isinstance(cur_node, Operation):
            for input_node in cur_node.input_nodes:
                if input_node not in visit_nodes:
                    visit_nodes.add(input_node)
                    queue.append(input_node)
    return grad_table


class Optimizer(abc.ABC):
    def __init__(self, lr_rate: float = 1e-3):
        self.learning_rate = lr_rate

    @abc.abstractmethod
    def minimize(self, loss_node: Operation):
        ...


class SGD(Optimizer):
    def __init__(self, learning_rate: float = 1e-3):
        super().__init__(lr_rate = learning_rate)

    def minimize(self, loss_node: Operation):
        lr = self.learning_rate
        grad_table = backwards(op_node = loss_node)
        for node in grad_table:
            if isinstance(node, Variable):
                grad = grad_table[node]
                node.data -= lr * grad

        runtime.grad_table = grad_table
        return grad_table


# 一些优化器
optimizers = {
    "SGD": SGD,
    # "Momentum" : Momentum,
    # "AdaGrad" : AdaGrad,
    # "RMSProp" : RMSProp,
    # "Adan" : Adam
}
