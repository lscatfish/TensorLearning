# -*- encoding:utf-8 -*-
"""
实现深度学习静态计算图框架
构建计算图 → 管理变量/常量/占位符 → 会话执行前向传播
"""

import numpy as np
from typing import Union, List, Dict
import abc
from core.constant import runtime
from core.util import back_print, fore_print


# ====================== 预声明类 ======================
class Node: ...


class add: ...


class minus: ...


class negative: ...


class multiply: ...


class matmul: ...


class elementwise_pow: ...


# 输入占位：运行时通过feed_dict传入数据，无默认值
class Placeholder(Node): ...


class Node(abc.ABC):
    """核心节点基类
    所有计算节点的抽象父类
    完成定义通用属性于运算符重载
    """

    def __init__(self, node_name: str = ""):
        """
        节点初始化：自动加入全局计算图，初始化基础属性
        :param node_name: 节点自定义名称（调试用）
        """
        self.next_nodes = []  # 后继节点列表：存储依赖当前节点的操作
        self.data = None  # 节点计算结果：会话运行后才会赋值
        self.node_name = node_name  # 节点标识名称
        runtime.global_calc_graph.append(self)  # 将节点加入全局计算图

    # ====================== 运算符重载 ======================
    def __neg__(self):
        """-node → 返回负号操作节点"""
        return negative(self)

    def __add__(self, node: Node):
        """node1 + node2 → 返回加法操作节点"""
        return add(self, node)

    def __sub__(self, node: Node):
        """node1 - node2 → 返回减法操作节点"""
        return minus(self, node)

    def __mul__(self, node: Node):
        """node1 * node2 → 返回元素乘节点"""
        return multiply(self, node)

    def __pow__(self, y: Union[int, float]):
        """node ** 2 → 返回逐元素幂节点"""
        return elementwise_pow(self, y)

    def __matmul__(self, node: Node):
        """node1 @ node2 → 返回矩阵乘节点"""
        return matmul(self, node)

    def __str__(self):
        """打印节点"""
        return "{}({})".format(self.__class__.__name__, str(self.data))

    @property
    def numpy(self):
        """节点数据转为numpy数组"""
        if self.data is None:
            raise ValueError(f"the node {self.__class__} is empty, please run the session first!")
        return np.array(self.data)

    @property
    def shape(self):
        if self.data is None:
            raise ValueError(f"the node {self.__class__} is empty, please run the session first!")
        return np.array(self.data).shape

    # 显式转为numpy数组
    def to_numpy(self):
        if self.data is None:
            raise ValueError(f"the node {self.__class__} is empty, please run the session first!")
        return np.array(self.data)

    # 显式转为Python列表
    def to_list(self):
        if self.data is None:
            raise ValueError(f"the node {self.__class__} is empty, please run the session first!")
        return list(self.data)


class Operation(Node, abc.ABC):
    """计算操作的抽象类"""

    def __init__(self, input_nodes: List = None, node_name: str = ""):
        """
        操作节点初始化：绑定输入节点，构建计算图依赖关系
        :param input_nodes: 该操作的输入节点列表
        """
        super().__init__(node_name = node_name)
        self.input_nodes = input_nodes if input_nodes is not None else []
        # 为输入节点绑定当前操作（建立节点依赖）
        for node in input_nodes:
            node.next_nodes.append(self)

    @abc.abstractmethod
    def compute(self, *args):
        """强制子类实现计算逻辑"""


class Data(Node):
    """常量节点, 存储固定数据，不可修改"""

    def __init__(self, data, node_name: str = ""):
        super().__init__(node_name)
        # 强制数据类型为numpy数组
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a numpy array!")
        self.data = data


class Variable(Node):
    """变量, 存储训练参数"""

    def __init__(self, init_value: Union[np.ndarray, List] = None, node_name: str = ""):
        super().__init__(node_name = node_name)
        self.data = init_value  # 初始值，后续可更新


class Session(object):
    """会话
    遍历全局计算图，执行前向传播，填充所有节点的data"""

    def run(self, root_op: Operation, feed_dict = None, use_batch: bool = False):
        """
        执行单次前向传播
        :param root_op: 最终要获取结果的根操作节点
        :param feed_dict: 占位符赋值字典 {Placeholder: 数据}
        :param use_batch: 是否为批量模式
        :return: 计算完成后的根节点
        """
        if feed_dict is None:
            feed_dict = {}

        # 遍历全局计算图，按节点类型赋值/计算
        for node in runtime.global_calc_graph:
            # 变量节点：统一转为np 格式
            if isinstance(node, Variable):
                node.data = np.array(node.data)
            # 占位符节点：从字典中读取数据
            elif isinstance(node, Placeholder):
                node.data = np.array(feed_dict[node])
            # 常量节点：无需处理
            elif isinstance(node, Data):
                pass
            # 操作节点：执行compute计算
            elif isinstance(node, Operation):
                input_datas = [n.data for n in node.input_nodes]
                # 批量模式：追加结果；普通模式：直接赋值
                if use_batch:
                    node.data.append(node.compute(*input_datas))
                else:
                    node.data = node.compute(*input_datas)
            else:
                raise TypeError("Unknown node type in global calcualtion graph: {}".format(type(node)))
        return root_op

    def run_batch(self, root_op: Operation, feed_dict: Dict = None):
        """
        执行批量前向传播（逐批次喂数据）
        :param root_op: 根操作
        :param feed_dict: 批量数据字典，value为批次列表
        """
        if feed_dict is None:
            feed_dict = {}
        if len(feed_dict) == 0:
            raise ValueError("feed_dict must contain something!")

        # 校验所有占位符的批次大小一致
        flag = self.check_feed_dict(feed_dict)
        if not flag:
            raise ValueError("input placeholder must be the same!")

        # 获取批次大小
        batch_size = -1
        for k in feed_dict:
            batch_size = len(feed_dict[k])
            break

        # 初始化：清空操作节点的结果列表
        for node in runtime.global_calc_graph:
            if isinstance(node, Operation):
                node.data = []

        # 逐批次执行计算
        for i in range(batch_size):
            one_batch = {k: feed_dict[k][i] for k in feed_dict}
            self.run(root_op = root_op, feed_dict = one_batch)

    def check_feed_dict(self, feed_dict: dict) -> bool:
        """校验批量数据的批次大小是否统一"""
        batch_size = -1
        for k in feed_dict:
            if batch_size == -1:
                batch_size = len(feed_dict[k])
            elif batch_size != len(feed_dict[k]):
                return False
        return True


class add(Operation):
    """加法操作"""

    def __init__(self, x: Node, y: Node, node_name: str = ""):
        super().__init__(input_nodes = [x, y], node_name = node_name)

    def compute(self, x_v: np.ndarray, y_v: np.ndarray):
        return x_v + y_v


class minus(Operation):
    """减法操作"""

    def __init__(self, x: Node, y: Node, node_name: str = ""):
        super().__init__(input_nodes = [x, y], node_name = node_name)

    def compute(self, x_v: np.ndarray, y_v: np.ndarray):
        return x_v - y_v


class negative(Operation):
    """负号操作"""

    def __init__(self, x: Node, node_name: str = ""):
        super().__init__(input_nodes = [x], node_name = node_name)

    def compute(self, x_v: np.ndarray):
        return -1. * x_v


class elementwise_pow(Operation):
    """逐元素幂操作"""

    def __init__(self, x: Node, y: Union[int, float], node_name: str = ""):
        super().__init__(input_nodes = [x], node_name = node_name)
        self.y = y  # 幂次参数

    def compute(self, x_v: np.ndarray):
        return x_v ** self.y


class matmul(Operation):
    """矩阵乘法操作"""

    def __init__(self, x: Node, y: Node, node_name: str = ""):
        super().__init__(input_nodes = [x, y], node_name = node_name)

    def compute(self, x_v: np.ndarray, y_v: np.ndarray):
        return x_v @ y_v


class multiply(Operation):
    """逐元素乘法操作"""

    def __init__(self, x: Node, y: Node, node_name: str = ""):
        super().__init__(input_nodes = [x, y], node_name = node_name)

    def compute(self, x_v: np.ndarray, y_v: np.ndarray):
        return x_v * y_v


class reduce_sum(Operation):
    """求和操作"""

    def __init__(self, x: Node, axis: int = None, node_name: str = ""):
        super().__init__(input_nodes = [x], node_name = node_name)
        self.axis = axis  # 求和维度

    def compute(self, x_v: np.ndarray):
        return np.sum(x_v, axis = self.axis)


class reduce_mean(Operation):
    """ 均值操作 """

    def __init__(self, x: Node, axis: int = None, node_name: str = ""):
        super().__init__(input_nodes = [x], node_name = node_name)
        self.axis = axis  # 均值维度

    def compute(self, x_v: np.ndarray):
        return np.mean(x_v, axis = self.axis)


class log(Operation):
    """对数操作"""

    def __init__(self, x: Node, node_name: str = ""):
        super().__init__(input_nodes = [x], node_name = node_name)

    def compute(self, x_v: np.ndarray):
        # 非法值校验：对数输入不能≤0
        if (x_v <= 0).any():
            back_print("Oops, invalid value encountered in 'log', I guess you forget activation function", color = "yellow")
        return np.log(x_v)
