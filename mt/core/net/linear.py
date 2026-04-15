# -*- encoding:utf-8 -*-
"""
线性层（全连接层）模块
实现深度学习最基础的全连接层（Linear/Dense Layer）
公式：output = X @ W + b （线性变换），可选激活函数
继承自nnVarOperator：带可训练参数的神经网络算子
"""
import numpy as np
import mt.core.base
from mt.core.base import Node, Variable, nnVarOperator, Placeholder
from mt.core.constant import runtime


class Linear(nnVarOperator):
    def __init__(self,
                 input_features: int,
                 output_features: int,
                 bias: bool = True,
                 activate_func: str = None,
                 init: str = 'randn'):
        """
        全连接层初始化，参数校验、权重/偏置初始化、激活函数绑定
        :param input_features: 输入特征维度（输入神经元数量）
        :param output_features: 输出特征维度（输出神经元数量）
        :param bias: 是否使用偏置项b，默认开启
        :param activate_func: 激活函数名称（需提前注册）
        :param init: 参数初始化方法（需提前注册）
        """
        # 调用父类初始化：生成唯一层名称，自动计数
        super().__init__()
        # 保存基础配置参数
        self.input_features = input_features
        self.output_features = output_features
        self.bias = bias
        # 校验激活函数：如果传入了激活函数，必须在全局注册的激活函数列表中
        if activate_func and activate_func not in runtime.activate_func:
            raise ValueError(
                f"input activate function '{activate_func}' is not in registered activate function "
                f"list:{list(runtime.activate_func.keys())}")
        # 校验初始化方法：必须在全局注册的初始化函数列表中
        if init and init not in runtime.init_func:
            raise ValueError(f"init not in registered init methods! "
                             f"Avaliable methods are {list(runtime.init_func.keys())}")

        self.init = init  # 保存初始化方法名称

        W = self.get_params(self.input_features, self.output_features)  # 调用初始化函数生成权重：形状 [输入维度, 输出维度]
        self.W = Variable(W, node_name = self.cur_name)  # 封装为Variable节点（可训练参数）

        if self.bias:
            b = self.get_params(1, self.output_features)  # 偏置形状 [1, 输出维度]，支持广播
            self.b = Variable(b, node_name = self.cur_name)

        self.act = activate_func  # 保存激活函数名称

    def get_params(self, input_size: int, output_size: int) -> np.ndarray:
        """
                获取初始化的参数矩阵（权重/偏置）
                :param input_size: 输入维度
                :param output_size: 输出维度
                :return: 初始化后的numpy数组
                """
        if self.init:
            return runtime.init_func[self.init]((input_size, output_size))  # 使用注册的初始化函数生成参数
        else:
            return np.random.randn(input_size, output_size)  # 兜底：默认标准正态分布初始化

    def __call__(self, X: Node) -> Node:
        """
        前向传播：执行线性变换 + 偏置 + 激活函数
        :param X: 输入节点（必须是计算图Node类型）
        :return: 层输出节点
        """
        if not isinstance(X, Node | Placeholder):  # 输入类型校验：必须是框架的Node节点
            raise ValueError("Linear's parameter X must be a Node!")
        out = mt.core.base.matmul(X, self.W, node_name = self.cur_name)  # 矩阵乘法 X @ W （线性变换核心）
        out = mt.core.base.add(out, self.b, node_name = self.cur_name)  # 加上偏置项 out = X@W + b

        if self.act:  # 如果指定了激活函数，执行激活
            act_func = runtime.activate_func[self.act]
            return act_func(out, node_name = self.cur_name)
        else:
            return out

    def reset_params(self):
        """
        重置层参数：重新初始化权重和偏置
        适用场景：模型重新训练、重置参数
        """
        W = self.get_params(self.input_features, self.output_features)  # 重新初始化权重
        self.W = Variable(W, node_name = self.cur_name)

        if self.bias:
            b = self.get_params(1, self.output_features)  # 重新初始化偏置
            self.b = Variable(b, node_name = self.cur_name)
