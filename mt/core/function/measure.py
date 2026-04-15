# -*- encoding:utf-8 -*-
"""
损失函数模块，实现深度学习中最常用的两种损失函数
交叉熵损失 (Cross Entropy)：分类任务专用
均方误差损失 (MSE)：回归任务专用
均基于自定义计算图节点实现
"""

from mt.core.base import Node, nnOperator, reduce_mean, log, reduce_sum, multiply, negative


# ====================== 交叉熵损失（类实现） ======================
# 继承nnOperator：作为神经网络层算子，适配框架计算图
class CrossEntropy(nnOperator):
    # 内置归一化方式映射：字符串 -》 对应的计算图操作节点
    __reduction__ = {
        "mean": reduce_mean,  # 均值归一化（默认常用）
        "sum" : reduce_sum  # 求和归一化
    }

    def __init__(self, reduction: str = "sum") -> None:
        """
        初始化交叉熵损失
        :param reduction: 损失归一化方式，可选 mean/sum
        """
        super().__init__()  # 调用父类初始化，生成唯一算子名称
        # 校验传入的归一化方式是否合法
        if reduction not in self.__reduction__:
            raise ValueError("{} not in available reduction function:{}".format(reduction, self.__reduction__))
        self.reduction = reduction  # 保存归一化方式

    def __call__(self, predict: Node, label: Node):
        """
        前向计算交叉熵损失（核心逻辑）
        CrossEntropy = - Σ (label * log(predict))
        :param predict: 模型预测值节点（经过softmax的概率值）
        :param label: 真实标签节点（独热编码格式）
        :return: 交叉熵损失计算节点
        """
        # 第一步：对预测值取对数 log(predict)
        p_pre = log(predict, node_name = self.cur_name)
        # 第二步：标签 × 对数预测值 label * log(predict)
        p_pre = multiply(label, p_pre, node_name = self.cur_name)
        # 第三步：按指定方式求和/均值
        reduce_p = self.__reduction__[self.reduction](p_pre, node_name = self.cur_name)
        # 第四步：取负数，得到最终交叉熵损失
        return negative(reduce_p, node_name = self.cur_name)


# ====================== 交叉熵损失（函数简化实现） ======================
def cross_entropy(predict: Node, label: Node, reduction: str = "mean"):
    """
    函数式交叉熵损失
    :param predict: 预测节点
    :param label: 标签节点
    :param reduction: 归一化方式，默认mean
    :return: 损失节点
    """
    __reductions__ = ["mean", "sum"]  # 支持的归一化方式
    if reduction == "mean":
        # 均值交叉熵：-mean(label * log(predict))
        return - reduce_mean(label * log(predict))
    elif reduction == "sum":
        # 求和交叉熵：-sum(label * log(predict))
        return - reduce_sum(label * log(predict))
    else:
        # 非法参数抛出异常
        raise Exception(f"reduction only receive {__reductions__}")


# ====================== 均方误差损失（MSE） ======================
def mean_square_error(predict: Node, label: Node, reduction: str = "mean"):
    """
    均方误差损失 (MSE)，回归任务专用
    MSE = mean((predict - label)²)
    :param predict: 模型预测值节点
    :param label: 真实标签节点
    :param reduction: 归一化方式，默认mean
    :return: MSE损失节点
    """
    __reductions__ = ["mean", "sum"]  # 支持的归一化方式
    if reduction == "mean":
        # 均值MSE：mean((预测值-标签值)²)
        return reduce_mean((predict - label) ** 2)
    elif reduction == "sum":
        # 求和MSE：sum((预测值-标签值)²)
        return reduce_sum((predict - label) ** 2)
    else:
        # 非法参数抛出异常
        raise Exception(f"reduction only receive {__reductions__}")
