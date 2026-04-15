# -*- encoding:utf-8 -*-
# 用于注册基础运算

from collections.abc import MutableMapping
from typing import Iterator, TypeVar
from colorama import Back, Fore, Style
import numpy as np

KT = TypeVar("KT")  # key   type
VT = TypeVar("VT")  # value type


class Register(MutableMapping):
    """继承MutableMapping：实现标准字典的所有接口（增删改查）"""
    __slots__ = "__data"  # 限制类的属性，优化内存占用，仅允许__data属性

    def __init__(self, *args, **kwargs):
        """
        初始化注册器
        继承父类初始化，内部封装一个字典存储注册的对象 """
        super(Register, self).__init__(*args, **kwargs)
        self.__data = dict(*args, **kwargs)

    def register(self, target):
        """
        核心注册方法：支持装饰器用法，将可调用对象注册到注册器
        :param target: 待注册的目标（函数/类）
        :return: 注册后的目标对象 """

        def add_register_items(__k, __v):
            """内部函数：执行实际的注册逻辑"""
            if not callable(__v):
                raise ValueError("register object must be callable, but receive {}".format(type(__v)))
            # 校验：键已存在，打印警告并覆盖
            if __k in self.__data:
                back_print(
                    "warning: {} has been registered before, so we will overriden it".format(__v.__name__),
                    color = "red")
            # 将键值对存入注册器
            self[__k] = __v
            return __v

        return add_register_items(target.__name__, target) if callable(target) else lambda x: add_register_items(target, x)

    def __call__(self, *args, **kwargs):
        """ 重载()运算符：让注册器可以直接当装饰器使用"""
        return self.register(*args, **kwargs)

    def __setitem__(self, __k: KT, __v: VT) -> None:
        """字典赋值：obj[key] = value"""
        self.__data[__k] = __v

    def __getitem__(self, __k: KT) -> VT:
        """字典取值：obj[key]"""
        return self.__data.__getitem__(__k)

    def __contains__(self, __k: KT) -> bool:
        """判断键是否存在：key in obj"""
        return self.__data.__contains__(__k)

    def __delitem__(self, __v: VT) -> None:
        """删除键值对：del obj[key]"""
        return self.__data.__delitem__(__v)

    def __len__(self) -> int:
        """获取注册器长度：len(obj)"""
        return self.__data.__len__()

    def __iter__(self) -> Iterator[KT]:
        """迭代器：遍历注册器的键"""
        return self.__data.__iter__()

    def __str__(self) -> str:
        """打印注册器：输出内部字典内容"""
        return self.__data.__str__()

    def keys(self):
        """获取所有注册的键（名称）"""
        return self.__data.keys()

    def values(self):
        """获取所有注册的对象"""
        return self.__data.values()

    def items(self):
        """获取所有键值对：(名称, 对象)"""
        return self.__data.items()


r = Register()


def back_print(*args, color: str = None) -> None:
    """
    背景色打印函数
    :param args: 打印内容
    :param color: 背景色（red/yellow/green等）
    """
    if color is None:
        print(*args)
    else:
        print(getattr(Back, color.upper(), ""), *args, Style.RESET_ALL)


def fore_print(*args, color: str = None) -> None:
    """
    前景色（文字色）打印函数
    :param args: 打印内容
    :param color: 文字颜色
    """
    if color is None:
        print(*args)
    else:
        print(getattr(Fore, color.upper(), ""), *args, Style.RESET_ALL)


def numpy_one_hot(y: np.ndarray, class_num: int = None) -> np.ndarray:
    """
    将分类标签转换为独热编码（One-Hot）
    适用场景：深度学习分类任务的标签预处理
    :param y: 原始标签数组（形状任意，自动展平）
    :param class_num: 类别总数，默认自动计算标签唯一值数量
    :return: 独热编码数组 (样本数, 类别数)
    """
    y = y.reshape(-1)
    if class_num is None:
        class_num = len(set(y))
    else:
        y_unique = len(set(y))
        if y_unique < class_num:
            fore_print("#unique value in Y is smaller than class_num, though porgram can still be execuated, I advise you to check Y", color = "yellow")
        elif y_unique > class_num:
            raise ValueError("#unique value in Y is greater than class_num!")

    one_hot = np.zeros((y.shape[0], class_num))
    for i in range(y.shape[0]):
        one_hot[i][y[i]] = 1
    return one_hot
