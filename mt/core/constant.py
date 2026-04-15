# -*- encoding:utf-8 -*-
# 运行时常量


import numpy as np
from mt.core.util import Register
from collections import defaultdict

EPSILON = 0.00001


# 防止计算溢出
class Clip:
    PRECISE_LOW = 1e-127
    PRECISE_HIGH = 1e128
    EXP_PRECISE_LOW = -292.42
    EXP_RPECISE_HIGH = 294.73
    EXP = lambda x: np.exp(np.clip(x, Clip.EXP_PRECISE_LOW, Clip.EXP_RPECISE_HIGH))


# 定义注册器，运行时图
class runtime:
    activate_func = Register()
    gradient_func = Register()
    init_func = Register()
    global_calc_graph = list()
    nn_cnt = defaultdict(int)
    grad_table = None
