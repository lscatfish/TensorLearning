# -*- encoding:utf-8 -*-
# 线性连接层
import numpy as np
from core.base import Node,Variable,nnVarOperator
from core.util import back_print
from core.constant import runtime

class Linear(nnVarOperator):
    def __init__(self,input_features:int,output_features,bias:bool=True,activate_func:str=None,init:str=''):
        super().__init__()
        self.input_features=input_features
        self.output_features=output_features
        self.bias=bias
        if activate_func and activate_func not in runtime.activate_func:
            raise ValueError(f"input activate function '{activate_func}' is not in registered activate function list:{list(runtime.activate_func.keys())}")
        if init and init not in runtime.init_func:
            raise ValueError(f"init not in registered init methods! Avaliable methods are {list(runtime.init_func.keys())}")
        self.init=init

        pass



