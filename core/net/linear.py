# -*- encoding:utf-8 -*-
# 线性连接层
import numpy as np
import core.base
import core
from core.base import Node,Variable,nnVarOperator
import core.base
from core.util import back_print
from core.constant import runtime

class Linear(nnVarOperator):
    def __init__(self,input_features:int,output_features,bias:bool=True,activate_func:str=None,init:str='randn'):
        super().__init__()
        self.input_features=input_features
        self.output_features=output_features
        self.bias=bias
        if activate_func and activate_func not in runtime.activate_func:
            raise ValueError(f"input activate function '{activate_func}' is not in registered activate function list:{list(runtime.activate_func.keys())}")
        if init and init not in runtime.init_func:
            raise ValueError(f"init not in registered init methods! Avaliable methods are {list(runtime.init_func.keys())}")
        self.init=init
        
        W=self.get_params(self.input_features,self.output_features)
        self.W=Variable(W,node_name=self.cur_name)

        if self.bias:
            b=self.get_params(1,self.output_features)
            self.b=Variable(b,node_name=self.cur_name)

        self.act=activate_func

    def get_params(self,input_size,output_size)->np.ndarray:
        if self.init:
            return runtime.init_func[self.init]((input_size,output_size))
        else:
            return np.random.randn(input_size,output_size)
    def __call__(self,X:Node):
        if not isinstance(X,Node):
            raise ValueError("Linear's parameter X must be a Node!")
        out =core.base.matmul(X,self.W,node_name=self.cur_name)
        out =core.base.add(out,self.b,node_name=self.cur_name)

    def reset_params(self):
        W=self.get_params(self.input_features,self.output_features)
        self.W=Variable(W,node_name=self.cur_name)

        if self.bias:
            b=self.get_params(1,self.output_features)
            self.b=Variable(b,node_name=self.cur_name)

