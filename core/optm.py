# -*- encoding:utf-8 -*-
# 优化器


from collections import deque
import numpy as np
import abc

from core.base import Data, Operation, Placeholder, runtime

def backwards(op_node:Operation):
    """
    前馈
    """
    grad_table={}#梯度表
    grad_table[op_node]=1.
    visit_nodes=set()#已经便利的node
    queue=deque()#访问队列

    visit_nodes.add(op_node)
    queue.append(op_node)

    while len(queue)>0:#遍历所有
        cur_node=queue.popleft()#先遍历左节点(模型参数)
        if cur_node!=op_node and not (isinstance(cur_node,Placeholder)or isinstance(cur_node,Data)):
            grad_table[cur_node]=0
            for next_node in cur_node.next_nodes:
                grad_loss_wrt_next_node:np.ndarray=grad_table[next_node]
                next_node_op_name:str=next_node.__class__.__name__
                gradient_func=runtime.gradient_func[next_node_op_name]
                grad_loss_wrt_cur_node=gradient_func(next_node,grad_loss_wrt_next_node)

                if len(next_node.input_nodes)==1:
                    grad_table[cur_node]+=grad_loss_wrt_cur_node[0]
                else:
                    cur_node_in_next_node_index=next_node.input_nodes.index(cur_node)
                    grad_table[cur_node]+=grad_loss_wrt_cur_node[cur_node_in_next_node_index]
        if isinstance(cur_node,Operation):
            # TODO:continue
            pass



class Optimizer(abc.ABC ):
    def __init__(self,lr_rate:float=1e-3):
        self.lr_rate=lr_rate
    @abc.abstactmethod
    def minimize(self,loss_node:Operation):...

def SGD(Optimizer):
    pass

