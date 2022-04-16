import torch
import torch.nn as nn
import torch.nn.functional as nf

from typing import Callable

class SyntaxNode(object):
    def __init__(self, op_idx: int, parent = None):
        self.left = None
        self.right = None
        self.parent = parent
        self.op_idx = op_idx
        
        # stored op properties
        arg_list = [2,2,2,2,1,1,1,1,0,0]
        op_list = [torch.add,
                        torch.subtract,
                        torch.multiply,
                        torch.divide,
                        torch.sin,
                        torch.cos,
                        torch.exp,
                        torch.log,
                        None,
                        torch.rand(1, requires_grad=True)]
        
        # operation, numper of arguments
        self.op = op_list[self.op_idx]
        self.n_args = arg_list[self.op_idx]

    def add_node(self, node_idx : int) -> bool:
        if self.n_args != 0:
            if self.left == None:
                self.left = SyntaxNode(node_idx, self)
                return True
            elif (self.n_args != 1) and (self.right == None):
                self.right = SyntaxNode(node_idx, self)
                return True
            else:
                return False
        else:
            return False

    def get_function(self, X: torch.Tensor) -> Callable:
        if self.n_args == 1:
            return self.op(self.left.get_function(X))
        elif self.n_args == 2:
            return self.op(self.left.get_function(X),
                           self.right.get_function(X))
        else:
            if self.op != None:
                return self.op
            else:
                return X

    def get_preorder(self) -> list:
        indices = [self.op_idx]
        if self.left != None:
            indices += self.left.get_preorder()
        if self.right != None:
            indices += self.right.get_preorder()
        return indices

if __name__ == '__main__':
    # lambda x: exp(ax + b)
    root = SyntaxNode(6) # exp
    root.add_node(0) # exp( + )
    root.left.add_node(2) # exp(( * ) + )
    root.left.add_node(9) # exp(( * ) + b)
    root.left.left.add_node(9) # exp((a * ) + b)
    root.left.left.add_node(8) # exp((a * x) + b)
    x = torch.tensor(3.5)
    print(root.get_function(x))
    print(root.get_preorder())


    
