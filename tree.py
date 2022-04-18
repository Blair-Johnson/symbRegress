import torch
import torch.nn as nn
import torch.nn.functional as nf

from typing import Callable, Tuple

class SyntaxNode(object):
    # stored op properties in class attribute
    n_args = {
        "+": 2,
        "-": 2,
        "*": 2,
        "/": 2,
        "sin": 1,
        "cos": 1,
        "exp": 1,
        "log": 1,
        "var": 0,
        "const": 0}

    op_list = {
        "+": torch.add,
        "-": torch.subtract,
        "*": torch.multiply,
        "/": torch.divide,
        "sin": torch.sin,
        "cos": torch.cos,
        "exp": torch.exp,
        "log": torch.log,
        "var": None,
        "const": torch.rand(1, requires_grad=True)}

    illegal = {
        '+': [],
        '-': [],
        '*': [],
        '/': [],
        'sin': ['sin', 'cos', 'exp', 'log'],
        'cos': ['sin', 'cos', 'exp', 'log'],
        'exp': ['sin', 'cos', 'exp', 'log'],
        'var': [],
        'const': []}

    def __init__(self, op: str, parent = None):
        self.left = None
        self.right = None
        self.parent = parent
        assert op in SyntaxNode.op_list.keys()
        self.value = op
            
        # operation, number of arguments
        self.op = SyntaxNode.op_list[op]
        self.n_args = SyntaxNode.n_args[op]

    def __bool__(self):
        return True

    def add_node(self, node_value : str) -> bool:
        if self.n_args != 0:
            if self.left == None:
                self.left = SyntaxNode(node_value, self)
                return True
            elif (self.n_args != 1) and (self.right == None):
                self.right = SyntaxNode(node_value, self)
                return True
            else:
                return False
        else:
            return False

    def from_preorder(self, node_list: list) -> list:
        if self.n_args == 2:
            self.left = SyntaxNode(node_list.pop(0))
            node_list = self.left.from_preorder(node_list)
            self.right = SyntaxNode(node_list.pop(0))
            return self.right.from_preorder(node_list)
        elif self.n_args == 1:
            self.left = SyntaxNode(node_list.pop(0))
            return self.left.from_preorder(node_list)
        else:
            return node_list

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
        traversal = [self.value]
        if self.left != None:
            traversal += self.left.get_preorder()
        if self.right != None:
            traversal += self.right.get_preorder()
        return traversal

    def get_last(self):
        if self.n_args == 2:
            if self.left != None:
                res = self.left.get_last()
                if res:
                    return res
            else:
                return self
            if self.right != None:
                res = self.right.get_last()
                if res:
                    return res
                else:
                    return False
            else:
                return self
        elif self.n_args == 1:
            if self.left != None:
                return self.left.get_last()
            else:
                return self
        else:
            return False


def get_expression(root:SyntaxNode) -> str:
    if root.left == None and root.right == None:
        return root.value
    elif root.right == None:
        return root.value + '(' + get_expression(root.left) + ')'
    else:
        return '('+ get_expression(root.left) + root.value + get_expression(root.right) + ')'       

def tree_from_preorder(preorder : list) -> SyntaxNode:
    root = SyntaxNode(preorder.pop(0))
    for node_id in preorder:
        root.add_node(node_id)
    return root

if __name__ == '__main__':
    # lambda x: exp(ax + b)
    root = SyntaxNode("exp") # exp
    root.add_node("+") # exp( + )
    root.left.add_node("*") # exp(( * ) + )
    root.left.add_node("const") # exp(( * ) + b)
    root.left.left.add_node("const") # exp((a * ) + b)
    root.left.left.add_node("var") # exp((a * x) + b)
    x = torch.tensor(3.5)
    print(root.get_function(x))
    print(root.get_preorder())
    print(get_expression(root))
    print('reconstruction test')
    root2 = SyntaxNode("exp")
    assert root2.from_preorder(['+','*','const','var','const']) == []
    print(get_expression(root2))
    print('last node test')
    print('const + exp(?)')
    preorder = ['+','const','exp']
    tree = tree_from_preorder(preorder)
    print(tree.get_last().op)

    
