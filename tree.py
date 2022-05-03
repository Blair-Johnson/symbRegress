import torch
import torch.nn as nn
import torch.nn.functional as nf

from typing import Callable, Tuple, Union, List

# TODO: Need the ability to 'append' a new node in preorder without reconstructing the whole tree
# TODO: Need to add member variable for node embeddings within tree
# TODO: Need to get tree depth -> Done

class SyntaxNode(object):
    # stored op properties in class attribute
    n_args = {
        "+": 2,
        "-": 2,
        "*": 2,
        "/": 2,
        #"^": 2,
        "sin": 1,
        "cos": 1,
        "exp": 1,
        "log": 1,
        "var": 0,
        "const": 0,
        "start": 1}

    op_list = {
        "+": torch.add,
        "-": torch.subtract,
        "*": torch.multiply,
        "/": torch.divide,
        #"^": torch.pow,
        "sin": torch.sin,
        "cos": torch.cos,
        "exp": torch.exp,
        "log": torch.log,
        "var": None,
        "const": torch.rand,
        "start": lambda x: x}

    illegal = {
        '+': [],
        '-': [],
        '*': [],
        '/': [],
        #'^': ['exp', 'log'],
        'sin': ['sin', 'cos', 'exp', 'log'],
        'cos': ['sin', 'cos', 'exp', 'log'],
        'exp': ['sin', 'cos', 'exp', 'log'],
        'var': [],
        'const': [],
        'start': []}

    # TODO: Probably a better way to track parameters, potentially a memory leak
    parameters = {}
    
    instance_counter = 0
    
    def __init__(self, op: str, parent:'SyntaxNode' = None, data:torch.tensor = None):
        self.left = None
        self.right = None
        self.tree_idx = None
        self.parent = parent
        self.data = data
        assert op in SyntaxNode.op_list.keys()
        self.value = op
        self.illegal = SyntaxNode.illegal[self.value]
            
        # operation, number of arguments
        self.op = SyntaxNode.op_list[op]
        self.n_args = SyntaxNode.n_args[op]
        
        # update parameters
        if self.value == 'start':
            SyntaxNode.instance_counter += 1
            self.tree_idx = SyntaxNode.instance_counter
            SyntaxNode.parameters[self.tree_idx] = []
        elif self.value == 'const':
            self.tree_idx = self.parent.tree_idx
            self.op = self.op(1, requires_grad=True)
            SyntaxNode.parameters[self.tree_idx].append(self.op)
        else:
            self.tree_idx = self.parent.tree_idx

    def __bool__(self):
        return True
    
    def __del__(self):
        if self.value == 'start':
            if self.tree_idx in SyntaxNode.parameters.keys():
                #print(f'deleting {self.tree_idx} from param keys: {SyntaxNode.parameters.keys()}')
                del SyntaxNode.parameters[self.tree_idx]

    def add_node(self, node:'SyntaxNode') -> bool:
        if self.n_args != 0:
            if self.left == None:
                self.left = node
                return True
            elif (self.n_args != 1) and (self.right == None):
                self.right = node
                return True
            else:
                return False
        else:
            return False

    def from_preorder(self, node_list: list) -> list:
        if self.n_args == 2:
            self.left = SyntaxNode(node_list.pop(0), self)
            node_list = self.left.from_preorder(node_list)
            self.right = SyntaxNode(node_list.pop(0), self)
            return self.right.from_preorder(node_list)
        elif self.n_args == 1:
            self.left = SyntaxNode(node_list.pop(0), self)
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
            if (self.op != None):
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

    def get_leaf_nodes(self) -> list:
        leaf_list = []
        if self.left is None and self.right is None:
            leaf_list.append(self)
        else:
            leaf_list += self.left.get_leaf_nodes()
            if self.right is not None:
                leaf_list += self.right.get_leaf_nodes()
        return leaf_list
    
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
        
    def append(self, node_type:str, data:torch.tensor = None) -> bool:
        last_parent = self.get_last()
        return last_parent.add_node(SyntaxNode(node_type, parent=self, data=data))

def get_tree_depth(root:SyntaxNode) -> int:    
    if root == None:
        return 0
    left_depth = get_tree_depth(root.left)
    right_depth = get_tree_depth(root.right)
    return 1 + max(left_depth, right_depth)

def tree_from_preorder(preorder:list) -> SyntaxNode:
    root = SyntaxNode(preorder.pop(0))
    root.from_preorder(preorder)
    return root

def get_expression(root:SyntaxNode) -> str:
    # TODO: Fix this implementation, doesn't work properly with incomplete expressions
    if root.value != 'start':
        if root.left == None and root.right == None:
            if (root.value != 'const') and (root.value != 'var'):
                return root.value
            elif root.value == 'var':
                return 'x'
            else:
                return str(round(root.op.item(),3))
        elif root.right == None:
            return root.value + '(' + get_expression(root.left) + ')'
        else:
            return '('+ get_expression(root.left) + root.value + get_expression(root.right) + ')'
    else:
        return get_expression(root.left)

def tree_complete(root: SyntaxNode) -> bool:
    ''' Test to see if tree is complete'''
    leaf_list = tree.get_leaf_nodes()
    for leaf in leaf_list:
        if leaf.n_args != 0:
            return False
    return True

if __name__ == '__main__':
    # lambda x: exp(ax + b)
    # root = SyntaxNode("exp") # exp
    # root.add_node("+") # exp( + )
    # root.left.add_node("*") # exp(( * ) + )
    # root.left.add_node("const") # exp(( * ) + b)
    # root.left.left.add_node("const") # exp((a * ) + b)
    # root.left.left.add_node("var") # exp((a * x) + b)
    # x = torch.tensor(3.5)
    # print(root.get_function(x))
    # print(root.get_preorder())
    # print(get_expression(root))

    print('reconstruction test:')
    root2 = SyntaxNode("start")
    assert root2.from_preorder(['exp','+','*','const','var','const']) == []
    print(get_expression(root2))

    print('Last node test:')
    print('const + exp(?)')
    preorder = ['start','+','const','exp']
    tree = SyntaxNode(preorder.pop(0))
    _ = [tree.append(node) for node in preorder]
    print(tree.get_last().op)

    print("Non-empty list test:")
    root3 = SyntaxNode("start")
    l = root3.from_preorder(['exp','+','*','const','var','const'])
    print("l:", l)
    print("root3:", get_expression(root3))
    
    print("Leaf nodes test:")    
    leafs = root3.get_leaf_nodes()
    for l in leafs:
        print(l.value, l.n_args)
        
    print("Parameter Tracking Test:")
    root2.__del__()
    root3.__del__()
    tree.__del__()
    root4 = SyntaxNode("start")
    root4.from_preorder(['+','*','const','var','const'])
    print(SyntaxNode.parameters)
    print(SyntaxNode.parameters.keys())
    
    print("Fit test")
    root4.__del__()
    root = SyntaxNode('start')
    root.from_preorder(['exp', '+','*','const','var','const'])
    print("Initial Expression:")
    print(get_expression(root))
    print("Target Expression: exp(2.5*x + 6.34)")
    #_ = [param.cuda() for param in SyntaxNode.parameters[root.tree_idx]]
    func_optim = torch.optim.Adam(SyntaxNode.parameters[root.tree_idx], .1)
    def func(x):
        return torch.exp(2.5*x + 6.34)
    X = torch.rand((1,20)) #.cuda()
    y = func(X)
    for step in range(1000):
        func_optim.zero_grad()
        y_hat = root.get_function(X)
        func_loss = torch.mean((y_hat - y)**2)
        func_loss.backward()
        if step % 100 == 0:
            print(func_loss.item())
        func_optim.step()
    print("Fit Expression:")
    print(get_expression(root))
    
    print("Append Test:")
    root.__del__()
    root = SyntaxNode('start')
    root.append('exp')
    root.append('+')
    root.append('*')
    root.append('const')
    root.append('var')
    root.append('const')
    print(get_expression(root))