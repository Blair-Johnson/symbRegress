import torch
import torch.nn as nn
import torch.nn.functional as nf
from tree import *

from torch.distributions.categorical import Categorical
from tree import SyntaxNode, tree_from_preorder

N_SAMPLES = 20

class SyntaxTreeLSTM(nn.Module):
    
    OP_NAMES = ['+',
                '-',
                '*',
                '/',
                'sin',
                'cos',
                'exp',
                'log',
                'var',
                'const',
                'start',
                'none']

    def __init__(self, n_samples: int):
        super(SyntaxTreeLSTM, self).__init__()
        self.n_samples = n_samples

        self.conv1 = nn.Conv1d(1, 32, 7, padding=(0))
        self.conv2 = nn.Conv1d(32, 64, 7, padding=(0))
        self.conv3 = nn.Conv1d(64, 128, 7, padding=(0))
        # (1, 128, 20)
        self.flatten = nn.Flatten()
        self.lstm1 = nn.LSTMCell(256, 128)
        self.lstm2 = nn.LSTMCell(128, 128)
        self.class_head = nn.Linear(128, 9)
        

    def stop_criterion(self, tree: SyntaxNode) -> bool:
        '''Test to see if tree is complete'''
        leaf_list = tree.get_leaf_nodes()
        for leaf in leaf_list:
            if leaf.n_args != 0:
                return False
        return True

    def update_tree(self, tree: SyntaxNode, model_dist: torch.tensor) -> Tuple[SyntaxTree, 
                                                                               torch.tensor,
                                                                               torch.tensor]:
        '''Sample new token from distribution, ignore illegal choices, update tree'''
        # make model_dist probabilities
        model_dist = nf.softmax(model_dist, axis=0)
        # get last node in tree
        last_node = tree.get_last()
        # get list of node names thay may not follow the last node
        illegal_nodes = last_node.illegal[last_node.value]
        # convert node names to node indices
        illegal_nodes = [OP_NAMES.find(name) for name in illegal_nodes]
        # zero out the logit probabilities for illegal next nodes
        for index in illegal_nodes:
            model_dist[index] = 0
        # get the tree preorder
        preorder = tree.get_preorder()
        # sample a new node from the logit distribution and add to preorder
        dist = Categorical(model_dist)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        preorder.append(OP_NAMES[action.item()])
        # reconstruct tree from the updated preorder
        tree = tree_from_preorder(preorder)
        return tree, action, log_prob

    def get_parent_sibling(self, tree: SyntaxNode) -> torch.tensor:
        '''take tree and return the parent and sibling of next node in traversal'''
        

    def forward(self, x: torch.tensor) -> list:

        out = nf.elu(self.conv1(x))
        out = nf.elu(self.conv2(out))
        out = nf.elu(self.conv3(out))
        out = self.flatten(out)

        out = torch.cat((out, torch.zeros_like(out)), dim=1)
        
        hidden_state_1, cell_state_1 = self.lstm1(out, (hidden_state_1, cell_state_1))
        hidden_state_2, cell_state_2 = self.lstm2(hidden_state_1, (hidden_state_2, cell_state_2))
        node_logits = self.class_head(hidden_state_2)
        syntax_tree = SyntaxNode(torch.argmax(node_logits))

        while not self.stop_criterion(syntax_tree):
            parent_sibling = self.get_parent_sibling(syntax_tree)
            hidden_state_1, cell_state_1 = self.lstm1(parent_sibling, (hidden_state_1, cell_state_1))
            hidden_state_2, cell_state_2 = self.lstm2(hidden_state_1, (hidden_state_2, cell_state_2))
            node_logits = self.class_head(hidden_state_2)
            syntax_tree = self.update_tree(syntax_tree, node_logits)
        return syntax_tree


if __name__ == '__main__':
    test = torch.ones(1,1,20)
    model = SyntaxTreeLSTM(20)
    print(model(test).shape) # 1,128,20
