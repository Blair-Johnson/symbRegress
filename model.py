import torch
import torch.nn as nn
import torch.nn.functional as nf
from tree import *

N_SAMPLES = 20

class SyntaxTreeLSTM(nn.Module):
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
        
    def stop_criterion(self, node_list: list) -> bool:
        '''Test to see if tree is complete'''
        root = SyntaxNode(node_list.pop(0))
        node_list = root.from_preorder(node_list) # Build tree

        leaf_list = root.get_leaf_nodes()
        for leaf in leaf_list:
            if leaf.n_args != 0:
                return False
        return True

    def update_tree(self, tree: list, node_logits: torch.tensor) -> list:
        '''Sample new token from distribution, ignore illegal choices, update tree'''

    def get_parent_sibling(self, tree: list) -> torch.tensor:
        '''take tree and return the parent and sibling of next node in traversal'''

    def forward(self, x: torch.tensor) -> list:

        out = nf.elu(self.conv1(x))
        out = nf.elu(self.conv2(out))
        out = nf.elu(self.conv3(out))
        out = self.flatten(out)

        out = torch.cat((out, torch.zeros_like(out)), dim=1)
        tree = []
        
        hidden_state_1, cell_state_1 = self.lstm1(out, (hidden_state_1, cell_state_1))
        hidden_state_2, cell_state_2 = self.lstm2(hidden_state_1, (hidden_state_2, cell_state_2))
        node_logits = self.class_head(hidden_state_2)
        tree = self.update_tree(tree, node_logits)

        while not self.stop_criterion(tree):
            parent_sibling = self.get_parent_sibling(tree)
            hidden_state_1, cell_state_1 = self.lstm1(parent_sibling, (hidden_state_1, cell_state_1))
            hidden_state_2, cell_state_2 = self.lstm2(hidden_state_1, (hidden_state_2, cell_state_2))
            node_logits = self.class_head(hidden_state_2)
            tree = self.update_tree(tree, node_logits)
        return tree


if __name__ == '__main__':
    test = torch.ones(1,1,20)
    model = SyntaxTreeLSTM(20)
    print(model(test).shape) # 1,128,20
