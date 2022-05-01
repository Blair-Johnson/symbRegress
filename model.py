import torch
import torch.nn as nn
import torch.nn.functional as nf
from tree import *

from torch.distributions.categorical import Categorical
#from tree import SyntaxNode, tree_from_preorder

N_SAMPLES = 20

class SyntaxTreeLSTM(nn.Module):
    ''' Define the set of possible operations'''
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

    def __init__(self, n_samples:int, n_hidden: int):
        super(SyntaxTreeLSTM, self).__init__()
        ''' n_samples: the length of the input array of data samples
            n_hidden: number of hidden nodes in LSTM'''
        self.n_samples = n_samples
        self.n_hidden = n_hidden
        
        # define feature extractor
        self.conv1 = nn.Conv1d(2, 32, 7, padding=(0))
        self.conv2 = nn.Conv1d(32, 64, 7, padding=(0))
        self.conv3 = nn.Conv1d(64, 128, 7, padding=(0))
        self.flatten = nn.Flatten()
        
        # embedding layer to project feature extractor into embedding dimension
        self.embed = nn.Linear(self.__test_forward(), self.n_hidden)
        # lstm cell for model execution
        self.lstm = nn.LSTMCell(len(SyntaxTreeLSTM.OP_NAMES), self.n_hidden)
        # classification head to project hidden states to policy distributions
        self.class_head = nn.Linear(self.n_hidden, len(SyntaxTreeLSTM.OP_NAMES))
    
    def __test_forward(self) -> int:
        ''' Get the output dimension of the convolutional feature extractor'''
        with torch.no_grad():
            test = torch.ones((1, 2, self.n_samples))
            out = self.conv1(test)
            out = self.conv2(out)
            out = self.conv3(out)
            out = self.flatten(out)
            return out.shape[-1]
    
    # def stop_criterion(self, tree: SyntaxNode) -> bool:
    #     '''Test to see if tree is complete'''
    #     leaf_list = tree.get_leaf_nodes()
    #     for leaf in leaf_list:
    #         if leaf.n_args != 0:
    #             return False
    #     return True


    # def update_tree(self, tree: SyntaxNode, model_dist: torch.tensor) -> Tuple[SyntaxTree, 
    #                                                                            torch.tensor,
    #                                                                            torch.tensor]:
    #     '''Sample new token from distribution, ignore illegal choices, update tree'''
    #     # make model_dist probabilities
    #     model_dist = nf.softmax(model_dist, axis=0)
    #     # get last node in tree
    #     last_node = tree.get_last()
    #     # get list of node names thay may not follow the last node
    #     illegal_nodes = last_node.illegal[last_node.value]
    #     # convert node names to node indices
    #     illegal_nodes = [OP_NAMES.find(name) for name in illegal_nodes]
    #     # zero out the logit probabilities for illegal next nodes
    #     for index in illegal_nodes:
    #         model_dist[index] = 0
    #     # get the tree preorder
    #     preorder = tree.get_preorder()
    #     # sample a new node from the logit distribution and add to preorder
    #     dist = Categorical(model_dist)
    #     action = dist.sample()
    #     log_prob = dist.log_prob(action)
    #     preorder.append(OP_NAMES[action.item()])
    #     # reconstruct tree from the updated preorder
    #     tree = tree_from_preorder(preorder)
    #     return tree, action, log_prob



    # def get_parent_sibling(self, tree: SyntaxNode) -> torch.tensor:
    #     '''take tree and return the parent and sibling of next node in traversal'''
        

    def forward(self, 
                cell_state_0: torch.tensor,
                node_0: torch.tensor,
                x = None, 
                hidden_state_0 = None) -> Tuple[torch.tensor,
                                                torch.tensor,
                                                torch.tensor]:
        '''
        Args:
            cell_state_0: initial cell state for model
            node_0: initial node embedding for model
            x: optional initial input data, mutually exclusive with hidden_state_0
            hidden_state_0: initial hidden state for model, optional and mutually exclusive with x
        Returns:
            node_logits: logits for generating PI distribution over nodes
            hidden_state_1: new hidden state
            cell_state_1: new cell state

        '''
        
        # ensure that the user provides either an input array or a hidden cell state
        assert (type(x) != type(None)) or (type(hidden_state_0) != type(None))
        
        if type(x) != type(None):
            # apply 1d convolution to input data to create hidden state
            hidden_state_0 = nf.elu(self.conv1(x))
            hidden_state_0 = nf.elu(self.conv2(hidden_state_0))
            hidden_state_0 = nf.elu(self.conv3(hidden_state_0))
            hidden_state_0 = self.flatten(hidden_state_0)
            hidden_state_0 = self.embed(hidden_state_0)
        
        # input node, hidden, cell states into LSTM
        hidden_state_1, cell_state_1 = self.lstm(node_0, (hidden_state_0, cell_state_0))
        # get node logits from hidden state
        node_logits = self.class_head(hidden_state_1)

        return node_logits, hidden_state_1, cell_state_1


if __name__ == '__main__':
    x = torch.ones(1,2,20)
    node = torch.ones(1,12)
    cs = torch.zeros(1,256)
    model = SyntaxTreeLSTM(20, 256)
    print(model(cs, node, x=x))
