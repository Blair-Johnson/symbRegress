import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

from itertools import count
from torch.distributions.categorical import Categorical
from tree import *
from model import *
from genDataset import *

N_HIDDEN = 256
N_SAMPLES = 20

parser = argparse.ArgumentParser()
# number of episodes to run
parser.add_argument('--num_episodes', type=int, default=1000)
# maximum search depth
parser.add_argument('--max_depth', type=int, default=12)
# risk threshold for policy gradient calculation
# parser.add_argument('--r_epsilon', type=float, default=9.0) 

def main(args):
    
    state_history = []
    action_history = []
    reward_history = []
    
    for episode in range(args.num_episodes):
        # execute many episodes of exploration and policy updates
        
        # 1. get initial batch of data from dataloader
        # 2. initialize empty cell state
        cell_state = torch.zeros((1, N_HIDDEN))
        # 3. create start node embedding
        
        # 4. initialize the syntax tree (environment)
        
        # exploration step (construct and fit tree, populate reward history)
        with torch.no_grad():
            for step in count():
                # loop until tree is complete or passes max depth

                # step LSTM and get new node logits
                if step != 0:
                    node_logits, hidden_state_1, cell_state_1 = policy_model(cell_state_0,
                                                                             node_embed_0,
                                                                             hidden_state_0)
                else:
                    node_logits, hidden_state_1, cell_state_1 = policy_model(cell_state_0,
                                                                             node_embed_0,
                                                                             x)
                # create discrete policy distribution (TODO: Double check this is the correct dim)
                pi_dist = F.softmax(node_logits, dim=-1)
                # TODO: Zero elements of the distribution that correspond to illegal nodes
                #       - query tree for parent and sibling nodes
                #       - query tree for illegal nodes based on parent
                #       - set corresponding elements of pi_dist to zero

                pi_dist_obj = Categorical(pi_dist)
                # sample action (node) from policy distribution
                node = pi_dist_obj.sample()

                # get next state, reward, done from environment based on action
                # TODO:
                # - append new syntax node to syntax tree (containing hidden state)

                # NOTE:
                # during exploration, this hidden state can be detached from computation graph.
                # in the policy update, the tree will need to be repopulated with 'live'
                # hidden states that remain in the compute graph so that backprop can effect
                # previous steps of the LSTM.
                tree.add_node(node, hidden_state_1.detach())
                # - check if tree complete
                if complete:
                    # fit model...

                    # calculate reward based on inverse of L2 norm between f(x) and y
                    reward = 1 / (1 + torch.linalg.norm(f(x[0,0]) - x[0,1]))
                    pass
                else:
                    reward = 0
                # - get next parent node type, parent node hidden state
                # - get next sibling node type
                # - if tree complete, generate function and fit constants to data
                #   - then evaluate reward function based on L2 norm

                # update state history with policy model inputs
                if step !=0:
                    state_history.append([cell_state_0,
                                          node_embed_0,
                                          hidden_state_0])
                else:
                    state_history.append([cell_state_0,
                                          node_embed_0,
                                          x])
                # update action history with sampled node from policy distribution
                action_history.append(node)
                # update reward history with calculated reward if tree is complete
                reward_history.append(reward)

                # step states for next iteration
                # TODO: break these off from computation graph
                cell_state_0 = cell_state_1
                node_embed_0 = tree.get_sibling()
                hidden_state_0 = tree.get_parent_state()
            
            
            
            
if __name__ == '__main__':
    args = parser.parse_args()
    
    