import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import random

from itertools import count
from torch.distributions.categorical import Categorical
from tree import *
from model import *
from genDataset import *

N_HIDDEN = 256
N_SAMPLES = 20
FUNC_LR = 1e-3
FUNC_OPTIM_STEPS = 100

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
        func_index = random.randint(1,9)
        x_train, x_test, z_train, z_test = gen_datasets(f'Nguyen-{func_index}', 2022, 6254)
        x_train = torch.tensor(np.stack([x_train, z_train])).view(1, 2, -1)
        x_test = torch.tensor(np.stack([x_test, z_test])).view(1, 2, -1)
        # 2. initialize empty cell state
        cell_state = torch.zeros((1, N_HIDDEN))
        # 3. create start node embedding
        node_embed_0 = F.one_hot(torch.tensor(SyntaxNode.op_list.keys().find('start')), len(SyntaxNode.op_list.keys()))
        node_embed_0 = node_embed_0.view(1, -1)
        # 4. initialize the syntax tree (environment)
        tree = SyntaxNode('start')
        
        # exploration step (construct and fit tree, populate reward history)
        with torch.no_grad():
            for step in count():
                # loop until tree is complete or passes max depth

                # step LSTM and get new node logits
                if step != 0:
                    node_logits, hidden_state_1, cell_state_1 = policy_model(cell_state_0,
                                                                             node_embed_0,
                                                                             hidden_state_0 = hidden_state_0)
                else:
                    node_logits, hidden_state_1, cell_state_1 = policy_model(cell_state_0,
                                                                             node_embed_0,
                                                                             x = x_train)
                # create discrete policy distribution (TODO: Double check this is the correct dim)
                pi_dist = F.softmax(node_logits, dim=-1)
                
                # TODO: Zero elements of the distribution that correspond to illegal nodes
                #       - query tree for parent and sibling nodes
                #       - query tree for illegal nodes based on parent
                #       - set corresponding elements of pi_dist to zero
                
                # query parent of next node
                parent_node = tree.get_last()
                # convert illegal node names into indices
                illegal_nodes = [SyntaxNode.op_list.keys().find(node) for node in parent_node.illegal]
                # zero probability of illegal nodes
                for index in illegal_nodes:
                    pi_dist[0, index] = 0
                # create categorical distribution from discrete distribution vector (for sampling)
                pi_dist_obj = Categorical(pi_dist)
                # sample action (node) from policy distribution
                node = pi_dist_obj.sample()

                # get next state, reward, done from environment based on action
                # TODO:
                # - append new syntax node to syntax tree (containing hidden state)
                # - implement append function and add embedding member to tree
                tree.append(SyntaxNode.op_list.keys()[node], hidden_state_1.detach())
                # NOTE:
                # during exploration, this hidden state can be detached from computation graph.
                # in the policy update, the tree will need to be repopulated with 'live'
                # hidden states that remain in the compute graph so that backprop can effect
                # previous steps of the LSTM. Policy distributions will change from the exploration phase as LSTM trains.
                
                # - check if tree complete
                if tree_complete(tree):
                    # fit function...
                    func_optim = torch.optim.Adam(SyntaxNode.parameters[tree.tree_idx], lr = FUNC_LR)
                    for step in range(FUNC_OPTIM_STEPS):
                        func_optim.zero_grad()
                        func_loss = torch.mean((tree.get_function(x_train[0,0]) - x_train[0,1])**2)
                        func_loss.backward()
                        func_optim.step()

                    # calculate reward based on inverse of L2 norm between f(x) and y
                    reward = 1 / (1 + torch.mean((f(x_test[0,0]) - x_test[0,1])**2))
                    
                else:
                    reward = 0
                # TODO:
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
                
        # update policy model (not using risk-aware policy at the moment)
        if (episode > 0):
            # init a new tree
            
            # calculate Bellman discounted rewards
            
            # normalize rewards
            
            optimizer.zero_grad()
            
            for i in range(step + 1):
                # read (s,a,r) from history
                state = state_history[i]
                action = action_history[i]
                reward = reward_history[i]
                
                # get sibling type and parent hidden state from tree
                # these should be deterministic from input data and history of sampled nodes, so
                # no need to track these in state TODO: fix this / simplify to remove unnecessary state info
                sibling_type = tree.get_sibling()
                parent_hidden = tree.get_parent_embed()
                
                if i != 0:
                    node_logits, hidden_state, cell_state = model(cell_state,
                                                                  sibling_embedding,
                                                                  hidden_state_0 = parent_hidden)
                else:
                    node_logits, hidden_state, cell_state = model(*state)
                
                pi_dist = F.softmax(node_logits, dim=-1)
                pi_dist = Categorical(pi_dist)
                # gradients propogate through likelihood function back to model and hidden state structure
                # TODO: Modify to include baseline term
                loss = -pi_dist.log_prob(action) * reward
                loss.backward()
            
            optimizer.step()
            
            state_history = []
            action_history = []
            reward_history = []
            
            
            
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
    
