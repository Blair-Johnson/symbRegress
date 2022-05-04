import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import random

from itertools import count
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from tree import *
from model import *
from genDatasets import *

if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

N_HIDDEN = 256
N_SAMPLES = 20
FUNC_LR = .1
POLICY_LR = 1e-3
FUNC_OPTIM_STEPS = 1000
BELLMAN_GAMMA = .95

parser = argparse.ArgumentParser()
# number of episodes to run
parser.add_argument('--num_episodes', type=int, default=1000)
# maximum search depth
parser.add_argument('--max_depth', type=int, default=10)
parser.add_argument('--min_depth', type=int, default=3)
# risk threshold for policy gradient calculation
# parser.add_argument('--r_epsilon', type=float, default=9.0)
# log directory
parser.add_argument('--logdir', type=str, default='./logs')

def main(args):
    
    policy_model = SyntaxTreeLSTM(N_SAMPLES, N_HIDDEN).to(DEVICE)
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=POLICY_LR)
    writer = SummaryWriter(args.logdir)
    print('Init model...')
    
    for episode in range(args.num_episodes):
        # execute many episodes of exploration and policy updates
        print(f'Episode: {episode}')
        state_history = []
        action_history = []
        reward_history = []
        exceeds_max_depth = False
        
        # 1. get initial batch of data from dataloader
        func_index = random.randint(1,8)
        # MANUAL NGUYEN 1 FOR TESTING
        func_index = 1
        x_train, x_test, z_train, z_test = gen_datasets(f'Nguyen-{func_index}', 2022, 6254)
        x_train = torch.tensor(np.stack([x_train, z_train])).view(1, 2, -1).to(DEVICE).float()
        x_test = torch.tensor(np.stack([x_test, z_test])).view(1, 2, -1).to(DEVICE).float()
        
        # 2. initialize empty cell state
        cell_state_0 = torch.zeros((1, N_HIDDEN)).to(DEVICE).float()
        # 3. create start node embedding
        node_embed_0 = F.one_hot(torch.tensor(list(SyntaxNode.op_list.keys()).index('start')), len(SyntaxNode.op_list.keys()))
        node_embed_0 = node_embed_0.view(1, -1).to(DEVICE).float()
        # 4. initialize the syntax tree (environment)
        tree = SyntaxNode('start')
        
        # exploration step (construct and fit tree, populate reward history)
        for step in count():
            exit = False
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
            illegal_nodes = [list(SyntaxNode.op_list.keys()).index(node) for node in parent_node.illegal]
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
            tree.append(list(SyntaxNode.op_list.keys())[node], hidden_state_1.detach())
            # NOTE:
            # during exploration, this hidden state can be detached from computation graph.
            # in the policy update, the tree will need to be repopulated with 'live'
            # hidden states that remain in the compute graph so that backprop can effect
            # previous steps of the LSTM. Policy distributions will change from the exploration phase as LSTM trains.

            # - check if tree complete
            if tree_complete(tree):
                print('Tree complete!')
                print(f'Expression: {get_expression(tree)}')

                # check that function is parameterized
                if len(SyntaxNode.parameters[tree.tree_idx]) > 0:
                    if 'var' in tree.get_preorder():
                        # fit function...
                        # TODO: Figure out if it's even worth it to do this on the gpu, or keep it on the cpu
                        #SyntaxNode.parameters[tree.tree_idx] = [param.to(DEVICE) for param in SyntaxNode.parameters[tree.tree_idx]]
                        if get_tree_depth(tree) < args.min_depth:
                            reward = torch.tensor(-1)
                            exit = True
                        else:
                            func_optim = torch.optim.Adam(SyntaxNode.parameters[tree.tree_idx], lr = FUNC_LR)
                            for i in range(FUNC_OPTIM_STEPS):
                                func_optim.zero_grad()
                                func_loss = torch.mean(torch.pow(tree.get_function(x_train[0,0].cpu()) - x_train[0,1].cpu(), 2))
                                func_loss.backward()
                                func_optim.step()
                                if (i % 100) == 0:
                                    print(f'Fitting func: {func_loss.item()}')
                            # calculate reward based on inverse of L2 norm between f(x) and y
                            reward = 1 / (1 + torch.mean(torch.pow(tree.get_function(x_test[0,0].cpu()) - x_test[0,1].cpu(), 2)))
                            print(f'reward: {reward}')
                            exit = True
                    else:
                        print('Function is constant')
                        exit = True
                        # haven't exceeded max depth, but we'll use this to skip to the next episode
                        reward = torch.tensor(-1)
                        #exceeds_max_depth = True
                else:
                    print('Function not parameterized')
                    #reward = 1 / (1 + torch.mean(torch.pow(tree.get_function(x_test[0,0].cpu()) - x_test[0,1].cpu(), 2)))
                    #print(f'reward: {reward}')
                    reward = torch.tensor(-1)
                    exit = True
                    #exceeds_max_depth = True
            elif get_tree_depth(tree) >= args.max_depth:
                print('Max depth exceeded.')
                reward = torch.tensor(-1)
                #exceeds_max_depth = True
                exit = True
            else:
                reward = torch.tensor(-1)
            # TODO:
            # - get next parent node type, parent node hidden state
            # - get next sibling node type
            # - if tree complete, generate function and fit constants to data
            #   - then evaluate reward function based on L2 norm

            # update state history with policy model inputs
            # TODO: Don't need all of these, only the first
            if step !=0:
                state_history.append([cell_state_0,
                                      node_embed_0,
                                      hidden_state_0])
            else:
                state_history.append([cell_state_0,
                                      node_embed_0,
                                      x_train])
            # update action history with sampled node from policy distribution
            action_history.append(node)
            # update reward history with calculated reward if tree is complete
            reward_history.append(reward.item())
            
            # breakout if episode trajectory complete
            if exit:
                break
            
            # step states for next iteration
            # TODO: break these off from computation graph
            cell_state_0 = cell_state_1
            node_embed_0 = tree.get_sibling_token().to(DEVICE).float()
            hidden_state_0 = tree.get_parent_state()
                
        if exceeds_max_depth:
            # skip to next episode if max depth exceeded
            continue
        # update policy model (not using risk-aware policy at the moment)

        print('Training phase...')
        # calculate Bellman discounted rewards
        if len(reward_history) > 1:
            bellman_rewards = []
            #reward_sum = reward_history.pop(-1)
            #bellman_rewards.append(reward_sum)
            #for reward in reward_history:
            #    reward_sum += reward + BELLMAN_GAMMA * reward_sum
            #    bellman_rewards.append(reward_sum)
            for i in range(len(reward_history)):
                bellman_rewards.append(reward_history[-i]*BELLMAN_GAMMA**i)
            bellman_rewards.reverse()
            bellman_rewards = np.array(bellman_rewards)
            if np.isnan(np.sum(bellman_rewards)):
                print('NaN reward, skipping episode')
                continue

            # normalize rewards
            mu = np.mean(bellman_rewards)
            sigma = np.std(bellman_rewards)
            bellman_rewards = (bellman_rewards - mu) / sigma

            # replace reward history with normalized bellman rewards
            reward_history = list(bellman_rewards)

            # log episode mean reward
            writer.add_scalar('reward/train', float(mu), episode)
        else:
            writer.add_scalar('reward/train', reward_history[-1], episode)

        # zero model optimizer gradient for training
        optimizer.zero_grad()

        tree.__del__()
        tree = SyntaxNode('start')
        for traj_step in range(len(state_history)):
            #tree.__del__()
            # init a new tree, will step through cumulative tree iterations
            # this is necessary because each optimizer step will change the policy model,
            # thus tree must be reconstructed with the new policy model each time
            #tree = SyntaxNode('start')

            # read (s,a,r) from history for the end state of this partial trajectory
            state = state_history[traj_step]
            action = action_history[traj_step]
            reward = reward_history[traj_step]

            # get sibling type and parent hidden state from tree
            # these should be deterministic from input data and history of sampled nodes, so
            # no need to track these in state TODO: fix this / simplify to remove unnecessary state info

            sibling_token = tree.get_sibling_token().to(DEVICE)
            parent_hidden = tree.get_parent_state()

            if traj_step != 0:
                node_logits, hidden_state, cell_state = policy_model(cell_state,
                                                              sibling_token,
                                                              hidden_state_0 = parent_hidden)
            else:
                # initial function data and state vectors
                node_logits, hidden_state, cell_state = policy_model(state[0].to(DEVICE),
                                                                     state[1].to(DEVICE),
                                                                     state[2].to(DEVICE))

            tree.append(list(SyntaxNode.op_list.keys())[action], hidden_state)

            pi_dist = F.softmax(node_logits, dim=-1)
            pi_dist = Categorical(pi_dist)
            loss = -pi_dist.log_prob(action) * reward
            loss.backward(retain_graph = True)

        # step optimizer on accumulated gradient
        optimizer.step()
        print('optim step.')

            # # zero model optimizer gradient for training
            # optimizer.zero_grad()

            # build tree through inference to current state
    #                 for train_step in range(history_step + 1):
    #                     # ex: hist -> [exp, +, var, const]
    #                     # hist step 0 -> training steps: [exp]
    #                     # hist step 1 -> training steps: [exp, +]
    #                     # hist step 2 -> training steps: [exp, +, var]

    #                     sibling_token = tree.get_sibling_token()
    #                     parent_hidden = tree.get_parent_state()

    #                     if train_step != 0:
    #                         node_logits, hidden_state, cell_state = model(cell_state,
    #                                                                       sibling_token,
    #                                                                       hidden_state_0 = parent_hidden)
    #                     else:
    #                         # initial function data and state vectors
    #                         node_logits, hidden_state, cell_state = model(*state_history[0])

    #                     # retaining hidden state within torch graph, allows lstm to backprop through time
    #                     tree.append(SyntaxNode.op_list.keys()[action_history[train_step]], hidden_state)

    #                 pi_dist = F.softmax(node_logits, dim=-1)
    #                 pi_dist = Categorical(pi_dist)
    #                 # gradients propogate through likelihood function back to model and hidden state structure
    #                 # TODO: Modify to include baseline term
    #                 loss = -pi_dist.log_prob(action) * reward
    #                 loss.backward()
    #                 # possibly de-indent optim.step() if we want gradients to accumulate and then step after whole trajectory
    #                 # this would eliminate the need for the inner training loop
    #                 optimizer.step()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
    
