#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 14:44:12 2021

@author: adalsteinnpalsson
"""

from muzero_nets2 import Nets

from os.path import join
from utils import load_saved_model, load_saved_model_only
import time
from os import listdir
import numpy as np
#%%
base_path = "/home/adalsteinn/Documents/random/muzero/model_checkpoints"
files = [file  for file in listdir(base_path) if len(file.split('_')) == 6]

#%%
from tqdm import tqdm




file_path = 'torch_model_checkpoint_20210303.pkl'
nets = Nets().float()
nets, episodes = load_saved_model(join(base_path, file_path), nets)
    


result_dict = {}
for file in tqdm(files):
    model2_path = file #'torch_model_only_checkpoint_20210303_093907.pkl'
        
    
    
    nets2 = Nets().float()
    nets2  = load_saved_model_only(join(base_path, model2_path), nets2)
    
    
    
    from breakthrough import State, Board
    
    def model_vs_model(nets, nets2, n=100):
        
        results = {'net1': 0, 'net2': 0}
        
        results_by_color = {}
        for i in range(n):
            # print('muuuu',i)
            first_turn_white = i % 2 == 0
            turn = (i % 2) + 1
            turn_alternator = first_turn_white 
            state = State()
            t0 = time.time()
            while not state.terminal():
                if turn_alternator:
                    policy, _ = nets.predict_all(state, [])[-1]
                    me = state.player_to_move
                    
                    #tree = Tree(nets)   
                    #policy = tree.think(state, num_simulations = 30, temperature = 1, show = False)
    
                    # action = sorted([(a, policy[a]) for a in state.legal_actions()], key=lambda x:-x[1])[0][0]
                    action = np.random.choice(state.legal_actions(), p=policy[state.legal_actions()]/np.sum(policy[state.legal_actions()]))
    
                else:
                    policy, _ = nets2.predict_all(state, [])[-1]
                    me = state.player_to_move
                    # action = sorted([(a, policy[a]) for a in state.legal_actions()], key=lambda x:-x[1])[0][0]
                    action = np.random.choice(state.legal_actions(), p=policy[state.legal_actions()]/np.sum(policy[state.legal_actions()]))
                    
                state.play(action)
                turn_alternator = not turn_alternator
            t1 = time.time()
            white_net = ('net1' if first_turn_white else 'net2')
            
            # print('White is: ',('net1' if first_turn_white else 'net2'))
            # print('Winner: ', ('White' if state.terminal_reward() == 1 else 'Black'))
            
            
            net1_reward = state.terminal_reward() if first_turn_white else -state.terminal_reward()
            winner = ('net1' if net1_reward == 1 else 'net2')
                
            results[winner] = results.get(winner, 0) + 1
            
            
        return results
    
    res = model_vs_model(nets, nets2, n=100)
    
    result_dict[file] = res
    
    

#%%


import pandas as pd

df = pd.DataFrame.from_dict(result_dict, orient = 'index')


#%%

from utils import load_saved_model_only
from muzero_nets2 import Nets
import numpy as np
from mcts import Tree

from os.path import join
import time

from tqdm import tqdm
from breakthrough import State, Board

base_path = "/home/adalsteinn/Documents/random/muzero/game/muzero_model/model_checkpoints"
model_name = 'torch_model_only_checkpoint_20210315_101309.pkl'

#%%

nets = Nets().float()
nets  = load_saved_model_only(join(base_path, model_name), nets)


res = model_vs_model(nets, nets)
print(res)


#%%
def model_vs_model(nets, nets2, n=100):
    
    results = {'net1': 0, 'net2': 0}
    
    results_by_color = {}
    for i in tqdm(range(n)):
        # print('muuuu',i)
        first_turn_white = i % 2 == 0
        turn = (i % 2) + 1
        turn_alternator = first_turn_white 
        state = State()
        t0 = time.time()
        while not state.terminal():
            if turn_alternator:
                #policy, _ = nets.predict_all(state, [])[-1]
                me = state.player_to_move
                
                tree = Tree(nets)   
                policy = tree.think(state, num_simulations = 30, temperature = 1, show = False)

                # action = sorted([(a, policy[a]) for a in state.legal_actions()], key=lambda x:-x[1])[0][0]
                action = np.random.choice(state.legal_actions(), p=policy[state.legal_actions()]/np.sum(policy[state.legal_actions()]))

            else:
                policy, _ = nets2.predict_all(state, [])[-1]
                me = state.player_to_move
                # action = sorted([(a, policy[a]) for a in state.legal_actions()], key=lambda x:-x[1])[0][0]
                action = np.random.choice(state.legal_actions(), p=policy[state.legal_actions()]/np.sum(policy[state.legal_actions()]))
                
            state.play(action)
            turn_alternator = not turn_alternator
        t1 = time.time()
        white_net = ('net1' if first_turn_white else 'net2')
        
        # print('White is: ',('net1' if first_turn_white else 'net2'))
        # print('Winner: ', ('White' if state.terminal_reward() == 1 else 'Black'))
        
        
        net1_reward = state.terminal_reward() if first_turn_white else -state.terminal_reward()
        winner = ('net1' if net1_reward == 1 else 'net2')
            
        results[winner] = results.get(winner, 0) + 1
        print(results)
        
        
    return results







