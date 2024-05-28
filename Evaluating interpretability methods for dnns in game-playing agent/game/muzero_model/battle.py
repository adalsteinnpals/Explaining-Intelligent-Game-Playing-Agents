#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 17:20:54 2021

@author: ap
"""

import numpy as np
import copy
from tqdm import tqdm
from mcts import Tree
from breakthrough import State
import time


def expert_action(state):
    legal_actions = state.legal_actions()
    feature = state.feature()
    
    num_player = np.sum(feature[0] == 1)
    num_opponent = np.sum(feature[1] == 1)
    
    capture_actions = []
    found_terminal = False
    for action_ in legal_actions:
        state_ = copy.deepcopy(state)
        state_.play(action_)
        
        if state_.terminal():  
            return action_
                            
        if num_opponent > np.sum(state_.feature()[0] == 1):            
            capture_actions.append(action_)
    
    if len(capture_actions) > 0:
        return np.random.choice(capture_actions)
    
    return np.random.choice(legal_actions)


def vs_expert(nets, n=100):
    
    
    results = {}
    results_by_color = {}
    for i in range(n):
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
                policy = tree.think(state, num_simulations = 30, temperature = 0.5, show = False)

                #pdb.set_trace()

                action = sorted([(a, policy[a]) for a in state.legal_actions()], key=lambda x:-x[1])[0][0]
            else:
                action = expert_action(state)
                    
                                    
            state = state.play(action)
            turn_alternator = not turn_alternator
            
        t1 = time.time()
        r = state.terminal_reward() if first_turn_white else -state.terminal_reward()
        results[r] = results.get(r, 0) + 1
        if r == 1:
            results_by_color[turn] = results_by_color.get(turn, 0) + 1
        
    
    for r in [0,-1,1]:
        if r not in results:
            results[r] = 0
    return results



#  Battle against random agents

def vs_random(nets, n=100):
    

    results = {}
    results_by_color = {}
    for i in range(n):
        first_turn_white = i % 2 == 0
        turn = (i % 2) + 1
        turn_alternator = first_turn_white 
        state = State()
        t0 = time.time()
        while not state.terminal():
            if turn_alternator:
                #tree = Tree(nets)   
                #policy = tree.think(state, num_simulations = 1, temperature = 1, show = False)
                me = state.player_to_move

                policy, _ = nets.prediction.inference(state.feature())
                
                #tree = Tree(nets)   
                #policy = tree.think(state, num_simulations = 30, temperature = 1, show = False)

                #pdb.set_trace()

                action = sorted([(a, policy[a]) for a in state.legal_actions()], key=lambda x:-x[1])[0][0]
            else:
                action = np.random.choice(state.legal_actions())

            state.play(action)
            turn_alternator = not turn_alternator
            
        t1 = time.time()
        r = state.terminal_reward() if first_turn_white else -state.terminal_reward()
        results[r] = results.get(r, 0) + 1
        if r == 1:
            results_by_color[turn] = results_by_color.get(turn, 0) + 1
        
    
    for r in [0,-1,1]:
        if r not in results:
            results[r] = 0
    return results

