#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from muzero_model.breakthrough import State
from muzero_model.mcts import Tree
import numpy as np



def selfplay(nets, g, config, play_on_cuda = None):
    
    if play_on_cuda is None:
        play_on_cuda = config['SELFPLAY'].getboolean('play_on_cuda')
    num_simulations = config['SELFPLAY'].getint('num_simulations')
    
    record, p_targets, features, action_features = [], [], [], []
    state = State()
    
    temperature = get_temperature(g)
    
    if play_on_cuda: 
        nets.cuda()
    else:
        nets.cpu()
        
    while not state.terminal():
        tree = Tree(nets)   


        p_target = tree.think(state, num_simulations, temperature)
        
        p_target *= state.available_moves().reshape(-1)
        p_target /= np.sum(p_target)

        # print(p_target)
        p_targets.append(p_target)
        features.append(state.feature())
        # Select action with generated distribution, and then make a transition by that action
        
        action = np.random.choice(np.arange(len(p_target)), p=p_target)
        action_features.append(state.action_feature(action))
        # print(action)
        state.play(action)
        record.append(action)


    # reward seen from the first turn player
    reward = state.terminal_reward() 
    
    episode = (record, reward, features, action_features, p_targets)
    
    #print('Reward: ',reward)
    #print('Player to move: ',state.player_to_move)
    #print('Record: ', record)
    #print(state)
    
    #q.put(episode)
    return episode



def get_temperature(epoch):

    temperature_thresholds = [(0, 1), (5000, 1), (10000, 1)] 
    
    if epoch >= temperature_thresholds[0][0]:
        if epoch >= temperature_thresholds[1][0]:
            if epoch >= temperature_thresholds[2][0]:
                temperature = temperature_thresholds[2][1]
            else:
                temperature = temperature_thresholds[1][1]
        else:
            temperature = temperature_thresholds[0][1]
    else:
        temperature = 1
    return temperature
