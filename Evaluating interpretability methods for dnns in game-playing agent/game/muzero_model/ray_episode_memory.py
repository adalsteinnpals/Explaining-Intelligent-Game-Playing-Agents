#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 16:52:27 2021

@author: adalsteinnpalsson
"""
import ray
import numpy as np




@ray.remote(num_cpus = 1)
class EpisodeMemory(object):
    def __init__(self,
                episodes = []):
        self.episodes = episodes
        self.num_played_games = len(episodes)
        print('Episode Memory Initialized')

    def add_one(self, episode):
        self.episodes.append(episode)
        self.num_played_games += 1
        #if self.num_played_games % 100 == 0:
        #    print(self.num_played_games)
        
    def add_many(self, many_episodes):
        self.episodes = self.episodes + many_episodes
        self.num_played_games += len(many_episodes)

    def get_all(self):
        return self.episodes
    
    def get_all_from_idx(self, idx):
        return self.episodes[idx:]
    
    def how_many(self):
        return self.num_played_games
    
    def get_batch(self, N):
        n_episodes = self.num_played_games
        return [self.episodes[np.random.randint(n_episodes)] for j in range(N)]


#%%