#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 16:40:14 2021

@author: ap
"""

import ray
import time
import numpy as np
import configparser
from muzero_model.ray_selfplay import *
from muzero_model.models import Nets, Alphazero_wrapper
from muzero_model.utils import load_saved_model, save_model, load_episodes_only, load_saved_model_only, dict_to_cpu, save_model_from_dict, import_all_episodes

            
if __name__ == '__main__':
    ray.init(num_cpus=7, num_gpus=1)
    
    
    num_cpu_workers = 5
    num_gpu_workers = 2
    
    
    num_filters = 64
    num_blocks = 5
    architecture = 'alphazero'
    
    
    

    episodes = []
    nets = Alphazero_wrapper(num_blocks=num_blocks, num_filters=num_filters).float()
    
    
    # Load old model or episodes
    # model_PATH = 'model_checkpoints/alpha_v04/models/alpha_v04_model_0000523471_ep_20210520_031402.pkl'
    # nets = load_saved_model_only(model_PATH, nets)
    # episodes = import_all_episodes(config, import_from_model = 'alpha_v04')
    
    
        
    print('Using net with:')
    print('Number of filtes: ',nets.num_filters)
    print('Number of blocks: ',nets.num_blocks)
    print('')
    print('Number of episodes: ',len(episodes))
    print('')
    
    
    checkpoint = {
            "weights": None,
            "num_played_games": 0,
            "terminate": False,
            "training_step": 0,
            "lr":0,
            "value_loss":0,
            "policy_loss":0,
            "num_played_games":0,
            "wins_vs_random_100":0,
            "wins_vs_expert_100":0,
        }
    
    checkpoint['weights'] = dict_to_cpu(nets.state_dict())
    
    
    print('starting shared storage...')
    shared_storage = SharedStorage.remote(checkpoint, config)
    print('starting episode memory...')
    episode_memory = EpisodeMemory.remote()
    
    print('starting trainer...')
    trainer = ray_trainer.remote(nets, config)
    trainer.continuous_training.remote(episode_memory, shared_storage)
    
    
    print('starting self play workers...')
    Selfplay_list = [Selfplay.remote(nets, config) for _ in range(num_gpu_workers)]
    selfplay_gpu_workers = [sf.continuous_self_play.remote(episode_memory, shared_storage) for sf in Selfplay_list]
    
    SelfplayCpu_list = [SelfplayCpu.remote(nets, config) for _ in range(num_cpu_workers)]
    selfplay_cpu_workers = [sf.continuous_self_play.remote(episode_memory, shared_storage) for sf in SelfplayCpu_list]
    
    print('starting logger...')
    logger = ray_logger.remote(nets, config)
    logger.continuous_logging.remote(episode_memory, 
                                     shared_storage)
    
    #print('starting saver...')
    #saver = ray_logger.remote(nets, config)
    #saver.continuous_saving.remote(episode_memory, 
    #                                 shared_storage)
    
    print('starting battler...')
    battler = ray_battler.remote(nets,config)
    battler.continuous_battle.remote(episode_memory, 
                                     shared_storage)
    
    
    chunk_size = len(episodes)//100
    idx = 0
    while len(episodes) > 0:
        print('Adding chunk: ',idx)
        episode_memory.add_many.remote(episodes[:chunk_size])
        episodes = episodes[chunk_size:]
        idx += 1
        
        
    
    
    
    last_save_model = ray.get(episode_memory.how_many.remote())
    print('Num episodes: ',last_save_model)
    while True:
        time.sleep(100)
        
        num_games = ray.get(episode_memory.how_many.remote())

        if num_games - last_save_model > 5000:
            

            last_save_model = num_games
            
            new_episodes = ray.get(episode_memory.get_all_from_idx.remote(len(episodes)))
            
            episodes = episodes + new_episodes
            
            model_weights = ray.get(shared_storage.get_info.remote("weights"))
            
            save_model_from_dict(model_weights, episodes, config)
            