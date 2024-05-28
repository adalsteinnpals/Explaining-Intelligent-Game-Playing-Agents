# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 09:16:19 2019

@author: adals
"""

from datetime import datetime
import yaml
import numpy as np

from torch.utils.tensorboard import SummaryWriter
import time
import configparser



# IMPORT MODULES
from models import Nets, Alphazero_wrapper
from battle import vs_expert, vs_random
from trainer import train
from self_play import selfplay
from utils import load_saved_model, save_model, load_episodes_only




from torch.multiprocessing import Pool, set_start_method
from tqdm import tqdm

try:
     set_start_method('spawn')
except RuntimeError:
    pass

#%%
def main():
    # Main algorithm of MuZero
    
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    writer = SummaryWriter()    
    
    num_filters = config['MODEL'].getint('num_filters')
    num_blocks = config['MODEL'].getint('num_blocks')
    architecture = config['MODEL']['architecture']

    episodes = []
    if architecture == 'muzero':
        print('Using MuZero')
        nets = Nets(num_blocks=num_blocks, num_filters=num_filters).float()
    elif architecture == 'alphazero':
        print('Using AlphaZero')
        nets = Alphazero_wrapper(num_blocks=num_blocks, num_filters=num_filters).float()
    else:
        raise NotImplementedError 
    
    
    load_latest = config['TRAINING'].getboolean('load_latest')
    load_episodes_only_flag = config['TRAINING'].getboolean('load_episodes_only')
    
    if load_episodes_only_flag:
        PATH = config['TRAINING']['episodes_path']
        print('Loading only episodes from file: ',PATH)
        episodes = load_episodes_only(PATH)
        print('Done loading...')
    elif load_latest:
        print('Loading latest model...')
        model_name = config['TRAINING']['model_name']
        PATH = 'model_checkpoints/'+model_name+'_model_and_episodes.pkl'
        nets, episodes = load_saved_model(PATH, nets)
        print('Done loading...')
        
    print('Using net with:')
    print('Number of filtes: ',nets.num_filters)
    print('Number of blocks: ',nets.num_blocks)
    print('')
    print('Number of episodes: ',len(episodes))
    print('')
   
    vs_random_sum = vs_random(nets)
    print('vs_random = ', sorted(vs_random_sum.items()))
    writer.add_scalar('wins_vs_random_100', vs_random_sum[1], 0)
    
    vs_expert_sum = vs_expert(nets)
    print('vs_expert = ', sorted(vs_expert_sum.items()))
    writer.add_scalar('wins_vs_expert_100', vs_expert_sum[1], 0)
    
    
    
    result_distribution = {1:0, 0:0, -1:0}

    start_epoch = len(episodes)

    t0 = time.time()
    
    num_per_step = config['TRAINING'].getint('self_play_batch_size')
    num_games    = config['TRAINING'].getint('num_games')
    
    for g in range(num_per_step + start_epoch, 
                   start_epoch + num_games, 
                   num_per_step):

        t1 = time.time()


        # Generate one 1 episode
        print('Game: ', len(episodes))
        
        nets.share_memory()      
        
        
        with Pool(processes=3) as pool:
            episodes_ = pool.starmap(selfplay, tqdm([(nets, g, config) for i in range(num_per_step)], total = num_per_step))
            
                    
        #episode = selfplay(nets, num_simulations, g, temperature_thresholds)
        #episodes.append(episode)
        
        
        for episode in episodes_:
            result_distribution[episode[1]] += 1
            episodes.append(episode)
            
        
        print('Elapsed since start: ',(time.time()-t0))
        print('Last game: ',(time.time()-t1))
        
        
        # Training of neural nets
        print('generated = ', sorted(result_distribution.items()))
        
        print('TRAINING...')
        nets = train(episodes, nets, g, writer, config)
        print('DONE TRAINING...')
        
        
        
        
        
        vs_random_dict = vs_random(nets)
        writer.add_scalar('wins_vs_random_100', vs_random_dict[1], g)

        print('vs_random = ', sorted(vs_random_dict.items()))
        
        
        vs_expert_dict = vs_expert(nets)
        writer.add_scalar('wins_vs_expert_100', vs_expert_dict[1], g)

        print('vs_expert = ', sorted(vs_expert_dict.items()))
        
        
        for r, n in vs_random_dict.items():
            vs_random_sum[r] += n
            
        print('Total sum = ', sorted(vs_random_sum.items()))
            

        save_model(nets,episodes, config, verbose = 1)

    print('finished')


if __name__ == '__main__':
    main()

    
