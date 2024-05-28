#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 16:52:35 2021

@author: ap
"""

from dataclasses import dataclass

@dataclass
class ConfigClass:
    
    # TRAINING
    batch_size: int = 512
    num_epochs: int = 5
    train_on_cuda: bool = True
    max_batches_per_epoch: int = 20000
    num_games: int = 1000000
    self_play_batch_size: int = 200
    model_name: str = 'alpha_v05'
    
    learning_rate: float = 1e-3
    momentum: float = 0.75
    weight_decay: float = 1e-4
    learning_decay_rate: float = 0.9
        
    
    # SELFPLAY
    play_on_cuda: bool = True
    num_simulations: int = 25
    num_cpu_workers: int = 5
    num_gpu_workers: int = 2
    
    
    # MODEL
    num_filters: int = 64
    num_blocks: int = 5
    architecture: str = 'alphazero'
    
    
    # BOARD SIZE
    rows: int = 8
    columns: int = 6
    
    
    

    #%%
if __name__ == '__main__':
    
    c = ConfigClass