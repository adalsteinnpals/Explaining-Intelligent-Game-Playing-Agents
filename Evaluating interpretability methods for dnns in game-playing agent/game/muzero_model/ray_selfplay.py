#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 17:00:50 2021

@author: adalsteinnpalsson
"""



import ray
import torch.optim as optim
import torch

from torch.utils.tensorboard import SummaryWriter

import time
import numpy as np
import configparser
from ray_episode_memory import EpisodeMemory
from muzero_model.utils import load_saved_model, save_model, load_episodes_only, load_saved_model_only, dict_to_cpu, save_model_from_dict, import_all_episodes
from muzero_model.models import Nets, Alphazero_wrapper
from muzero_model.ray_shared_storage import SharedStorage
from muzero_model.trainer import gen_target
from muzero_model.self_play import selfplay
from muzero_model.battle import vs_expert, vs_random
import copy

from datetime import datetime



@ray.remote(num_gpus = 0.24)
class Selfplay(object):
    def __init__(self,
                 model,
                 config):
        
        self.model = model
        self.config = config
        
    def continuous_self_play(self,
                             episode_memory, 
                             shared_storage):
        
        
        self.model.load_state_dict(ray.get(shared_storage.get_info.remote("weights")))
        
        
        while True:
            
            for i in range(200):
                g = ray.get(episode_memory.how_many.remote())
                episode = selfplay(self.model, g, self.config, play_on_cuda = True)
                
                t1 = time.time()
                episode_memory.add_one.remote(episode)
                print('Sending one episode in: ',time.time()-t1,' seconds...')
            
            self.model.load_state_dict(ray.get(shared_storage.get_info.remote("weights")))
            

@ray.remote(num_cpus = 1)
class SelfplayCpu(object):
    def __init__(self,
                 model,
                 config):
        self.model = model
        self.config = config
        
    def continuous_self_play(self,
                             episode_memory, 
                             shared_storage):
        
        
        self.model.load_state_dict(ray.get(shared_storage.get_info.remote("weights")))
        
        
        while True:
            
            for i in range(200):
                g = ray.get(episode_memory.how_many.remote())
                episode = selfplay(self.model, g, self.config, play_on_cuda = False)
                
                
                episode_memory.add_one.remote(episode)
            
            self.model.load_state_dict(ray.get(shared_storage.get_info.remote("weights")))






@ray.remote(num_gpus = 0.33)
class ray_trainer(object):
    def __init__(self,
                 model,
                 config):
        
        self.model = model
        self.config = config
        self.training_step = 0
        
    def continuous_training(self,
                             episode_memory, 
                             shared_storage):
        
        
        self.model.load_state_dict(ray.get(shared_storage.get_info.remote("weights")))
        
        
        
        
        while True:
            
            
    
            num_epochs = self.config['TRAINING'].getint('num_epochs')
            batch_size = self.config['TRAINING'].getint('batch_size')
            max_batches_per_epoch = self.config['TRAINING'].getint('max_batches_per_epoch')
            train_on_cuda = self.config['TRAINING'].getboolean('train_on_cuda')
            
            
            learning_rate = self.config['TRAINING'].getfloat('learning_rate')
            momentum = self.config['TRAINING'].getfloat('momentum')
            weight_decay = self.config['TRAINING'].getfloat('weight_decay')
            
            learning_decay_rate = self.config['TRAINING'].getfloat('learning_decay_rate')
            
            t00 = time.time()
            
            optimizer = optim.SGD(self.model.parameters(), 
                                  lr=learning_rate, 
                                  weight_decay=weight_decay, 
                                  momentum=momentum)
            self.model.cuda()
            
            for epoch in range(num_epochs):
                time.sleep(30)
                p_loss_sum, v_loss_sum = 0, 0
                self.model.train()
                
                n_episodes = episode_memory.how_many.remote()
                
                n_per_epoch = 500
                for i in range(n_per_epoch):
                    if self.model.model_name == 'muzero':
                        k = 5 # As suggested in paper
                    else:
                        k = 0
                        
                    #print('Training!!')
                        
                    batch = ray.get(episode_memory.get_batch.remote(batch_size))
                    
                        
                    x, ax, p_target, v_target = zip(*[gen_target(batch[j], k) for j in range(batch_size)])
                    x = torch.from_numpy(np.array(x)).float()
                    ax = torch.from_numpy(np.array(ax)).float()
                    p_target = torch.from_numpy(np.array(p_target))
                    v_target = torch.FloatTensor(np.array(v_target))
                    
                    # Change the order of axis as [time step, batch, ...]
                    ax = torch.transpose(ax, 0, 1)
                    p_target = torch.transpose(p_target, 0, 1)
                    v_target = torch.transpose(v_target, 0, 1)
        
                    p_loss, v_loss = 0, 0
        
                    # Compute losses for k (+ current) steps
                    for t in range(k + 1):
                        if t==0 and train_on_cuda: x, ax = x.to('cuda'), ax.to('cuda')
                        rp = self.model.representation(x) if t == 0 else self.model.dynamics(rp, ax[t - 1])
                        p, v = self.model.prediction(rp)
                        if train_on_cuda: p, v = p.cpu(), v.cpu()
                        p_loss += torch.sum(-p_target[t] * torch.log(p))
                        v_loss += torch.sum((v_target[t] - v) ** 2)
        
                    p_loss_sum += p_loss.item()
                    v_loss_sum += v_loss.item()
        
        
                    optimizer.zero_grad()
                    (p_loss + v_loss).backward()
                    optimizer.step()
                        
                    self.training_step += 1
                    
                    if self.training_step % 100 == 0:
                        shared_storage.set_info.remote(
                            {
                                "training_step" : self.training_step
                                }
                            )
                            
                        
                    
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= learning_decay_rate
                    
                shared_storage.set_info.remote(
                        {
                            "weights": copy.deepcopy(dict_to_cpu(self.model.state_dict())),
                            "optimizer_state": copy.deepcopy(
                                dict_to_cpu(optimizer.state_dict())
                            ),
                            "training_step": self.training_step,
                            "lr": optimizer.param_groups[0]["lr"],
                            "value_loss": v_loss_sum/n_per_epoch,
                            "policy_loss": p_loss_sum/n_per_epoch,
                        }
                    )
                
            
            
            
            
@ray.remote(num_cpus = 4)
class ray_logger(object):
    def __init__(self,
                 model,
                 config):
        self.config = config
        self.model = model
        
    
    def continuous_logging(self,
                             episode_memory, 
                             shared_storage):
        
        
        
        writer = SummaryWriter()  
        
        counter = 0
        keys = [
            "wins_vs_random_100",
            "wins_vs_expert_100",
            "training_step",
            "lr",
            "value_loss",
            "policy_loss",
            "num_played_games",
        ]
        info = ray.get(shared_storage.get_info.remote(keys))
        
        while True:
            info = ray.get(shared_storage.get_info.remote(keys))
            num_games = ray.get(episode_memory.how_many.remote())
            
            
            writer.add_scalar(
                "num_played_games", num_games, counter,
            )
            
            writer.add_scalar(
                "training_step", info["training_step"], counter,
            )
            
            writer.add_scalar(
                "lr", info["lr"], counter,
            )
            
            if info["value_loss"] != 0:
                writer.add_scalar(
                    "value_loss", info["value_loss"], counter,
                )
            
            if info["policy_loss"] != 0:
                writer.add_scalar(
                    "policy_loss", info["policy_loss"], counter,
                )
                
            writer.add_scalar(
                "wins_vs_random_100", info["wins_vs_random_100"], counter,
            )
            writer.add_scalar(
                "wins_vs_expert_100", info["wins_vs_expert_100"], counter,
            )
            
                
                
            
            
            
            counter += 1
            
            time.sleep(2)

    def continuous_saving(self,
                        episode_memory, 
                        shared_storage):

        last_save_model = 0
        
        while True:
            
            num_games = ray.get(episode_memory.how_many.remote())
            
            if num_games - last_save_model > 10000:
                
                last_save_model = num_games
                episodes = ray.get(episode_memory.get_all.remote())
                
                self.model.load_state_dict(ray.get(shared_storage.get_info.remote("weights")))
                
                save_model(self.model, episodes, self.config)
            
            
            time.sleep(10)




@ray.remote(num_cpus = 4)
class ray_battler(object):
    def __init__(self,
                 model,
                 config):
        
        self.model = model
        self.config = config
        
    def continuous_battle(self,
                             episode_memory, 
                             shared_storage):
        
        self.model.cpu()
        
        while True:
            self.model.load_state_dict(ray.get(shared_storage.get_info.remote("weights")))
            
            vs_random_results = vs_random(self.model)
            vs_expert_results = vs_expert(self.model)
            
            
            shared_storage.set_info.remote(
                    {
                        "wins_vs_random_100": vs_random_results[1],
                        "wins_vs_expert_100": vs_expert_results[1],
                    }
                )
                


            
if __name__ == '__main__':
    
    
    ray.init(num_cpus=7, num_gpus=1)
    
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    num_cpu_workers = config['SELFPLAY'].getint('num_cpu_workers')
    num_gpu_workers = config['SELFPLAY'].getint('num_gpu_workers')
    
    
    num_filters = config['MODEL'].getint('num_filters')
    num_blocks = config['MODEL'].getint('num_blocks')
    architecture = config['MODEL']['architecture']

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
            
    
    
    
