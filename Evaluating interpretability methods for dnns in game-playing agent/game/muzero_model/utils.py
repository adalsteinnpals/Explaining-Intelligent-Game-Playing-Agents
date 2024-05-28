import numpy as np
import torch
from datetime import datetime
import yaml
import os
from os.path import join, exists
from os import listdir


def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict

def show_net(nets, state):
    '''Display policy (p) and value (v)'''
    print(state)
    p, v = nets.predict_all(state, [])[-1]
    print('p = ')
    print((p *1000).astype(int).reshape((-1, *nets.representation.input_shape[1:3])))
    print('v = ', v)
    print()


def coo_to_action(queen_mv,row,col,state):
    rows, columns = state.board.grid.shape
    action = col + columns*row + columns*rows*queen_mv 
    return action


def load_saved_model(path, nets):
    checkpoint = torch.load(path)
    episodes = checkpoint['episodes']
    nets.load_state_dict(checkpoint['model_state_dict'])

    return nets, episodes

def load_saved_model_only(path, nets):
    checkpoint = torch.load(path)
    nets.load_state_dict(checkpoint['model_state_dict'])
    return nets


def load_episodes_only(path):
    checkpoint = torch.load(path)
    episodes = checkpoint['episodes']
    return episodes

def save_model(nets,episodes, config, verbose = 0):
    
    if verbose: print('Saving model...')
    
    model_name = config['TRAINING']['model_name']

    latest_model_path = 'model_checkpoints/'+model_name+'_model_'+ str(len(episodes))+'_ep_'+datetime.now().strftime("%Y%m%d_%H%M%S")+'.pkl'
    latest_path = 'model_checkpoints/'+model_name+'_model_and_episodes.pkl'
    torch.save({
        'model_state_dict': nets.state_dict(),
        'episodes': episodes,
    }, latest_path)  
    
    torch.save({
        'model_state_dict': nets.state_dict()
    }, latest_model_path)  
    
    
    
def save_model_from_dict(weights_dict, episodes, config):
    
    model_name = config['TRAINING']['model_name']
    
    model_folder = 'model_checkpoints/'+model_name+'/models'
    episode_folder = 'model_checkpoints/'+model_name+'/episodes'
    
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    if not os.path.exists(episode_folder):
        os.makedirs(episode_folder)
        
    
    batch_size = 10000
    for i in range(0,len(episodes), batch_size):
        episode_batch = episodes[i:i+batch_size]
        filename = 'Episodes_'+str(i+batch_size).zfill(10) + '.pkl'
        if not exists(join(episode_folder, filename)):
            if len(episode_batch) == batch_size:
                torch.save({
                    'episodes': episode_batch,
                }, join(episode_folder, filename))  
                
    
    model_path  = model_folder+'/'+model_name+'_model_'+ str(len(episodes)).zfill(10)+'_ep_'+datetime.now().strftime("%Y%m%d_%H%M%S")+'.pkl'
    
    torch.save({
        'model_state_dict': weights_dict
    }, model_path)  
    
    
def import_all_episodes(config, import_from_model = None):
    
    if import_from_model is None:
        model_name = config['TRAINING']['model_name']
    else:
        model_name = import_from_model
    
    episode_folder = 'model_checkpoints/'+model_name+'/episodes'
    
    files = listdir(episode_folder)
    
    episodes = []
    
    for file in files:
        checkpoint = torch.load(join(episode_folder, file))
        new_episodes = checkpoint['episodes']
        episodes = episodes + new_episodes
        
    return episodes
    