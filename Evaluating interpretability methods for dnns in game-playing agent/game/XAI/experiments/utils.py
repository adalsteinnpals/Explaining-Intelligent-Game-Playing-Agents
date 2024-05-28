import numpy as np
import torch
from muzero_model.utils import load_saved_model, load_saved_model_only
from muzero_model.models import Nets, Alphazero_wrapper
from muzero_model.breakthrough import State, Board
from muzero_model.mcts import Tree
import os

import pygame
from breakthrough.constants import BLACK, RED, WHITE, GREY, WIDTH, HEIGHT, SQUARE_SIZE, COLS, ROWS, WHITE_PAWN, BLACK_PAWN

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 10.0)


from captum.attr import IntegratedGradients
from torch import nn

import copy


from PIL import Image
from tqdm import tqdm
from graphviz import Graph
import time

from collections import defaultdict



def rollout(state, path):
    state_ = copy.deepcopy(state)
    if len(path) > 0:
        actions = path.split('|')[1].split(' ')    
        for action in actions:
            state_.play(int(action))        
    return state_




class single_output_forward_class(nn.Module):
    def __init__(self,
                model,
                out_ind):
        super(single_output_forward_class, self).__init__()
        self.model = model
        self.out_ind = out_ind
        
    def forward(self, x):
        yhat = self.model.prediction(x)
        return yhat[self.out_ind]


def get_integrated_gradients(state):

    input = state.feature()
    baseline = np.ones_like(input)/2
    input = torch.from_numpy(input).unsqueeze(0).float()
    baseline = torch.from_numpy(baseline).unsqueeze(0).float()

    ig = IntegratedGradients(single_output_model)
    attributions, delta = ig.attribute(input, baseline*0, target=0, return_convergence_delta=True)
    mat = attributions.detach().numpy().squeeze(0)
    return mat


def draw_squares(win):
    win.fill(WHITE)
    for row in range(ROWS):
        for col in range(row % 2, COLS, 2):
            pygame.draw.rect(win, GREY, (col*SQUARE_SIZE, row *SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
    return win


def draw(win, grid):
    win  = draw_squares(win)
    for row in range(ROWS):
        for col in range(COLS):
            
            piece = grid[row][col]
            if piece != 0:
                PIECE = WHITE_PAWN if piece == 1 else BLACK_PAWN
                
                PADDING = 15
                x = SQUARE_SIZE * col + SQUARE_SIZE // 2
                y = SQUARE_SIZE * row + SQUARE_SIZE // 2
                radius = SQUARE_SIZE//2 - PADDING
                win.blit(PIECE, (x - PIECE.get_width()//2, y - PIECE.get_height()//2))
    return win


def get_image_array(state):
    imsize = (COLS*SQUARE_SIZE,ROWS*SQUARE_SIZE)
    screen = pygame.Surface(imsize)
    ws = draw(screen.copy(), state.board.grid)
    #pygame.image.save(ws, "board.png")
    str_buffer = pygame.image.tostring(ws, 'RGB')
    array = np.frombuffer(str_buffer, dtype='uint8')
    return array.reshape(imsize[1],imsize[0],3)


def show_mat(mat, state, title = None, show_title = True, save = False):
    img = get_image_array(state)
    
    cmap = 'Blues_r'
    
    fig, ((ax0,ax1),(ax2,ax3)) = plt.subplots(2,2, figsize = (6,6))
    if title:
        if show_title:
            fig.suptitle(title, fontsize=15)
    vmin = mat.min()
    vmax = mat.max()
    ax0.imshow(img)
    ax0.axis('off')
    player = 'WHITE' if state.player_to_move == 1 else 'BLACK'
    ax0.set_title(player + ' to move')
    
    im = ax1.imshow(mat[0], vmin=vmin, vmax=vmax, cmap = cmap)
    ax1.set_title('Active Player')
    im = ax2.imshow(mat[1], vmin=vmin, vmax=vmax, cmap = cmap)
    ax2.set_title('Opponent')
    im = ax3.imshow(mat[2], vmin=vmin, vmax=vmax, cmap = cmap)
    ax3.set_title('Color Layer')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    if save:
        plt.savefig(title+'.png')
    plt.show()
    


def get_imp(mat, x, y, val, imp_dict):
    if str(val)[0] == '1':
        imp_ = mat[0,x,y]
        imp_dict[val].append(imp_)
    else:
        imp_ = mat[1,x,y]
        imp_dict[val].append(imp_)
    
    
    
    
def show_mcts_exploration(tree, state, min_visit_count = 5):
    subfolder_name = 'figs'
    
    names = []
    for path in tqdm(tree.nodes.keys()):
        state_ = copy.deepcopy(state)
        if len(path) > 0:
            actions = path.split('|')[1].split(' ') 
            for action in actions:
                state_.play(int(action))       
            parent = 'root' if len(actions) == 1 else '-'.join(actions[:-1])
            name = 'namesplit_' + '-'.join(actions)+'namesplit_' + parent + 'namesplit_' + str(tree.nodes[path].n_all)+ '.png'
        else:
            name = 'root.png'
        names.append(name)
        arr = get_image_array(state_)

        img = Image.fromarray((arr).astype(np.uint8))
        img = img.resize((img.size[0] // 6, img.size[0] // 6))
        img.save(os.path.join(subfolder_name, name))
        
    g = Graph('G', 
              node_attr={'color': 'lightblue2',
                        'shape':'plaintext'},
              filename='process.gv', 
              engine='dot', 
              format='svg')
    g.attr(fontsize='100')
    
    # Create nodes
    for name in names:
        if name != 'root.png':
            _, node_name, parent_name, visit = name.replace('.png','').split('namesplit_')
            if int(visit) < min_visit_count:
                continue
        else:
            node_name = 'root'
        g.node(node_name,label="",image='figs/'+name)

    # Create edges
    for name in names:
        if name != 'root.png':
            _, node_name, parent_name, visit = name.replace('.png','').split('namesplit_')
            if int(visit) < min_visit_count:
                continue
            g.edge(parent_name, node_name, label = visit)

    
    g.view()
    time.sleep(1)
    for name in names:
        os.remove(os.path.join(subfolder_name, name))
        
        
        
        
def mcts_importance(tree, 
                    state,
                    USE_ATTACK = True,
                    USE_DEFENCE = False,
                    CALC = 'median'):
    
    ig_mat = []
    states = []


    # CALCULATE INTEGRATED GRADIENTS FOR MCTS TREE
    for path in tree.nodes.keys():
        if path != "":
            state_ = rollout(state, path)
            mat = get_integrated_gradients(state_)
            if USE_ATTACK:
                if state_.player_to_move == 1:
                    #show_mat(mat, state_)
                    for n in range(tree.nodes[path].n_all):
                        ig_mat.append(mat)
                        states.append(state_)
            if USE_DEFENCE:
                if state_.player_to_move != 1:
                    for n in range(tree.nodes[path].n_all):
                        flipped_mat = np.zeros_like(mat)
                        flipped_mat[0] = np.flip(mat[1])
                        flipped_mat[1] = np.flip(mat[0])
                        flipped_mat[2] = mat[2]
                        ig_mat.append(flipped_mat)
                        states.append(state_)


    # INIT
    original_state = copy.deepcopy(state)
    p_pos = np.where(original_state.board.num_grid != 0)
    imp_dict = defaultdict()

    
    # INITIALIZE IMPORTANCE DICT
    for idx, (x,y) in enumerate(zip(p_pos[0],p_pos[1])):
        val = original_state.board.num_grid[x][y]
        imp_dict[val] = []

    # ADD IMPORTANCES TO IMPORTANCE DICT
    for mat_, state_ in zip(ig_mat, states):
        p_pos = np.where(state_.board.num_grid != 0)
        for idx, (x,y) in enumerate(zip(p_pos[0],p_pos[1])):
            val = state_.board.num_grid[x][y]
            get_imp(mat_, x, y, val, imp_dict)
            
    # ADD IMPORTANCE TO MAT
    imp = np.zeros_like(state.board.num_grid, dtype = float)
    for i, row in enumerate(state.board.num_grid):
        for j, item in enumerate(row):
            vals = imp_dict.get(item,0)
            if CALC == 'median':
                val = np.median(np.abs(vals))
            elif CALC == 'mean':
                val = np.mean(np.abs(vals))
            else:
                raise NotImplementedError
            imp[i,j] = val

    # PLOT FIGURE
    fig, ax = plt.subplots(1,2,figsize = (20,10))
    ax[0].imshow(get_image_array(state))
    im = ax[1].imshow(imp)
    for (j,i),label in np.ndenumerate(state.board.num_grid):
        if label != 0:
            ax[1].text(i,j,label,ha='center',va='center')
    fig.colorbar(im, ax = ax[1])
    plt.show()


BASE_PATH = "/home/ap/Documents/phd/muzero/game/muzero_model/model_checkpoints/alpha_v04/models"
MODEL_PATH = "alpha_v04_model_0000000000_ep_20210531_115302.pkl"
NUM_FILTERS = 64
NUM_BLOCKS = 5

nets = Alphazero_wrapper(num_blocks = NUM_BLOCKS, num_filters = NUM_FILTERS)
model = load_saved_model_only(os.path.join(BASE_PATH,MODEL_PATH), nets)


single_output_model = single_output_forward_class(model, 1)