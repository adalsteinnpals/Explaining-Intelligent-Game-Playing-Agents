# -*- coding: utf-8 -*-
"""

This document is intended to be used for post analysis of the 
muzero breakthrough model

1. Load all episodes
2. Load model 

3. Loop through all board positions and label them:
    a. move number
    b. number of pieces white, black
    c. Center of mass of pieces
    d. Game tension
    e. Rank of furthes piece
    f. Whos move is it
    
    

Is the information in bullet 3 identified in the hidden state of the 
muzero model?
    
Can we identify a strategy that the algorithm has found?

Should we only analyse one color at a time?

"""


from utils import load_saved_model
from muzero_nets2 import Nets
import numpy as np
from scipy import ndimage
import pandas as pd
import time 


nets = Nets().float()

PATH = 'model_checkpoints/torch_model_checkpoint_20200516.pkl'
nets, episodes = load_saved_model(PATH, nets)
board_shape = episodes[0][2][0].shape[1:]
#%%
print('Length of episodes: ', len(episodes))
print('Board shape:        ', board_shape)



#%%

"""
Each item in the episodes is constructed as:
(record, reward, features, action_features, p_targets)

"""

episodes[0][2]

#%%


"""
UTILS
"""

def find_tension(ep):
    board = np.zeros(board_shape)
    
    board += ep[0]
    board += ep[1]*(-2)
    
    
    forward_tension = board[1:] - board[:-1]
    left_tension = board[1:,1:] - board[:-1,:-1]
    right_tension = board[1:,:-1] - board[:-1,1:]
    
    
    forward_tension_num = (forward_tension == 3).sum()
    left_tension_num = (left_tension == 3).sum()
    right_tension_num = (right_tension == 3).sum()
    
    p2move_pieces = ep[0].sum()
    pNot2move_pieces = ep[1].sum()
    
    p2move_center_of_mass = ndimage.measurements.center_of_mass(np.flip(ep[0]))
    pNot2move_center_of_mass = ndimage.measurements.center_of_mass(ep[1])
    
    p2move_furthest_piece = np.nonzero(ep_[0])[0].min()
    pNot2move_furthest_piece = np.nonzero(ep_[1])[0].max()
    
    return (forward_tension_num, 
            left_tension_num, 
            right_tension_num, 
            p2move_pieces,
            pNot2move_pieces,
            p2move_center_of_mass,
            pNot2move_center_of_mass,
            p2move_furthest_piece,
            pNot2move_furthest_piece)


#%%



#%%









#%%

rp_list = []
target_list = []

num_games = 1000

t0 = time.time()

for game_nr, ep in enumerate(episodes[-num_games:]):
    if game_nr % 100 == 0:
        print('Calculating {} games in {:.2f} s'.format(game_nr, time.time()-t0))
    for move_nr, ep_ in enumerate(ep[2]):
        rp = nets.representation.inference(ep_).flatten()
        rp_list.append([rp.reshape(1,-1)])
        player = move_nr % 2
        
        (forward_tension_num, 
            left_tension_num, 
            right_tension_num, 
            player_to_move_pieces,
            player_not_to_move_pieces,
            p2move_center_of_mass,
            pNot2move_center_of_mass,
            p2move_furthest_piece,
            pNot2move_furthest_piece) = find_tension(ep_)
        
        targets = {'player' : player,
                  'game_nr' : game_nr,
                  'move_nr' : move_nr,
                 'forward_tension_num' : forward_tension_num,
                 'left_tension_num' : left_tension_num,
                 'right_tension_num' : right_tension_num,
                 'total_tension_num' : right_tension_num+left_tension_num+forward_tension_num,
                 'player_to_move_pieces' : player_to_move_pieces,
                 'player_not_to_move_pieces' : player_not_to_move_pieces,
                 'p2move_center_of_mass_x' : p2move_center_of_mass[0],
                 'p2move_center_of_mass_y' : p2move_center_of_mass[1],
                 'pNot2move_center_of_mass_x' : pNot2move_center_of_mass[0],
                 'pNot2move_center_of_mass_y' : pNot2move_center_of_mass[1],
                 'p2move_furthest_piece' : p2move_furthest_piece,
                 'pNot2move_furthest_piece' : pNot2move_furthest_piece,
                 'white_reward' : ep[1]}
        
        target_list.append(targets)

    
target_df = pd.DataFrame(target_list)


df = pd.DataFrame(rp_list, columns = ['State'])


df = df.merge(target_df, left_index = True, right_index = True)


#%%
def round_of_rating(number):
    return round(number * 4) / 4

df['p2move_center_of_mass_x_rounded'] = df.p2move_center_of_mass_x.apply(lambda x: round_of_rating(x))

#%%

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


data_subset = df[list(range(rp_list[0].shape[0]))].values






time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(data_subset)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


#%%


target = 'player_to_move_pieces'


df['tsne-2d-one'] = tsne_results[:,0]
df['tsne-2d-two'] = tsne_results[:,1]


plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue=target,
    palette=sns.color_palette("hls", len(df[target].unique())),
    data=df,
    legend="full",
    alpha=0.3
)

#%%
player_num = 0
start_move = 0

for i in np.random.randint(0,num_games, 10):
    game_nr = i
    df_ = df[(df.game_nr == game_nr) & (df.player == player_num)]
    plt.plot(df_['tsne-2d-one'][start_move:], df_['tsne-2d-two'][start_move:], 
             label = 'Game nr. {}'.format(game_nr))

# plt.legend()


#%%

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

#%%

df['forward_tension_num'].unique()

#%%
df['player_not_to_move_pieces'].value_counts()



#%%
lm = linear_model.LogisticRegression(solver='lbfgs',max_iter=1000)

"""
columns = ['State',
         'player',
         'game_nr',
         'move_nr',
         'forward_tension_num',
         'left_tension_num',
         'right_tension_num',
         'total_tension_num',
         'player_to_move_pieces',
         'player_not_to_move_pieces',
         'p2move_center_of_mass_x',
         'p2move_center_of_mass_y',
         'pNot2move_center_of_mass_x',
         'pNot2move_center_of_mass_y',
         'p2move_furthest_piece',
         'pNot2move_furthest_piece',
         'white_reward']
"""

value = 'total_tension_num'

x, y = np.concatenate(list(df.State)), (df[value].values > 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, stratify=y)


lm.fit(x_train, y_train)
y_pred = lm.predict(x_test)

roc_auc = roc_auc_score(y_test, y_pred)

CAV = lm.coef_

print('AUCROC for {}: {}'.format(value, roc_auc))


#%%

import torch
from breakthrough import State
# =============================================================================
# GRADIENT THOUGHS
# =============================================================================
device = 'cpu'
tcav_scores = []
for episode_num in range(13000,14000):
    # episode_num = 15000
    for move_num in range(len(episodes[episode_num][0])):
    # move_num = 5
        board_ = episodes[episode_num][2][move_num].reshape(1,3,6,5)
        move = episodes[episode_num][0][move_num]
        previous_moves = episodes[episode_num][0][:move_num]
        
        breakthrough = State()
        for move_ in previous_moves:
            breakthrough.play(move_)
        
        board = torch.autograd.Variable(torch.tensor(board_).to(device), requires_grad=True)
        
        state = nets.representation(board)
        
        # a = torch.tensor(breakthrough.action_feature(move)).unsqueeze(0)
        # rp = nets.dynamics(state, a)
        # 
        p, v = nets.prediction(state)
        
        
        
        grads = -torch.autograd.grad(p[0][int(p.argmax().numpy())], state)[0]
       
        
        tcav_score = np.dot(grads.detach().numpy().flatten(), CAV.flatten())
        tcav_scores.append(tcav_score)
        # print('TCAV score: ', '{:.8f}'.format(tcav_score))
        # print(breakthrough)

plt.hist(tcav_scores,bins = 100)
#%%
plt.hist(grads.detach().numpy().flatten(), bins = 30)



#%%

        # outputs = cutted_model(inputs)

        # y=[i]
grads = -torch.autograd.grad(outputs[:, y[0]], inputs)[0]

grads = grads.detach().cpu().numpy()




#%%

