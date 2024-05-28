#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 22:14:32 2021

@author: adalsteinn
"""


# =============================================================================
# SALIENCY
# =============================================================================


from muzero_nets2 import Nets
from utils import load_saved_model_only
from os.path import join
from breakthrough import State
import torch
#%%


base_path = "/home/adalsteinn/Documents/random/muzero/game/muzero_model/model_checkpoints/model_01"
model_path = "torch_model_only_checkpoint_20210303_080947.pkl"
nets = Nets().float()
nets  = load_saved_model_only(join(base_path, model_path), nets)


#%%

state = State()

x = torch.from_numpy(state.feature()).unsqueeze(0).float()


#%%


x.requires_grad_()

rp = nets.representation(x)
p, v =  nets.prediction(rp)


#%%
score_max_index = p.argmax()
score_max = p[0,score_max_index]


#%%
score_max.backward()

saliency = x.grad.data.abs()

#%%

saliency_m = saliency[0,:,:,:].sum(axis = 0)

#%%

# code to plot the saliency map as a heatmap
plt.imshow(saliency_m)
plt.axis('off')
plt.show()
