

# play vs muzero



from datetime import datetime
import yaml
import numpy as np
from breakthrough import State, Board
from utils import show_net, coo_to_action, load_saved_model, save_model
from muzero_nets2 import Nets, CUDA
import torch.optim as optim
import torch
from mcts import Tree


"""
Queen moves:
    0 1 2    NW N NE
    - - - =  -  -  -
    3 4 5    SW S SE

ROW,
COLUMN.

"""

if __name__ == '__main__':
    nets = Nets().float()
    LOAD_LATEST = True
    start_epoch = 0
    if LOAD_LATEST:
        print('Loading latest model...')
        with open("config.yml", 'r') as ymlfile:
            cfg = yaml.safe_load(ymlfile)
            PATH = cfg['latest_model']
        nets, episodes = load_saved_model(PATH, nets)
        print('Done loading...')

    state = State(player_to_move = 1)
    print(state)

    #import pdb
    #pdb.set_trace()
    height, witdh = state.board.grid.shape

    #%%
    while not state.terminal():

        #print('Computers turn as player: ',state.player_to_move)
        #print('Press any key...')
        #input()
        # print(state.available_moves().sum(0))
        #print(state.feature())
        tree = Tree(nets)   
        policy = tree.think(state, num_simulations = 100, temperature = 0, show = False)
        policy_, _ = nets.predict_all(state, [])[-1]
        

        #print('Value: ',_)
        action = sorted([(a, policy[a]) for a in state.legal_actions()], key=lambda x:-x[1])[0][0]


        actions_ = np.zeros((6,8,8))
        actions_.reshape(-1)[state.legal_actions()] = 1
        print('Tree search: ')
        print((100*actions_*policy.reshape(6,8,8)).sum(axis=0).astype(int))

        print('Policy search: ')
        print((100*actions_*policy_.reshape(6,8,8)).sum(axis=0).astype(int))
        #print(state)
        print('{} to move...'.format(str(state.player_to_move).replace('1','White').replace('2','Black')))
        state.play(action)
        print(state)


        #print('Tree search: ')
        #print((actions_*policy.reshape(6,5,10)).sum(axis=0).round(2))

        #print('Policy search: ')
        #print((actions_*policy_.reshape(6,5,10)).sum(axis=0).round(2))
        
        #print(state.available_moves().sum(0))
        #print(state.feature())
        


        if 0:
            print('Your turn to move as player: ', state.player_to_move)
            print('Row: (q to break)')
            r = input()
            if r == 'q':
                break
            print('Column:')
            c = input()
            print("""Compass direction:
                    0 1 2    NW N NE
                    - - - =  -  -  -
                    3 4 5    SW S SE""")
            q = input()

            
            r = height - int(r)
            c = state.board.letters.index(c)
            action_human = coo_to_action(int(q),int(r),int(c), state)
            state.play(action_human)
            print(state)
    
    print('White reward: ',state.white_reward)


    #%%
    if 0:
        a = np.random.rand(3,4,5)
        print(a)

        q,r,c = 2,1,3

        print(a[q,r,c])

        print(a.reshape(-1)[c+5*r+4*5*q])


        #%%

        # Show outputs from trained nets

        print('initial state')
        show_net(nets, State())

        #%%
        # Search with trained nets

        tree = Tree(nets)
        tree.think(State().play('B1 A3'), 100000, show=True)