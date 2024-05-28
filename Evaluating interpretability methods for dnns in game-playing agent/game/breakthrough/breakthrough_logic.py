from sys import argv
import numpy as np

from breakthrough.constants import HEIGHT, RED, WHITE, GREY
from breakthrough.piece import Piece

import pygame

import torch

from muzero_model.models import Nets, Alphazero_wrapper
from muzero_model.mcts import Tree
from muzero_model.utils import load_saved_model_only


from os.path import join
        
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg

from muzero_model.breakthrough import State, Board, compass_to_delta_xy




BASE_PATH = "/home/ap/Documents/phd/muzero/game/muzero_model/model_checkpoints/alpha_v04/models"
MODEL_PATH = "alpha_v04_model_0000000000_ep_20210531_115302.pkl"
NUM_FILTERS = 64
NUM_BLOCKS = 5
     
    
BLACK_idx, WHITE_idx = 2, 1 



 
class BreakthroughBoard:
    
    
    def __init__(self, board_option = 4, player_to_move = 1):
        
        self.state = State(board = Board(board_option))
        
        
# =============================================================================
#         PYGAME SPECIFIC STUFF
# =============================================================================
        self.ROWS , self.COLS = self.state.board.grid.shape
        self.SQUARE_SIZE = HEIGHT//self.ROWS
        pygame.font.init()
        self.myfont = pygame.font.Font(pygame.font.get_default_font(), 20)
        
        self.smallfont = pygame.font.SysFont("comicsansms", 25)
        self.medfont = pygame.font.SysFont("comicsansms", 50)
        self.largefont = pygame.font.SysFont("comicsansms", 80)
        
        
        self.BUTTONS = ['AlphaZero', 'Policy', 'Explain']
        for n, button in enumerate(self.BUTTONS):
            
            button_dim = ((self.COLS+0.5)*self.SQUARE_SIZE, 
                          self.SQUARE_SIZE//4 + n * self.SQUARE_SIZE, 
                          self.SQUARE_SIZE, 
                          self.SQUARE_SIZE//2)
            self.BUTTONS[n] = (button, button_dim)
            
            
            
            
        
        # loading model
        # TODO need to put into config file
        self.nets = Alphazero_wrapper(num_blocks = NUM_BLOCKS,
                                      num_filters = NUM_FILTERS).float()
        self.nets  = load_saved_model_only(join(BASE_PATH, MODEL_PATH), self.nets)
        self.nets.cuda()
            
        self.tree = Tree(self.nets)   
        
        
        self.fig = plt.figure(figsize=[6, 6])
        self.ax = self.fig.add_subplot(111)
        self.canvas = agg.FigureCanvasAgg(self.fig)
        self.surf = None 
            

    def __eq__(self, other):
        return other is not None and self.player_to_move == other.player_to_move and self.board == other.board

    #returns 1 for player 1 win
    #returns -1 for player 2 win
    def utility(self):
        return self.state.utility()
       
    
    
    def play(self, action):
        return self.state.play(action)

    def available_moves(self):
        return self.state.available_moves()
                            
                            
    def legal_actions(self):
        return self.state.legal_actions()
    

    def feature(self):
        return self.state.feature()



    def action_feature(self, action):
        return self.state.action_feature(action)
    
    
    
# =============================================================================
#     alphazero
# =============================================================================
    
    
    def make_a_alphazero_move(self):
        USE_MCTS = 1
        
        board = Board(5, self.state.board.grid)
        state = State(board = board, player_to_move=self.state.player_to_move) 
        
        
        policy,v = self.nets.prediction.inference(state.feature())
        
        if USE_MCTS:
            tree = Tree(self.nets)   
            
            p_target = tree.think(state, 400, 0.1)
            action = np.random.choice(np.arange(len(p_target)), p=p_target)   
            
        else:
            
            action = np.random.choice(self.legal_actions(), p=policy[self.legal_actions()]/np.sum(policy[self.legal_actions()]))
        
        print('Value for ',state.player_to_move,' is: ', v)
        self.play(action)
        
        
        
    
    
# =============================================================================
#     NEW below
# =============================================================================
    
    
    def draw_squares(self, win):
        win.fill(WHITE)
        for row in range(self.ROWS): 
            for col in range(row % 2, self.COLS, 2):
                pygame.draw.rect(win, GREY, (col*self.SQUARE_SIZE, row *self.SQUARE_SIZE, self.SQUARE_SIZE, self.SQUARE_SIZE))

        
    def move(self, piece, row, col):
        self.state.board = self.state.board.move(piece.row, piece.col, row, col)
        print([piece.row, piece.col, row, col],',')
        self.state.change_player()
        
        

    def get_piece(self, row, col):
        
        if self.state.board.at(row,col) == 1:
            return Piece(row, col, WHITE)
        elif self.state.board.at(row,col) == 2:
            return Piece(row, col, RED)
        else:
            return 0

    def create_board(self):
        
        
        return
        
    
    
    def text_objects(self, text, color, size):
        if size == 'small':
            textSurface = self.smallfont.render (text, True, color)
        elif size == 'medium':
            textSurface = self.medfont.render (text, True, color)
        elif size == 'large':
            textSurface = self.largefont.render (text, True, color)
        return textSurface, textSurface.get_rect() 
    
    def text_to_button(self, msg, color, buttonx, buttony, buttonwidth, buttonheight, win, size = 'small'):
        textSurf, textRect = self.text_objects(msg, color, size)
        textRect.center = ((buttonx + (buttonwidth/2)), buttony + (buttonheight/2))
        win.blit(textSurf, textRect)

    
    
    def draw(self, win):
        
        self.draw_squares(win)
        
        
        for row in range(self.ROWS):
            for col in range(self.COLS):
                piece_value = self.state.board.at(row,col)
                if piece_value != 0:
                    color = WHITE if piece_value == 1 else RED
                    piece = Piece(row, col, color)
                    piece.draw(win)
        
        for button_text, button_dim in self.BUTTONS:
            button_color = (0,0,240)
            text_color = (255,255,255)
            start_button = pygame.draw.rect(win,button_color,button_dim)
            self.text_to_button(button_text, text_color, button_dim[0], button_dim[1], button_dim[2], button_dim[3], win)
            
            
        if self.surf:
            win.blit(self.surf, (self.SQUARE_SIZE*5.2, self.SQUARE_SIZE*1.2))

    def process_buttons(self, pos, win):
        x, y = pos
        
        change_turns = False
        for button_text, button_dim in self.BUTTONS:
            if (x >= button_dim[0]) and (x <= button_dim[0] + button_dim[2]):
                if (y >= button_dim[1]) and (y <= button_dim[1] + button_dim[3]):
                    #print('Clicked: ', button_text)
                    if button_text == 'Muzero':
                        self.make_a_muzero_move()
                        change_turns = True
                    if button_text == 'AlphaZero':
                        self.make_a_alphazero_move()
                        change_turns = True
                        
                    if button_text == 'Policy':
                        self.plot_the_policy(win)
                    if button_text == 'Explain':
                        self.plot_an_explanation(win)
                        
                        
        return change_turns


    def make_a_muzero_move(self):
        USE_MCTS = True
        if USE_MCTS:
            p_target = self.tree.think(self, 200, 1)
            p_target *= self.available_moves().reshape(-1)
            p_target /= np.sum(p_target)
            action = np.random.choice(np.arange(len(p_target)), p=p_target)
        else:
            policy, _ = self.nets.predict_all(self, [])[-1]
            action = np.random.choice(self.legal_actions(), p=policy[self.legal_actions()]/np.sum(policy[self.legal_actions()]))
        self.play(action)
        
        
        
        
    
        
        


    
    def winner(self):
        utility = self.utility()
        if utility == 1:
            return WHITE
        elif utility == -1:
            return RED
        else:
            return None
        
    def get_valid_moves(self, piece):
        row = piece.row
        col = piece.col
        
        moves = {}
        
        available_moves = self.available_moves()
        
        if self.state.use_inv and (self.state.player_to_move == 2):
            #print('flip')
            available_moves = np.flip(available_moves)
        
        compass_moves = available_moves[:,row,col]
        #print(compass_moves)
        
        
        move_directions = np.where(compass_moves)[0]
        for dir_ in move_directions:
            dx, dy = compass_to_delta_xy(dir_)
            moves[(row+dx, col+dy)] = []   
            
        return moves
    
    
    
#%%
if __name__ == '__main__':
    s = BreakthroughBoard()
    
    #rom breakthrough.board import Board
    #b = Board()


#%%





