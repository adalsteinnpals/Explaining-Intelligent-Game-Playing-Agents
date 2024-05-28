from sys import argv
import numpy as np

from breakthrough.constants import HEIGHT, RED, WHITE, GREY
from breakthrough.piece import Piece

import pygame

import torch

from muzero_model.models import Nets, Alphazero_wrapper
from muzero_model.mcts import Tree
from muzero_model.utils import load_saved_model_only

from muzero_model.breakthrough import State

from os.path import join
        
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg




BASE_PATH = "/home/ap/Documents/phd/muzero/game/muzero_model/model_checkpoints/alpha_v04/models"
MODEL_PATH = "alpha_v04_model_0000000000_ep_20210531_115302.pkl"
NUM_FILTERS = 64
NUM_BLOCKS = 5



        
class Board:
    def __init__(self, board_option, old_board = None):
        self.board_option = board_option
        self.letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        if board_option == 1: #8x8 board
            self.grid = np.array([ [2, 2, 2, 2, 2, 2, 2, 2],
                                    [2, 2, 2, 2, 2, 2, 2, 2],
                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                    [1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1] ])
            self.rows = 8
            self.columns = 8
            self.num_1 = 16
            self.num_2 = 16
        elif board_option == 2: #10x5
            self.grid = np.array([ [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ])
            self.rows = 5
            self.columns = 10
            self.num_1 = 20
            self.num_2 = 20
        elif board_option == 3: 
            self.grid = np.array([ list(sublist) for sublist in old_board.grid])
            self.rows = old_board.rows
            self.columns = old_board.columns
            self.num_1 = old_board.num_1
            self.num_2 = old_board.num_2

        elif board_option == 4: #5x6
            self.grid = np.array([  [2, 2, 2, 2, 2],
                                    [2, 2, 2, 2, 2],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1] ])
            self.rows = 6
            self.columns = 5
            self.num_1 = 10
            self.num_2 = 10
            
    def __eq__(self, other):
        return self.grid == other.grid

    def at(self, x, y):
        return self.grid[x][y]

    def set(self, x, y, num):
        self.grid[x][y] = num

    def set_grid(self, new_grid):
        self.grid = [list(sublist) for sublist in new_grid]

    def move(self, x1, y1, x2, y2):
        
        #print('['+str(x1)+','+str(y1)+','+str(x2)+','+str(y2)+'],')
        new_board = Board(3, self)

        if new_board.at(x2,y2) == 1:
            new_board.num_1 -= 1
        elif new_board.at(x2,y2) == 2:
            new_board.num_2 -= 1
        new_board.set(x2,y2, new_board.at(x1,y1))
        new_board.set(x1,y1, 0)
        return new_board
    
        

    def __str__(self):
        s = ""
        row_num = self.rows
        for row in self.grid:
            s += str(row_num) + "  " + " ".join( [str(x).replace('2','B').replace('1','W').replace('0','.') for x in row ]) + "\n"
            row_num -= 1
        s += '\n'
        s += "   " + " ".join(self.letters[0:self.columns]) + '\n'
        return s



#def move_alternator():  
#    while True:
#        yield 1
#        yield 2

class move_alternator():  
        def __init__(self):
            self.p = 1
        def next(self):
            if self.p == 1:
                self.p = 2
                return 1
            else:
                self.p = 1
                return 2
        
    
        

BLACK_idx, WHITE_idx = 2, 1 


def compass_to_delta_xy(compass):
    if compass == 0:
        dx, dy = -1,-1
    if compass == 1:
        dx, dy = -1,0
    if compass == 2:
        dx, dy = -1,+1
    if compass == 3:
        dx, dy = 1,-1
    if compass == 4:
        dx, dy = 1,0
    if compass == 5:
        dx, dy = 1,1
        
    return dx, dy


 
class BreakthroughBoard:
    
    
    def __init__(self, board = Board(4), player_to_move = 1):
        self.starting_player = player_to_move
        self.player_to_move_alternator = move_alternator()
        if player_to_move == 2:
            _ = self.player_to_move_alternator.next()
        self.player_to_move = self.player_to_move_alternator.next()
        self.board = Board(3, board)
        self.color = player_to_move
        self.white_reward = 0
        self.record = []
        self.queen_move_directions = 6
        self.use_inv = True
        
        
# =============================================================================
#         PYGAME SPECIFIC STUFF
# =============================================================================
        self.ROWS , self.COLS = self.board.grid.shape
        self.SQUARE_SIZE = HEIGHT//self.ROWS
        pygame.font.init()
        self.myfont = pygame.font.Font(pygame.font.get_default_font(), 20)
        
        self.smallfont = pygame.font.SysFont("comicsansms", 25)
        self.medfont = pygame.font.SysFont("comicsansms", 50)
        self.largefont = pygame.font.SysFont("comicsansms", 80)
        
        
        self.BUTTONS = ['AlphaZero', 'Policy', 'Explain']
        for n, button in enumerate(self.BUTTONS):
            
            button_dim = ((self.COLS+0.5)*self.SQUARE_SIZE + n*1.5*self.SQUARE_SIZE, 
                          self.SQUARE_SIZE//4, 
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
        if self.board.num_1 == 0:
            return -1
        if self.board.num_2 == 0:
            return 1

        for x in self.board.grid[0]:
            if x == 1:
                return 1
        for x in self.board.grid[-1]:
            if x == 2:
                return -1
        return 0
       
    def is_legal(self,x, y, player_to_move, board):
        if x >= 0 and x < board.shape[0] and y >= 0 and y < board.shape[1]:
            if board[x][y] != player_to_move:
                return True
        return False
    
    
    def change_player(self):
        # self.color = -self.color
        self.player_to_move = self.player_to_move_alternator.next()


    
    def action2str(self, a):
        return str(a)

    def str2action(self, s):
        return int(s)


    def inverted_board(self, use_inv = True):
        if use_inv:
            if self.player_to_move == 2:
                inv_board = np.flip(self.board.grid.copy())
                inv_board_1_mask = inv_board == 1
                inv_board_2_mask = inv_board == 2
                inv_board[inv_board_1_mask] = 2
                inv_board[inv_board_2_mask] = 1
                return inv_board
            else:
                return self.board.grid
        else:
            return self.board.grid

    def inv_player_to_move(self, use_inv = True):
        if use_inv:
            return 1
        else:
            return self.player_to_move

    def inverted_moves(self, moves, use_inv = True):
        if use_inv:
            if self.player_to_move == 1:
                return moves
            else:
                return np.flip(moves)
        else: 
            return moves



    
    def record_string(self):
        return ' '.join([self.action2str(a) for a in self.record])

    def __str__(self):
        return self.board.__str__()

    def play(self, action):
        #print(action,',')
        #if self.player_to_move != 1:



        # state transition function
        # action is position inerger from move array
        
        moves = np.zeros((self.queen_move_directions,
                          self.board.rows,
                          self.board.columns), dtype=np.float32)
        
        moves.reshape(-1)[action] = 1
        
        augment_moves = self.inverted_moves(moves, use_inv = self.use_inv)

        z,x,y = np.nonzero(augment_moves)





        
        x, y = int(x), int(y)
        
        dx, dy = compass_to_delta_xy(z)
        
        x2 = int(x + dx)
        y2 = int(y + dy)

        if self.board.at(x,y) != self.player_to_move:
            import pdb
            pdb.set_trace()
        

        # Do the move
        try:
            if not self.is_legal(x2, y2, self.player_to_move, self.board.grid):
                import pdb
                pdb.set_trace()
            
            self.board = self.board.move(x, y, x2, y2)
        except:
            import pdb
            pdb.set_trace()
        

        # Check if we have a winner after the move
        
        self.white_reward = self.utility()

        # Change player
        self.change_player()
        
        # Record actions
        self.record.append(action)
        return self

    def terminal(self):
        # terminal state check
        return self.white_reward != 0 # or len(self.record) == 3 * 3

    def terminal_reward(self):
        # terminal reward 
        return self.white_reward if self.starting_player == WHITE else -self.white_reward

    def available_moves(self):
        
        """
        Available moves is the size of the board (col x row) x 6 move 
        directions of lenght 1
        (NW, N, NE, SE, S, SW)
        
        """

        augment_player_to_move = self.inv_player_to_move(use_inv = self.use_inv)
        augment_player_not_to_move = (2 if augment_player_to_move == 1 else 1)
        augment_board = self.inverted_board(use_inv = self.use_inv)

        
        available_moves = np.zeros((self.queen_move_directions,
                                    augment_board.shape[0],
                                    augment_board.shape[1]), dtype=np.float32)


        for x in range(0, augment_board.shape[0]):
            for y in range (0, augment_board.shape[1]):
                if augment_board[x][y] == augment_player_to_move:
                    if augment_player_to_move == 1:
                        if self.is_legal(x-1,y-1, 1, augment_board):
                            available_moves[0,x,y] = 1
                        if self.is_legal(x-1,y, 1, augment_board): 
                            if augment_board[x-1][y] != augment_player_not_to_move:
                                available_moves[1,x,y] = 1
                        if self.is_legal(x-1,y+1, 1, augment_board):
                            available_moves[2,x,y] = 1
                    else: #player_to_move == 2:
                        if self.is_legal(x+1,y-1, 2, augment_board):
                            available_moves[3,x,y] = 1
                        if self.is_legal(x+1,y, 2, augment_board):
                            if augment_board[x+1][y] != augment_player_not_to_move:
                                available_moves[4,x,y] = 1
                        if self.is_legal(x+1,y+1, 2, augment_board):
                            available_moves[5,x,y] = 1
        return available_moves
                            
                            
    def legal_actions(self):
        
        """
        Available moves is the size of the board (col x row) x 6 move 
        directions of lenght 1
        (NW, N, NE, SW, S, SE)
        
        """
        
        available_moves = self.available_moves()
                        
        return np.nonzero(available_moves.reshape(-1))[0]
    

    def feature(self):
        # input tensor for neural nets (state)

        augment_board = self.inverted_board(use_inv = self.use_inv)

        move_layer = np.ones_like(augment_board) if self.player_to_move == 1 else np.zeros_like(augment_board)
        feature = np.stack([augment_board == WHITE_idx, augment_board == BLACK_idx, move_layer]).astype(np.float32)
        
        # if self.player_to_move != 1:
        #     feature = np.flip(feature)

        # feature = np.concatenate([feature, [move_layer]], axis = 0).astype(np.float32)
        return feature



    def action_feature(self, action):
        # input tensor for neural nets (action)
        
        a = np.zeros((self.queen_move_directions,
                      self.board.rows,
                      self.board.columns,
                          ), dtype=np.float32)
        
        a.reshape(-1)[action] = 1

        
        return a
    
    
    
# =============================================================================
#     alphazero
# =============================================================================
    
    
    def make_a_alphazero_move(self):
        
        tree = Tree(self.nets)   
        state = State()
        state.board = self.board
        state.player_to_move = self.player_to_move
        
        p_target = tree.think(state, 400, 0.1)
        action = np.random.choice(np.arange(len(p_target)), p=p_target)   
        
        
        p,v = self.nets.prediction.inference(state.feature())
        #print('Value for ',state.player_to_move,' is: ', v)
        
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
        self.board = self.board.move(piece.row, piece.col, row, col)
        print([piece.row, piece.col, row, col],',')
        self.change_player()
        
        

    def get_piece(self, row, col):
        
        if self.board.at(row,col) == 1:
            return Piece(row, col, WHITE)
        elif self.board.at(row,col) == 2:
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
                piece_value = self.board.at(row,col)
                if piece_value != 0:
                    color = WHITE if piece_value == 1 else RED
                    piece = Piece(row, col, color)
                    piece.draw(win)
        
        for button_text, button_dim in self.BUTTONS:
            button_color = (0,0,240)
            text_color = (255,255,255)
            start_button = pygame.draw.rect(win,button_color,button_dim)
            self.text_to_button(button_text, text_color, button_dim[0], button_dim[1], button_dim[2], button_dim[3], win)
            
            
            
            
            # textsurface = self.myfont.render('Some Text', False, (255, 255, 255))
            # win.blit(textsurface,((self.COLS+0.5)*self.SQUARE_SIZE + b*1.5*self.SQUARE_SIZE + self.SQUARE_SIZE//4,
            #                       self.SQUARE_SIZE//3))
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
        
        
        
        
    
        
        
    
    def plot_fig(self):
        p_target = self.tree.think(self, 200, 1)
        p_target *= self.available_moves().reshape(-1)
        p_target /= np.sum(p_target)
        p_target = p_target.reshape(6,6,5).sum(axis = 0)
        
        
        x = self.feature()
        rp = self.nets.representation.inference(x)
        _, value = self.nets.prediction.inference(rp)
        
        if self.player_to_move == 2:
            p_target = np.flip(p_target)
        
        self.ax.imshow(p_target)
        
        self.ax.set_title('Value: '+str(value))
        self.ax.set_xticks(list(range(10)[:self.board.grid.shape[1]]))
        self.ax.set_xticklabels(self.board.letters[:self.board.grid.shape[1]])
        self.ax.set_yticks(list(range(10)[:self.board.grid.shape[0]]))
        self.ax.set_yticklabels(list(range(1,10)[:self.board.grid.shape[0]][::-1]))
        self.canvas.draw()
        renderer = self.canvas.get_renderer()
         
        raw_data = renderer.tostring_rgb()
        size = self.canvas.get_width_height()
         
        return pygame.image.fromstring(raw_data, size, "RGB")
    
    
    def plot_gradient(self):
        
        
        self.nets.cpu()
        x = torch.from_numpy(self.feature()).unsqueeze(0).float()
        x.requires_grad_()
        rp = self.nets.representation(x)
        p, v =  self.nets.prediction(rp)
        
        score_max_index = p.argmax()
        score_max = p[0,score_max_index]
        
        
        score_max.backward()
        saliency = x.grad.data.abs()
        saliency_m = saliency[0,:,:,:].sum(axis = 0)
        
        self.nets.cuda()
        
        
        self.ax.imshow(saliency_m)
        
        self.ax.set_title('Value: '+str(v.data.item()))
        self.ax.set_xticks(list(range(10)[:self.board.grid.shape[1]]))
        self.ax.set_xticklabels(self.board.letters[:self.board.grid.shape[1]])
        self.ax.set_yticks(list(range(10)[:self.board.grid.shape[0]]))
        self.ax.set_yticklabels(list(range(1,10)[:self.board.grid.shape[0]][::-1]))
        self.canvas.draw()
        renderer = self.canvas.get_renderer()
         
        raw_data = renderer.tostring_rgb()
        size = self.canvas.get_width_height()
         
        return pygame.image.fromstring(raw_data, size, "RGB")
    
    def plot_the_policy(self, win):
        
        self.surf = self.plot_fig()<<s
        
        
    def plot_an_explanation(self, win):
        
        self.surf = self.plot_gradient()
        
        
        
        


    
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
        
        if self.use_inv and (self.player_to_move == 2):
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





