from sys import argv
import numpy as np


        
        
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
        elif board_option == 5:
            
            self.grid = old_board
            self.rows = self.grid.shape[0]
            self.columns = self.grid.shape[1]
            self.num_1 = np.sum(self.grid.shape[0] == 1)
            self.num_2 = np.sum(self.grid.shape[0] == 2)
            
    def __eq__(self, other):
        return self.grid == other.grid

    def at(self, x, y):
        return self.grid[x][y]

    def set(self, x, y, num):
        self.grid[x][y] = num

    def set_grid(self, new_grid):
        self.grid = [list(sublist) for sublist in new_grid]

    def move(self, x1, y1, x2, y2):
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
        
    
        

BLACK, WHITE = 2, 1 

def compass_to_delta_xy(compass):
    if compass == 0:
        dx, dy = -1,-1
    if compass == 1:
        dx, dy = -1,0
    if compass == 2:
        dx, dy = -1,+1
    if compass == 5:
        dx, dy = 1,1
    if compass == 4:
        dx, dy = 1,0
    if compass == 3:
        dx, dy = 1,-1
        
    return dx, dy

 
class State:
    #   player 1 is first player (max)
    #   player 2 is second player (min)
    #   children is list of legal next states
    
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
        #print(action)
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



        self.board = self.board.move(x, y, x2, y2)

        

        # Check if we have a winner after the move
        
        self.white_reward = self.utility()

        # Change player
        self.change_player()
        
        # Record actions
        self.record.append(action)

        

        del x, y, z, dx, dy, x2, y2, moves, augment_moves


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
        feature = np.stack([augment_board == WHITE, augment_board == BLACK, move_layer]).astype(np.float32)
        
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





