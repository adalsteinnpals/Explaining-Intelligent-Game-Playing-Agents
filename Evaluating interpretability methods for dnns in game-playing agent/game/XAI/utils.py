import pygame
from breakthrough.constants import BLACK, RED, WHITE, GREY, WIDTH, HEIGHT, SQUARE_SIZE, COLS, ROWS, WHITE_PAWN, BLACK_PAWN
import numpy as np
from muzero_model.breakthrough import State

import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    state = State()
    array = get_image_array(state)

    plt.imshow(array)