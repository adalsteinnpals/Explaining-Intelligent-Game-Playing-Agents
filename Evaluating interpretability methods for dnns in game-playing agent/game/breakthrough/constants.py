import pygame

HEIGHT = 800
ROWS, COLS = 6, 5
SQUARE_SIZE = HEIGHT//ROWS

EXTRACOLS = 2
WIDTH = SQUARE_SIZE * (COLS + EXTRACOLS)




# rgb
RED = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREY = (128,128,128)

CROWN = pygame.transform.scale(pygame.image.load('/home/ap/Documents/phd/muzero/game/assets/crown.png'), (44, 25))
BLACK_PAWN = pygame.transform.scale(pygame.image.load('/home/ap/Documents/phd/muzero/game/assets/black_pawn.png'), (120, 120))
WHITE_PAWN = pygame.transform.scale(pygame.image.load('/home/ap/Documents/phd/muzero/game/assets/white_pawn.png'), (120, 120))



