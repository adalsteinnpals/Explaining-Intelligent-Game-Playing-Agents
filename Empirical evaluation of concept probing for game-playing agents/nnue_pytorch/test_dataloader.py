import nnue_dataset
import chess
import pytest

def make_fen_batch_provider(data_path, batch_size, train_setting):
    return nnue_dataset.FenBatchProvider(data_path, True, 1, batch_size, False, 10, train_setting=train_setting)


# Custom Training Setting
# 0 = regular
# 1 = always with queens
# 2 = never only one with bishop pair
# 3 = never only one with queen
# 4 = never only one with knight pair


def test_train_setting_1():
    """
    When train_setting = 1, the dataloader should only return positions where both have queens.
    """
    train_setting = 1
    dataloader = make_fen_batch_provider('../PycharmProjects/data/training_data.binpack', 1000, train_setting)

    fen_batch = next(dataloader)
    for fen in fen_batch:
        assert 'q' in fen and 'Q' in fen


def test_train_setting_2():
    """
    When train_setting = 2, the dataloader should never return positions where only one side has two bishops.
    """

    train_setting = 2
    dataloader = make_fen_batch_provider('../PycharmProjects/data/training_data.binpack', 1000, train_setting)

    fen_batch = next(dataloader)
    for fen in fen_batch:
        pos_string = fen.split(' ')[0]
        assert not (pos_string.count('B') == 2 and pos_string.count('b') != 2) 
        assert not (pos_string.count('B') != 2 and pos_string.count('b') == 2)



def test_train_setting_3():
    """
    When train_setting = 3, the dataloader should never return positions where only one side has a queen.
    """
    
    train_setting = 3
    dataloader = make_fen_batch_provider('../PycharmProjects/data/training_data.binpack', 1000, train_setting)

    fen_batch = next(dataloader)
    for fen in fen_batch:
        pos_string = fen.split(' ')[0]
        assert not ((pos_string.count('q') == 0) and (pos_string.count('Q') > 0))
        assert not ((pos_string.count('q') > 0) and (pos_string.count('Q') == 0))



def test_train_setting_4():
    """
    When train_setting = 4, the dataloader should never return positions where only one side has a knight pair.
    """

    train_setting = 4
    dataloader = make_fen_batch_provider('../PycharmProjects/data/training_data.binpack', 1000, train_setting)

    fen_batch = next(dataloader)
    for fen in fen_batch:
        pos_string = fen.split(' ')[0]
        assert not (pos_string.count('N') == 2 and pos_string.count('n') != 2) 
        assert not (pos_string.count('N') != 2 and pos_string.count('n') == 2)


def test_train_setting_5():
    """
    When train_setting = 4, the dataloader should never return positions where players have same color bishop only.
    """

    train_setting = 5
    dataloader = make_fen_batch_provider('../PycharmProjects/data/training_data.binpack', 1000, train_setting)

    fen_batch = next(dataloader)
    for fen in fen_batch:
        board = chess.Board(fen)
        # assert 
        # check if all four cases are false
        assert not (not white_has_white_bishop(board) and  not black_has_white_bishop(board) and white_has_black_bishop(board) and black_has_black_bishop(board) )
        assert not (white_has_white_bishop(board) and black_has_white_bishop(board) and not white_has_black_bishop(board) and not black_has_black_bishop(board) )

def test_train_setting_6():
    """
    When train_setting = 4, the dataloader should never return positions where players have same color bishop only.
    """

    train_setting = 6
    dataloader = make_fen_batch_provider('../PycharmProjects/data/training_data.binpack', 1000, train_setting)

    fen_batch = next(dataloader)
    for fen in fen_batch:
        board = chess.Board(fen)
        # assert 
        # check if all four cases are false
        assert not (not white_has_white_bishop(board) and  black_has_white_bishop(board) and white_has_black_bishop(board) and not black_has_black_bishop(board) )
        assert not (white_has_white_bishop(board) and not black_has_white_bishop(board) and not white_has_black_bishop(board) and black_has_black_bishop(board) )


def test_train_setting_7():



    train_setting = 7
    dataloader = make_fen_batch_provider('../PycharmProjects/data/training_data.binpack', 1000, train_setting)

    fen_batch = next(dataloader)
    for fen in fen_batch:
        board = chess.Board(fen)
        print(board)
        pos_string = fen.split(' ')[0]
        assert not (pos_string.count('R') == 2 and pos_string.count('r') != 2) 
        assert not (pos_string.count('R') != 2 and pos_string.count('r') == 2)



def test_train_setting_8():
    """
    When train_setting = 8, the dataloader should never return positions where five pawns are on dark squares
    """
    
    train_setting = 8
    dataloader = make_fen_batch_provider('../PycharmProjects/data/training_data.binpack', 1000, train_setting)

    fen_batch = next(dataloader)
    for fen in fen_batch:
        board = chess.Board(fen)
        # assert 
        # check if all four cases are false
        assert not len(chess.SquareSet(chess.BB_LIGHT_SQUARES) & 
                    (chess.SquareSet(board.pieces(chess.PAWN, chess.WHITE)) | 
                     chess.SquareSet(board.pieces(chess.PAWN, chess.BLACK)))) == 5


def white_has_white_bishop(board):
    return any(chess.SquareSet(chess.BB_LIGHT_SQUARES) & chess.SquareSet(board.pieces(chess.BISHOP, chess.WHITE)))

def white_has_black_bishop(board):
    return any(chess.SquareSet(chess.BB_DARK_SQUARES) & chess.SquareSet(board.pieces(chess.BISHOP, chess.WHITE)))

def black_has_white_bishop(board):
    return any(chess.SquareSet(chess.BB_LIGHT_SQUARES) & chess.SquareSet(board.pieces(chess.BISHOP, chess.BLACK)))

def black_has_black_bishop(board):
    return any(chess.SquareSet(chess.BB_DARK_SQUARES) & chess.SquareSet(board.pieces(chess.BISHOP, chess.BLACK)))




if __name__ == '__main__':


    train_setting = 8
    dataloader = make_fen_batch_provider('../PycharmProjects/data/training_data.binpack', 1000, train_setting)

    fen_batch = next(dataloader)
    for fen in fen_batch:
        board = chess.Board(fen)
        #print(board)
        # assert 
        print(len(chess.SquareSet(chess.BB_LIGHT_SQUARES) & 
                    (chess.SquareSet(board.pieces(chess.PAWN, chess.WHITE)) | 
                     chess.SquareSet(board.pieces(chess.PAWN, chess.BLACK)))))
        
        #print((chess.SquareSet(chess.BB_LIGHT_SQUARES) & 
        #            (chess.SquareSet(board.pieces(chess.PAWN, chess.WHITE)) | 
        #             chess.SquareSet(board.pieces(chess.PAWN, chess.BLACK)))))
        assert not len(chess.SquareSet(chess.BB_LIGHT_SQUARES) & 
                    (chess.SquareSet(board.pieces(chess.PAWN, chess.WHITE)) | 
                     chess.SquareSet(board.pieces(chess.PAWN, chess.BLACK)))) == 5
        



    if 0:
        dataloader = make_fen_batch_provider('../PycharmProjects/data/training_data.binpack', 1000, 7)

        fen_batch = next(dataloader)
        for fen in fen_batch[:100]:
            board = chess.Board(fen)
            print('-------------------')


            print(fen)
            print(board)
            print('white_has_white_bishop', white_has_white_bishop(board))
            print('white_has_black_bishop', white_has_black_bishop(board))
            print('black_has_white_bishop', black_has_white_bishop(board))
            print('black_has_black_bishop', black_has_black_bishop(board))


            #assert not (not white_has_white_bishop(board) and  not black_has_white_bishop(board) and white_has_black_bishop(board) and black_has_black_bishop(board) )
            #assert not (white_has_white_bishop(board) and black_has_white_bishop(board) and not white_has_black_bishop(board) and not black_has_black_bishop(board) )
            
            assert not (not white_has_white_bishop(board) and  black_has_white_bishop(board) and white_has_black_bishop(board) and not black_has_black_bishop(board) )
            assert not (white_has_white_bishop(board) and not black_has_white_bishop(board) and not white_has_black_bishop(board) and black_has_black_bishop(board) )
            