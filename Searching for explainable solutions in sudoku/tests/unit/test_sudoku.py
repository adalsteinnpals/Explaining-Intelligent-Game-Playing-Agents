import pytest
"""


We should definately fill this out asap


from import Sudoku

test_board = [
    [1,2,3,4,5,6,7,8,9],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0]
]

test_board_2 = [
    [1,7,3,4,5,6,2,8,9],
    [2,4,5,0,0,0,0,0,0],
    [3,8,9,0,0,0,0,0,0],
    [4,0,0,1,2,3,1,6,7],
    [5,0,0,6,5,4,2,5,8],
    [6,0,0,7,8,9,3,4,9],
    [7,0,0,0,0,0,0,0,0],
    [8,0,0,0,0,0,0,0,0],
    [9,0,0,0,0,0,0,0,0]
]

def test_constructor():
    s = Sudoku(list(test_board))
    assert s.errors == False
    assert s.board == test_board


def test_get_row():
    s = Sudoku(list(test_board))
    first_row = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    second_row = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert s.get_row(0) == first_row
    assert s.get_row(1) == second_row


def test_get_col():
    s = Sudoku(list(test_board))
    first_col = [1, 0, 0, 0, 0, 0, 0, 0, 0]
    second_col = [2, 0, 0, 0, 0, 0, 0, 0, 0]
    assert s.get_col(0) == first_col
    assert s.get_col(1) == second_col

def test_get_box():
    s = Sudoku(list(test_board_2))
    first_box = [1,7,3,2,4,5,3,8,9]
    middle_box = [1, 2, 3, 6, 5, 4, 7, 8, 9]
    middle_right_box = [1, 6, 7, 2, 5, 8, 3, 4, 9]
    assert s.get_box(0,0) == first_box
    assert s.get_box(1,1) == middle_box
    assert s.get_box(2,1) == middle_right_box

def test_set_unit():
    s = Sudoku(list(test_board))
    assert not s.isset_unit(0,1)
    s.set_unit(0,1,9)
    assert s.isset_unit(0,1)

def test_unset_unit():
    s = Sudoku(list(test_board))
    print(test_board)
    assert not s.isset_unit(0,1)
    s.set_unit(0,1,9)
    assert s.isset_unit(0,1)
    s.unset_unit(0,1)
    assert not s.isset_unit(0,1)
"""
