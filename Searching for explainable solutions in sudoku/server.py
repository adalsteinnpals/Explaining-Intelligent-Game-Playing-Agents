from flask import Flask, render_template, url_for, send_from_directory, request, jsonify
from flask_cors import CORS

from CSP import *
from CSP.variable import Variable
from CSP.constraint import Constraint
from CSP.CSP import CSP
from CSP.Sudoku_variables import Sudoku_Vars
from CSP.constraint_helpers import all_different, non_directional
from CSP.btsearch_sudoku import BTSearchSudoku
from CSP.sudoku_search import SudokuSearch

import json

app = Flask(__name__)
CORS(app)

"""
@app.route('/')
def home():
    return render_template('home.html')
"""


@app.route('/')
def sudoku():
    return render_template('sudoku.html')


@app.route('/sudoku_solve', methods=['POST'])
def sudoku_solve():
    board = {i['name']: i['value'] for i in request.json}

    vars = []

    cells = [
        ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9'],
        ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9'],
        ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'],
        ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9'],
        ['e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8', 'e9'],
        ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9'],
        ['g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8', 'g9'],
        ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9'],
        ['i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9'],
    ]

    sudoku_board = [[0] * 9 for _ in range(9)]
    for idx, row in enumerate(cells):
        for idc, c in enumerate(row):
            if board[cells[idx][idc]] != '':
                sudoku_board[idx][idc] = int(board[cells[idx][idc]])

    S = Sudoku_Vars(sudoku_board)

    constraints = []
    constraints += all_different(S.get_box(0, 0))
    constraints += all_different(S.get_box(0, 1))
    constraints += all_different(S.get_box(0, 2))
    constraints += all_different(S.get_box(1, 0))
    constraints += all_different(S.get_box(1, 1))
    constraints += all_different(S.get_box(1, 2))
    constraints += all_different(S.get_box(2, 0))
    constraints += all_different(S.get_box(2, 1))
    constraints += all_different(S.get_box(2, 2))
    constraints += all_different(S.get_col(0))
    constraints += all_different(S.get_col(1))
    constraints += all_different(S.get_col(2))
    constraints += all_different(S.get_col(3))
    constraints += all_different(S.get_col(4))
    constraints += all_different(S.get_col(5))
    constraints += all_different(S.get_col(6))
    constraints += all_different(S.get_col(7))
    constraints += all_different(S.get_col(8))
    constraints += all_different(S.get_row(0))
    constraints += all_different(S.get_row(1))
    constraints += all_different(S.get_row(2))
    constraints += all_different(S.get_row(3))
    constraints += all_different(S.get_row(4))
    constraints += all_different(S.get_row(5))
    constraints += all_different(S.get_row(6))
    constraints += all_different(S.get_row(7))
    constraints += all_different(S.get_row(8))

    vars = []
    for r in range(9):
        vars += S.get_row(r)

    csp = CSP(vars, constraints)
    success, steps = SudokuSearch(csp, S)

    board.clear()

    for n in csp.get_nodes():
        board["#"+n.get_name().lower()] = n.value
    print(board)
    data = {
        "success": success,
        "board": board,
        "steps": steps
    }
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)
