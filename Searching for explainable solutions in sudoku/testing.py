from CSP import *
from CSP.variable import Variable
from CSP.constraint import Constraint
from CSP.CSP import CSP
from CSP.Sudoku_variables import Sudoku_Vars
from CSP.constraint_helpers import all_different, non_directional
from CSP.btsearch_sudoku import BTSearchSudoku
from CSP.sudoku_search import SudokuSearch


"""
S = Sudoku_Vars([
  [4,8,3,9,2,1,6,5,7],
  [9,0,0,3,0,5,0,0,1],
  [2,0,1,8,0,6,4,0,3],
  [5,0,8,1,0,2,9,0,6],
  [7,0,0,0,0,0,0,0,8],
  [1,0,6,7,0,8,2,0,5],
  [3,0,2,6,0,9,5,0,4],
  [8,0,0,2,0,3,0,0,9],
  [6,9,5,4,1,7,3,8,2]
])
"""
S = Sudoku_Vars([
  [0,0,3,0,2,0,6,0,0],
  [9,0,0,3,0,5,0,0,1],
  [0,0,1,8,0,6,4,0,0],
  [0,0,8,1,0,2,9,0,0],
  [7,0,0,0,0,0,0,0,8],
  [0,0,6,7,0,8,2,0,0],
  [0,0,2,6,0,9,5,0,0],
  [8,0,0,2,0,3,0,0,9],
  [0,0,5,0,1,0,3,0,0]
])
S = Sudoku_Vars([ # 29 unfilled
  [4,8,3,9,2,1,6,5,7],
  [9,0,0,3,0,5,0,0,1],
  [2,0,1,8,0,6,4,0,3],
  [5,0,8,1,0,2,9,0,6],
  [7,0,0,0,0,0,0,0,8],
  [1,0,6,7,0,8,2,0,5],
  [3,0,2,6,0,9,5,0,4],
  [8,0,0,2,0,3,0,0,9],
  [6,9,5,4,1,7,3,8,2]
])
"""
S = Sudoku_Vars([
  [5,3,4,6,7,8,9,1,2],
  [6,7,2,1,9,5,3,4,8],
  [1,9,8,3,4,2,5,6,7],
  [8,5,9,7,6,1,4,2,3],
  [4,2,6,8,5,3,7,9,1],
  [7,1,3,9,2,4,8,5,6],
  [9,6,1,5,3,7,2,8,4],
  [2,8,7,4,1,9,6,3,5],
  [3,4,5,2,8,6,1,7,9]
])
"""
S = Sudoku_Vars([
  [5,3,0,0,0,0,0,1,0],
  [6,7,0,1,9,5,3,0,8],
  [1,9,0,3,4,2,5,0,7],
  [8,5,0,7,6,1,4,2,0],
  [4,2,0,8,5,3,7,0,1],
  [7,1,3,9,2,4,8,0,6],
  [9,6,1,5,3,7,2,8,0],
  [2,8,7,4,1,9,6,0,5],
  [3,4,5,2,8,6,1,7,9]
])


constraints = []
constraints += all_different(S.get_box(0,0))
constraints += all_different(S.get_box(0,1))
constraints += all_different(S.get_box(0,2))
constraints += all_different(S.get_box(1,0))
constraints += all_different(S.get_box(1,1))
constraints += all_different(S.get_box(1,2))
constraints += all_different(S.get_box(2,0))
constraints += all_different(S.get_box(2,1))
constraints += all_different(S.get_box(2,2))

vars = []
for r in range(9):
  vars += S.get_row(r)
  constraints += all_different(S.get_row(r))
  constraints += all_different(S.get_col(r))


csp = CSP(vars, constraints)

import time
t0 = time.time()
SudokuSearch(csp,S)
print(f"time: {time.time() - t0}")
