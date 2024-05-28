
from CSP import *
from CSP.variable import Variable
from CSP.constraint import Constraint
from CSP.CSP import CSP
from CSP.Sudoku_variables import Sudoku_Vars
from CSP.constraint_helpers import all_different, non_directional
from CSP.strategysearch import SudokuSearch


import pandas as pd
import time
import matplotlib.pyplot as plt

simple = pd.read_csv('simple.csv', index_col = False)
easy = pd.read_csv('easy_50.csv', index_col = False)
inter = pd.read_csv('intermediate_50.csv', index_col = False)

t = pd.concat([simple, easy, inter], ignore_index=True)
t = t[['Puzzle', 'Difficulty']]
t["Score"] = 0
t["Time"] = 0
print(t)
count = 0

for idx, row in t[1:10].iterrows():
    puzzle_ = []
    for i in range(9):
        puzzle_.append(list(map(int, row.get('Puzzle')[i*9:(i+1)*9].replace('.','0'))))

    S = Sudoku_Vars(puzzle_)

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

    t0 = time.time()
    _, cost, available_moves = SudokuSearch(csp,S)
    print(f"time: {time.time() - t0}")

#    t.loc[idx,"Score"] = cost
#    t.loc[idx,"Time"] = time.time() - t0
    plt.figure()
    df = pd.DataFrame([dict(x) for x in available_moves]).fillna(0)
    plt.plot(df.sole_candidate, label = 'Sole Candidate')
    plt.plot(df.unique_candidates, label = 'Unique Candidate')
    if 'naked_double' in df.columns: plt.plot(df.naked_double, label = 'Naked Double Candidate')
    plt.plot(df.one_dim_candidates, label = 'One Dimensional Unique Candidate')
    plt.legend()
    plt.xlabel('Steps')
    plt.ylabel('Available moves')

t.to_csv(f'out{time.time()}.csv')
#%%