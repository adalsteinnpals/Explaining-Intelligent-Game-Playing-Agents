# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 13:35:19 2019

@author: adals
"""

from CSP import CSP, Variable
from collections import defaultdict, Counter
import operator
from Sudoku_variables import Sudoku_Vars
from constraint_helpers import all_different,non_directional
from copy import copy
import itertools
import numpy as np




#S = Sudoku_Vars([
#  [4,8,3,9,2,1,6,5,7],
#  [9,0,0,3,0,5,0,0,1],
#  [2,0,1,8,0,6,4,0,3],
#  [5,0,8,1,0,2,9,0,6],
#  [7,0,0,0,0,0,0,0,8],
#  [1,0,6,7,0,8,2,0,5],
#  [3,0,2,6,0,9,5,0,4],
#  [8,0,0,2,0,3,0,0,9],
#  [6,9,5,4,1,7,3,8,2]
#])

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

constraints = []
constraints += all_different(S.get_box(0,0))
constraints += all_different(S.get_box(1,0))
constraints += all_different(S.get_box(2,0))
constraints += all_different(S.get_box(1,0))
constraints += all_different(S.get_box(1,1))
constraints += all_different(S.get_box(1,2))
constraints += all_different(S.get_box(2,0))
constraints += all_different(S.get_box(2,1))
constraints += all_different(S.get_box(2,2))
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

def revise(Xi: Variable, Xj: Variable, op: operator) -> bool:
    revised = False
    to_remove = []
    for x in Xi.get_domain():
        if all([not op(x,y) for y in Xj.get_domain()]):
            to_remove.append(x)
            revised = True
    if revised:
        Xi.bulk_remove_from_domain(to_remove)
    return revised

def AC3(csp : CSP):

    arc_queue = csp.get_arcs()
    while len(arc_queue) > 0:
        xi,xj,op = arc_queue.pop(0)
        if revise(xi,xj,op):
            if len(xi.get_domain()) == 0:
                print('Domain empty...')
                return csp
            for xk,nop in csp.get_neighbours_of_v(xi):
                if xk == xj: continue
                arc_queue.append((xk,xi,nop))
    return csp




def is_full(csp):
    return False if csp.get_first_unassigned() else True



def set_first_sole_candidate(csp: CSP):
    if not csp.get_first_unassigned():
        print("All assigned...")

    for node in csp.get_nodes(): 
        if len(node.domain) == 1:
            if not node.status:
                node.set_value(node.domain[0])
                return True
    return False
   
    
constraints_dict = {'get_box' : [(0,0), (0,1), (0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)],
                    'get_col' : list(range(9)),
                    'get_row' : list(range(9))}


def set_first_unique_candidate(csp: CSP):
    for fun, val_list in constraints_dict.items():
        for val in val_list:
            call_fun = getattr(S,fun)
            if type(val) == tuple:
                variables_in_constraint = call_fun(val[0], val[1])
            else:
                variables_in_constraint = call_fun(val)
            un_set_variables = []
            for var in variables_in_constraint:
                if not var.status:
                    un_set_variables.append(var)
            
            value_count = dict(Counter(itertools.chain.from_iterable([var.domain for var in un_set_variables])))
            for possible_value, count in value_count.items():
                if count == 1:
                    variable_to_set = [var for var in un_set_variables if possible_value in var.domain][0]
                    
                    variable_to_set.set_value(possible_value)
                    return True
    return False
                    
                
                
                
    

def sole_candidate_method(csp : CSP):
    success = True
    csp_full = False
    while not csp_full and success:
        csp_full = is_full(csp)
        csp = AC3(csp)
        success = set_first_sole_candidate(csp)
    return csp

    

def sole_or_unique_candidate_method(csp : CSP):
    success = True
    csp_full = False
    num_sole = 0
    num_unique = 0
    csp = AC3(csp)
    while not csp_full and success:
        csp_full = is_full(csp)
        success_sole = set_first_sole_candidate(csp)
        if success_sole: num_sole += 1
        if not success_sole:
            success_unique = set_first_unique_candidate(csp)
            
            if success_unique: num_unique += 1
            if not success_unique:
                success= False
        
        csp = AC3(csp)
#        S.print_board()
    print('Sole candidate: {}'.format(num_sole))
    print('Unique candidate: {}'.format(num_unique))
    return csp



def random_sole_or_unique_candidate_method(csp : CSP):
    success = True
    csp_full = False
    num_sole = 0
    num_unique = 0
    csp = AC3(csp)
    while not csp_full and success:
        csp_full = is_full(csp)
        rand = np.random.rand(1)[0]
        if rand > 0.5:
            success_sole = set_first_sole_candidate(csp)
            if success_sole: num_sole += 1
            if not success_sole:
                success_unique = set_first_unique_candidate(csp)
                
                if success_unique: num_unique += 1
                if not success_unique:
                    success= False
        else:
            success_unique = set_first_unique_candidate(csp)
            if success_unique: num_unique += 1
            if not success_unique:
                success_sole = set_first_sole_candidate(csp)
                if success_sole: num_sole += 1
                if not success_sole:
                    success= False
            
        
        csp = AC3(csp)
    print('Sole candidate: {}'.format(num_sole))
    print('Unique candidate: {}'.format(num_unique))
    return csp






#%%
vars = []
for r in range(9):
  vars += S.get_row(r)
CSP1 = CSP(vars, constraints)
#csp = sole_candidate_method(CSP1)
#csp = sole_or_unique_candidate_method(CSP1)
csp = random_sole_or_unique_candidate_method(CSP1)


S.print_board()
#%%
    





















