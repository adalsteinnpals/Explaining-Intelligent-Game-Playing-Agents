from collections import defaultdict
from copy import copy
import operator

from .CSP import *
from .Sudoku_variables import *
from .constraint_helpers import *

def Backtrack(assignment: dict, csp: CSP):
  if len(assignment.keys()) == len(csp.get_nodes()):
    print("found assignment")
    print(assignment)
    return "success"

  curr_var = csp.get_first_unassigned()
  for v in curr_var.get_domain():
    assignment[curr_var.get_name()] = v
    curr_var.set_value(v)
    if csp.is_possible():
      result = Backtrack(assignment,csp)
      if result == "success":
        return result
    del assignment[curr_var.get_name()]
    curr_var.unset_value()
  return "failure"


def BacktrackingSearch(csp : CSP):

  assignment = {}
  for v in csp.get_nodes():
    if v.status:
      assignment[v.get_name()] = v.value

  out = Backtrack(assignment, csp)
  if out == "success":
    print("successfully found assignment")
  else:
    print("no assignment found")





if __name__ == '__main__':
    variables = {}
    domain = [1,2,3]
    variables['WA'] = Variable('WA', domain,False)
    variables['NT'] = Variable('NT', domain,False)
    variables['Q'] = Variable('Q', domain,False)
    variables['NSW'] = Variable('NSW', domain,False)
    variables['V'] = Variable('V', domain,False)
    variables['SA'] = Variable('SA', domain,False)
    variables['T'] = Variable('T', domain,False)

    vars = list(variables.values())

    constraints = []
    constraints += non_directional(variables['SA'], variables['WA'], 'neq')
    constraints += non_directional(variables['SA'], variables['NT'], 'neq')
    constraints += non_directional(variables['SA'], variables['Q'], 'neq')
    constraints += non_directional(variables['SA'], variables['NSW'], 'neq')
    constraints += non_directional(variables['SA'], variables['V'], 'neq')
    constraints += non_directional(variables['WA'], variables['NT'], 'neq')
    constraints += non_directional(variables['NT'], variables['Q'], 'neq')
    constraints += non_directional(variables['Q'], variables['NSW'], 'neq')
    constraints += non_directional(variables['NSW'], variables['V'], 'neq')
    Constraint(variables['NT'], 2, 'eq')
    # Constraint(variables['NSW'], 1, 'eq')


    CSP1 = CSP(vars, constraints)
    BacktrackingSearch(CSP1)

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

    vars = []
    for r in range(9):
      vars += S.get_row(r)

    CSP2 = CSP(vars, constraints)
    BacktrackingSearch(CSP2)
    for n in CSP2.get_nodes():
      print(n.get_domain())
    S.print_board()