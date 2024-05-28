from collections import defaultdict
import operator
from .Sudoku_variables import *
from .constraint_helpers import *
from .CSP import *

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
        return False
      for xk,nop in csp.get_neighbours_of_v(xi):
        if xk == xj: continue
        arc_queue.append((xk,xi,nop))
  return True




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
    Constraint(variables['NSW'], 1, 'eq')
    Constraint(variables['Q'], 3, 'eq')



    CSP1 = CSP(vars, constraints)
    out = AC3(CSP1)
    print("There exists a solution!" if out else "there doesn't exist a solution!")
    if out:
      for v in CSP1.get_nodes():
        print(v)
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

    print(S.get_box(0,0))

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
    print(len(CSP2.get_arcs()))
    out = AC3(CSP2)
    for v in CSP2.get_nodes():
      print(v.get_name(), v.get_domain())
    print("There exists a solution!" if out else "there doesn't exist a solution!")

