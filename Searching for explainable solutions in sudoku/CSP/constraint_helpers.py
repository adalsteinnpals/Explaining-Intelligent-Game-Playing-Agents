from .CSP import *
from itertools import combinations

def non_directional(first: Variable, second: Variable, op: str) -> list:
  return [Constraint(first, second, op), Constraint(second,first, op)]

def all_different(variables: list) -> list:
  constraints = []
  for (l,r) in combinations(variables, 2):
    constraints += non_directional(l,r,'neq')
  return constraints
