from itertools import combinations
import operator
from collections import defaultdict
from .variable import Variable
from .constraint import Constraint


class CSP:
    def __init__(self, variables: list, constraints: list):
        if any([type(var) != Variable for var in variables]):
            raise "type in variables list is not variable"
        if any([type(con) != Constraint for con in constraints]):
            raise "type in constraint list is not constraint"
        self.variables = variables
        self.constraints = constraints
        # now we create the connection list : A1 -> A2 iff (A1, A2) in constraints
        self.neighbours = defaultdict(list)
        for constraint in constraints:
            self.neighbours[constraint.left.get_name()].append((constraint.right,constraint.operation))

    def is_possible(self) -> bool:
        """
        print("========================")
        print("testing is possible")
        for c in self.constraints:
            print("{}[{}] {} {}[{}] = {}".format(c.left.get_name(), c.left.value if c.right.status else "unset", c.relation,c.right.get_name(), c.right.value if c.right.status else "unset", c.is_satisfied()))
        print("========================")
        """
        return not any([not c.is_satisfied() for c in self.constraints])

    def get_arcs(self) -> list:
        return [(c.left,c.right,c.operation) for c in self.constraints]

    def get_neighbours(self) -> defaultdict:
        return self.neighbours

    def get_neighbours_of_v(self,v:Variable) -> defaultdict:
        return self.neighbours[v.get_name()]

    def get_nodes(self) -> list:
        return self.variables

    def get_constraints(self) -> list:
        return self.constraints

    def get_first_unassigned(self) -> Variable:
        for v in self.variables:
            if not v.status:
                return v
