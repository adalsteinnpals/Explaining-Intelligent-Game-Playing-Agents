import operator
from collections import defaultdict
from .variable import Variable

class Constraint:
    ops = {
        'neq': operator.ne,
        'eq': operator.eq
    }

    def __init__(self, left:Variable, right , relation: str):
        if type(right) != Variable:
            if relation == 'eq':
                left.set_value(right)
            elif relation == 'neq':
                left.remove_from_domain(right)

        else:
            self.left = left
            self.right = right
            self.operation = self.ops[relation]
            self.relation = relation

    def is_satisfied(self) -> bool:
        return self.operation(self.left, self.right)

    def get_op(self) -> operator:
        return self.operation

    def __repr__(self):
        return "Constraint({},{},{})".format(repr(self.left), repr(self.right), self.operation)

    def __str__(self):
        return "Constraint: {} {} {}".format(self.left.get_name(), self.relation, self.right.get_name())
