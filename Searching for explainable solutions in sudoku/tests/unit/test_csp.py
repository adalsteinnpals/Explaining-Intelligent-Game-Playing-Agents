import pytest
from CSP.CSP import CSP


"""


We should definately fill this out asap




def test_constraints_eq():
    V1 = CSP.Variable('A1', [1], True, 1)
    V2 = CSP.Variable('A2', [1], True, 1)

    constraint = CSP.Constraint(V1, V2, 'eq')
    assert constraint.is_satisfied()


def test_constraints_neq():
    V1 = CSP.Variable('A1', [1], True, 1)
    V2 = CSP.Variable('A2', [1], True, 2)

    constraint = CSP.Constraint(V1, V2, 'neq')
    assert constraint.is_satisfied()


def test_constraints_one_unset():
    V1 = CSP.Variable('A1', [1], False, 0)
    V2 = CSP.Variable('A2', [1], True, 1)

    constraint = CSP.Constraint(V1, V2, 'eq')
    assert constraint.is_satisfied()


def test_CSP_constructor():
    V1 = CSP.Variable('A1', [1, 2], False, -1)
    V2 = CSP.Variable('A2', [2, 3], False, -1)
    C1 = CSP.Constraint(V1, V2, 'neq')
    C2 = CSP.Constraint(V1, 2, 'eq')
    CSP1 = CSP.CSP([V1, V2], [C1, C2])
    assert CSP1.get_neighbours() == {'A1': ['A2'], 'A2': ['A1']}


def test_CSP_getarcs():
    V1 = CSP.Variable('A1', [1, 2], False, -1)
    V2 = CSP.Variable('A2', [2, 3], False, -1)
    C1 = CSP.Constraint(V1, V2, 'neq')
    CSP1 = CSP.CSP([V1, V2], [C1])
    assert CSP1.get_arcs() == [(V1, V2), (V2, V1)]
    C2 = CSP.Constraint(V1, 2, 'eq')
    CSP1 = CSP.CSP([V1, V2], [C1, C2])
    print(CSP1.get_arcs())
    assert CSP1.get_arcs() == [(V1, V2), (V2, V1), (V1, 2)]
"""
