from collections import defaultdict
from copy import copy
import operator

from .Sudoku_variables import *
from .constraint_helpers import *
from .CSP import *


def revise(Xi: Variable, Xj: Variable, op: operator) -> bool:
    revised = False
    to_remove = []

    for x in Xi.get_domain():
        if not any([op(x, y) for y in Xj.get_domain()]):
            to_remove.append((Xi, x))
            revised = True

    if revised:
        Xi.bulk_remove_from_domain([i[1] for i in to_remove])

    return revised, to_remove


def MAC(initial_queue: list, csp: CSP) -> list:
    arc_queue = initial_queue
    revised_list = []

    while len(arc_queue) > 0:
        xi, xj, op = arc_queue.pop(0)
        revd, rlist = revise(xi, xj, op)

        if rlist:
            revised_list.extend(rlist)

        if revd:
            if len(xi.get_domain()) == 0:
                return False, revised_list

            for xk, nop in csp.get_neighbours_of_v(xi):
                if xk == xj:
                    continue
                arc_queue.append((xk, xi, nop))

    return True, revised_list


def get_sole_candidate(vars: list):
    found_sole_candidate = False
    curr_var = None
    curr_val = None

    for var in vars:
        if not var.status and len(var.get_domain()) == 1:
            curr_var = var
            curr_val = var.get_domain()[0]
            found_sole_candidate = True
            break
    return found_sole_candidate, curr_var, curr_val


def get_unique_candidate(vars: list):
    # print("=================== IN UNIQUE CANDIDATE =======================")
    # print([x.get_name() for x in vars])
    # print("===============================================================")
    found_unique_candidate = False

    value_counter = [0] * 9
    value_pointer = [None] * 9

    for var in vars:
        if var.status:
            continue
        for di in var.get_domain():
            value_counter[di-1] += 1
            value_pointer[di-1] = var

    curr_val = None
    curr_var = None

    for idx, count in enumerate(value_counter):
        if count == 1:
            curr_val = idx+1
            curr_var = value_pointer[idx]
            found_unique_candidate = True
            break

    return found_unique_candidate, curr_var, curr_val


def set_bt_assignment(curr_val, curr_var: Variable, assignment: dict, csp: CSP, steps: list, s: Sudoku_Vars, action: str):
    # print("found {} {}:{}".format(
        # action, curr_var.get_name(), curr_val))

    # s.print_board()

    assignment[curr_var.get_name()] = curr_val
    curr_var.set_value(curr_val)

    steps.append("[SET] {}:{}".format(curr_var.get_name(), curr_val))
    var_queue = [(ni[0], curr_var, ni[1])
                 for ni in csp.get_neighbours_of_v(curr_var)]

    mac_status, changed_list = MAC(var_queue, csp)
    # print("output from MAC", mac_status)
    # if not mac_status:
    # for n in csp.get_nodes():
    # print(n.get_name(), n.get_domain())
    # print("START changedlist====")
    # [print(x[0].get_name(), x[1]) for x in changed_list]
    # print("DONE changedlist====")
    # print("csp ispossible ", csp.is_possible())

    if csp.is_possible() and mac_status:
        result = BTMACSudoku(assignment, csp, steps, s)

        if result == "success":
            return result

    for i in changed_list:
        i[0].add_to_domain(i[1])

    # print("HAD TO UNDO {} {}:{}".format(
    #     action, curr_var.get_name(), curr_val))
    del assignment[curr_var.get_name()]
    curr_var.unset_value()
    steps.append("[UNDO] {}".format(curr_var.get_name()))


def BTMACSudoku(assignment: dict, csp: CSP, steps: list, s: Sudoku_Vars) -> str:
    if len(assignment.keys()) == len(csp.get_nodes()):
        return "success"

    changed_list = []
    curr_var = None
    curr_val = None
    did_nonbt = False
    """
      Lets do solecandidate:
        find a variable with len(domain) = 1
        we know what value it will have
    """

    found, curr_var, curr_val = get_sole_candidate(csp.get_nodes())

    if found:
        result = set_bt_assignment(curr_val, curr_var, assignment,
                                   csp, steps, s, "SOLE CANDIDATE")
        if result == "success":
            return result

    """
      Lets do uniquecandidate:
        find a variable in a group (row, col, box) that alone has a
        possible assignment, then we know what it will be
    """
    # print("==========================CHECKING ROWS ===========================")
    for row in range(9):
        found, curr_var, curr_val = get_unique_candidate(s.get_row(row))
        if found:
            break
    # print("==========================DONE CHECKING ROWS ===========================")

    if found:
        result = set_bt_assignment(curr_val, curr_var, assignment,
                                   csp, steps, s, "UNIQUE CANDIDATE")
        if result == "success":
            return result

    # print("==========================CHECKING COLS ===========================")
    for col in range(9):
        found, curr_var, curr_val = get_unique_candidate(s.get_col(col))
        if found:
            break
    # print("==========================DONE CHECKING COLS ===========================")

    if found:
        result = set_bt_assignment(curr_val, curr_var, assignment,
                                   csp, steps, s, "UNIQUE CANDIDATE")
        if result == "success":
            return result

    # print("==========================CHECKING BOXES ===========================")
    for x in range(3):
        for y in range(3):
            found, curr_var, curr_val = get_unique_candidate(
                s.get_box(x, y))
            if found:
                break
        if found:
            break
    # print("==========================DONE CHECKING BOXES ===========================")

    if found:
        result = set_bt_assignment(curr_val, curr_var, assignment,
                                   csp, steps, s, "UNIQUE CANDIDATE")
        if result == "success":
            return result

    """
      We are now in the backtracking phase. this is bad
    """
    # print("============BACKTRACKING===================="

    curr_var = csp.get_first_unassigned()

    for v in curr_var.get_domain():
        result = set_bt_assignment(v, curr_var, assignment, csp,
                                   steps, s, "BACKTRACK TESTING")
        if result == "success":
            return result

    return "failure"


def BTSearchSudoku(csp: CSP, s: Sudoku_Vars):

    # do AC3 once before maintaining.
    total_queue = csp.get_arcs()
    MAC(total_queue, csp)

    # print("INITIAL DOMAINS============================BEGIN")
    # for n in csp.get_nodes():
    # print("node {} d: {}".format(n.get_name(), n.get_domain()))
    # print("INITIAL DOMAINS============================DONE")

    assignment = {}
    for v in csp.get_nodes():
        if v.status:
            assignment[v.get_name()] = v.value
    steps = []
    out = BTMACSudoku(assignment, csp, steps, s)
    if out == "success":
        return True, steps
    else:
        return False, steps
