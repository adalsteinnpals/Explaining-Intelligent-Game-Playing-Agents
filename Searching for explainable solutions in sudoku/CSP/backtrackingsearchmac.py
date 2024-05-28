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


def BacktrackMAC(assignment: dict, csp: CSP, steps: list) -> str:
    if len(assignment.keys()) == len(csp.get_nodes()):
        return "success"

    changed_list = []
    curr_var = None

    curr_var = csp.get_first_unassigned()

    for v in curr_var.get_domain():
        steps.append("[SET] {}:{}".format(curr_var.get_name(), v))
        changed_list = []
        assignment[curr_var.get_name()] = v
        curr_var.set_value(v)

        var_queue = []
        for n in csp.get_neighbours_of_v(curr_var):
            if not n[0].status:
                var_queue.append((n[0], curr_var, n[1]))

        mac_status, mac_changed_list = MAC(var_queue, csp)

        changed_list = mac_changed_list

        if csp.is_possible() and mac_status:
            result = BacktrackMAC(assignment, csp, steps)

            if result == "success":
                return result

        for i in changed_list:
            i[0].add_to_domain(i[1])

        del assignment[curr_var.get_name()]
        curr_var.unset_value()
        steps.append("[UNDO] {}".format(curr_var.get_name()))
    return "failure"


def BacktrackingSearchMAC(csp: CSP):

    # do AC3 once before maintaining.
    total_queue = csp.get_arcs()
    MAC(total_queue, csp)

    assignment = {}
    for v in csp.get_nodes():
        if v.status:
            assignment[v.get_name()] = v.value
    steps = []
    out = BacktrackMAC(assignment, csp, steps)
    if out == "success":
        return True, steps
    else:
        return False, steps
