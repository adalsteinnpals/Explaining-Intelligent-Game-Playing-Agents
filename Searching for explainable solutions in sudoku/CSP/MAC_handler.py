import operator

from .CSP import CSP, Variable


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
                if xk == xj or xk.status:
                    continue
                arc_queue.append((xk, xi, nop))

    return True, revised_list
