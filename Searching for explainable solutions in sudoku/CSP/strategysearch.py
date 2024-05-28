
from collections import defaultdict
from copy import copy, deepcopy
import operator
from itertools import combinations, groupby

from .Sudoku_variables import Sudoku_Vars
from .constraint_helpers import *
from .CSP import *

from .MAC_handler import revise, MAC

import string

def get_sole_candidate(vars: list):
    curr_vars = []
    for var in vars:
        if not var.status and len(var.get_domain()) == 1:
            curr_vars.append((var.get_domain()[0], var, "sole"))
    return curr_vars


def get_unique_candidate(vars: list):
    value_counter = [0] * 9
    value_pointer = [None] * 9
    for var in vars:
        if var.status:
            continue
        for di in var.get_domain():
            value_counter[di-1] += 1
            value_pointer[di-1] = var
    curr_vars = []
    for idx, count in enumerate(value_counter):
        if count == 1:
            curr_vars.append((idx+1, value_pointer[idx], "unique"))

    return curr_vars

def get_macro_moves_sole(csp: CSP, s: Sudoku_Vars) -> list:
    sole_vars = get_sole_candidate(csp.get_nodes())
    return sole_vars

def get_macro_moves_unique(csp: CSP, s: Sudoku_Vars) -> list:
    unique_vars = []
    for n in range(9):
        unique_vars.extend(get_unique_candidate(s.get_row(n)))
        unique_vars.extend(get_unique_candidate(s.get_col(n)))
    for x in range(3):
        for y in range(3):
            unique_vars.extend(get_unique_candidate(s.get_box(x, y)))
    already_seen = set()
    pruned_unique_vars = []
    for i in unique_vars:
        if (i[0],i[1]) not in already_seen:
            already_seen.add((i[0], i[1]))
            pruned_unique_vars.append(i)
    return pruned_unique_vars


char_to_num = dict(zip(string.ascii_uppercase, range(26)))

def var_to_colum_row(varname):
    return (char_to_num[varname[0]], int(varname[1])-1)


def get_one_directional_unique(s: Sudoku_Vars, vars: list, axis = 0):
    move_name = 'one_dim_unique_candidate' #'h_unique' if axis == 0 else 'v_unique'
    init_domains = []
    for var in vars:
        if var.status:
            init_domains.append(var.get_domain()[:])
        else:
            init_domains.append(var.original_domain[:])

    for idx, var in enumerate(vars):
        if not var.status:
            row, col = var_to_colum_row(var.name)
            if axis == 0:
                vars_in_dim = s.get_row(row)
            elif axis == 1:
                vars_in_dim = s.get_col(col)
            for var_in_dim in vars_in_dim:
                if var_in_dim.status:
                    if var_in_dim.value in init_domains[idx]:
                        init_domains[idx].remove(var_in_dim.value)

    value_counter = [0] * 9
    value_pointer = [None] * 9
    for idx, var in enumerate(vars):
        for di in init_domains[idx]:
            value_counter[di-1] += 1
            value_pointer[di-1] = var
    curr_vars = []
    for idx, count in enumerate(value_counter):
        if count == 1:
            if not value_pointer[idx].status:
#                if len(value_pointer[idx].domain) != 1:
                curr_vars.append((idx+1, value_pointer[idx], move_name))

    return curr_vars


def get_naked_double_moves(vars: list):
    """
        returns all *sole candidates* move that
        become available if naked double pruning
        is used

        Logic: find all pairs of domain with that
        are the same and of length 2 (could be n)
        try removing those two from all other domains
        if after removing them from those domains
        and they have just a single value left in
        the domain, that means we found a sole
        candidate. (we could do the same for
        unique candidates, this is just a *tribute*)
    """
    moves = []
    domain_sets = [[set(v.get_domain()),v] for v in vars]
    set_pairs = combinations(domain_sets, 2)
    for p1,p2 in set_pairs:
        if p1[0] == p2[0] and len(p1[0]) == 2:
            domain_sets_copy = domain_sets.copy()
            for d in domain_sets_copy:
                if d[0] == p1[0] or d[1].status:
                    continue
                d_curr = d[0].copy() - p1[0]
                if len(d_curr) == 1:
                    moves.append((d_curr.pop(), d[1], "naked double"))
    return moves

def get_macro_moves_sole(csp: CSP, s: Sudoku_Vars) -> list:
    sole_vars = get_sole_candidate(csp.get_nodes())
    return sole_vars

def get_macro_moves_unique(csp: CSP, s: Sudoku_Vars) -> list:
    unique_vars = []
    for n in range(9):
        unique_vars.extend(get_unique_candidate(s.get_row(n)))
        unique_vars.extend(get_unique_candidate(s.get_col(n)))
    for x in range(3):
        for y in range(3):
            unique_vars.extend(get_unique_candidate(s.get_box(x, y)))
    already_seen = set()
    pruned_unique_vars = []
    for i in unique_vars:
        if (i[0],i[1]) not in already_seen:
            already_seen.add((i[0], i[1]))
            pruned_unique_vars.append(i)
    return pruned_unique_vars


def get_macro_moves_one_dim_unique(csp: CSP, s: Sudoku_Vars):
    one_dim_unique = []
    for x in range(3):
        for y in range(3):
            curr_vars = get_one_directional_unique(
                s,
                s.get_box(x, y),
                axis = 0)
            one_dim_unique.extend(curr_vars)
            curr_vars = get_one_directional_unique(
                s,
                s.get_box(x, y),
                axis = 1)
            one_dim_unique.extend(curr_vars)

    already_seen = set()
    pruned_one_dim_unique_vars = []
    for i in one_dim_unique:
        if (i[0],i[1]) not in already_seen:
            already_seen.add((i[0], i[1]))
            pruned_one_dim_unique_vars.append(i)
#    print('----', len(pruned_one_dim_unique_vars))
#    p = [print(x) for x in pruned_one_dim_unique_vars]
    return pruned_one_dim_unique_vars

def get_macro_moves_naked_double(csp: CSP, s: Sudoku_Vars) -> list:
    naked_double_vars = []
    for n in range(9):
        naked_double_vars.extend(get_naked_double_moves(s.get_row(n)))
        naked_double_vars.extend(get_naked_double_moves(s.get_col(n)))
    for x in range(3):
        for y in range(3):
            naked_double_vars.extend(get_naked_double_moves(s.get_box(x, y)))
    already_seen = set()
    pruned_naked_double_vars = []
    for i in naked_double_vars:
        if (i[0],i[1]) not in already_seen:
            already_seen.add((i[0], i[1]))
            pruned_naked_double_vars.append(i)
    return pruned_naked_double_vars


def get_actions(csp: CSP, s: Sudoku_Vars) -> list:

    strategy_actions = []

    sole_vars = get_sole_candidate(csp.get_nodes())

    if sole_vars:
      strategy_actions.append("sole_candidate")

    unique_vars = []
    for n in range(9):
        if unique_vars: break
        unique_vars.extend(get_unique_candidate(s.get_row(n)))
        unique_vars.extend(get_unique_candidate(s.get_col(n)))
    for x in range(3):
        for y in range(3):
            if unique_vars: break
            unique_vars.extend(get_unique_candidate(s.get_box(x, y)))

    if unique_vars:
        strategy_actions.append("unique_candidate")


    one_dim_unique = []
    for x in range(3):
        for y in range(3):
            if one_dim_unique: break
            curr_vars = get_one_directional_unique(
                s,
                s.get_box(x, y),
                axis = 0)
            one_dim_unique.extend(curr_vars)
            curr_vars = get_one_directional_unique(
                s,
                s.get_box(x, y),
                axis = 1)
            one_dim_unique.extend(curr_vars)


    if one_dim_unique:
        strategy_actions.append("one_dim_unique_candidate")


    naked_double_vars = []
    for n in range(9):
        if naked_double_vars: break
        naked_double_vars.extend(get_naked_double_moves(s.get_row(n)))
        naked_double_vars.extend(get_naked_double_moves(s.get_col(n)))
    for x in range(3):
        for y in range(3):
            if naked_double_vars: break
            naked_double_vars.extend(get_naked_double_moves(s.get_box(x, y)))

    if naked_double_vars:
        strategy_actions.append("naked_double_candidate")

    return strategy_actions



def get_all_actions(csp: CSP, s: Sudoku_Vars) -> list:

    strategy_actions = []

    sole_vars = get_sole_candidate(csp.get_nodes())
    if sole_vars:
        already_seen = set()
        pruned_sole_vars = []
        for i in sole_vars:
            if (i[0],i[1]) not in already_seen:
                already_seen.add((i[0], i[1]))
                pruned_sole_vars.append(i)

        strategy_actions.append(("sole_candidate", len(pruned_sole_vars)))

    unique_vars = []
    for n in range(9):
        unique_vars.extend(get_unique_candidate(s.get_row(n)))
        unique_vars.extend(get_unique_candidate(s.get_col(n)))
    for x in range(3):
        for y in range(3):
            unique_vars.extend(get_unique_candidate(s.get_box(x, y)))

    if unique_vars:
        already_seen = set()
        pruned_unique_vars = []
        for i in unique_vars:
            if (i[0],i[1]) not in already_seen:
                already_seen.add((i[0], i[1]))
                pruned_unique_vars.append(i)

        strategy_actions.append(("unique_candidates", len(pruned_unique_vars)))


    one_dim_unique = []
    for x in range(3):
        for y in range(3):
            curr_vars = get_one_directional_unique(
                s,
                s.get_box(x, y),
                axis = 0)
            one_dim_unique.extend(curr_vars)
            curr_vars = get_one_directional_unique(
                s,
                s.get_box(x, y),
                axis = 1)
            one_dim_unique.extend(curr_vars)


    if one_dim_unique:
        already_seen = set()
        pruned_one_dim_unique = []
        for i in one_dim_unique:
            if (i[0],i[1]) not in already_seen:
                already_seen.add((i[0], i[1]))
                pruned_one_dim_unique.append(i)

        strategy_actions.append(("one_dim_candidates", len(pruned_one_dim_unique)))


    naked_double_vars = []
    for n in range(9):
        if naked_double_vars: break
        naked_double_vars.extend(get_naked_double_moves(s.get_row(n)))
        naked_double_vars.extend(get_naked_double_moves(s.get_col(n)))
    for x in range(3):
        for y in range(3):
            if naked_double_vars: break
            naked_double_vars.extend(get_naked_double_moves(s.get_box(x, y)))

    if naked_double_vars:
        already_seen = set()
        pruned_naked_double_vars = []
        for i in naked_double_vars:
            if (i[0],i[1]) not in already_seen:
                already_seen.add((i[0], i[1]))
                pruned_naked_double_vars.append(i)

        strategy_actions.append(("naked_double", len(pruned_naked_double_vars)))


#    s.print_board()
#    print(pruned_unique_vars)
#    print('---')
#    print(pruned_sole_vars)
#    print('muuuuuu')
#    print(pruned_one_dim_unique)
#    print('')

    return strategy_actions


cost_dict = {
    "sole_candidate": 3,
    "unique_candidate": 5,
    "one_dim_unique_candidate": 1,
    "naked_double_candidate": 10,
    }

strategy_map = {
  "sole_candidate": get_macro_moves_sole,
  "unique_candidate": get_macro_moves_unique,
  "one_dim_unique_candidate": get_macro_moves_one_dim_unique,
  "naked_double_candidate": get_macro_moves_naked_double,
}

transposition_table = {}

do_batching = True
max_macro_moves = 1

def apply_macro_move(curr_val, curr_var: Variable, assignment: dict):
    assignment[curr_var.get_name()] = curr_val
    curr_var.set_value(curr_val)

def apply_action(strategy, csp: CSP, assignment: dict, s: Sudoku_Vars) -> list:
    """
      in strategy search when applying an action we exhaust our options for using a single method.
    """
    global strategy_map, cost_dict, max_macro_moves
    total_moves = []
    total_inferred = []

    DECAY_FACTOR = 0.95

    moves = strategy_map[strategy](csp,s)
    diffculty_scalar = (1+(1-(len(moves)/(81-len(assignment.keys())))))**2

    if do_batching:
        moves = moves[:max_macro_moves]

    for move in moves:
        apply_macro_move(move[0], move[1], assignment)

        total_moves.append(move[1])

        MAC_queue = [(ni[0], move[1], ni[1]) for ni in csp.get_neighbours_of_v(move[1])]
        _, inferred = MAC(MAC_queue, csp)
        total_inferred.extend(inferred)

    total_cost = cost_dict[strategy] * len(total_moves) * sum([DECAY_FACTOR**p for p in range(len(total_moves))])
    total_cost *= diffculty_scalar


    return (total_cost, total_moves, total_inferred)

def undo_action(vars: list, inferred_list: list, assignment: dict):
    for i in inferred_list:
        i[0].add_to_domain(i[1])
    for var in vars:
        del assignment[var.get_name()]
        var.unset_value()

prune_hit = 0
tt_hit = 0
end_hit = 0
max_depth = 0
end_costs = []
end_paths = []
done_acts = []
pruned_costs = []
func_hit = 0
best_end_cost = float("inf")

def Search(assignment: dict, csp: CSP, s: Sudoku_Vars, cumul_cost=0, best_cost=float("inf"), depth=0, done_act=set()) -> str:
    global prune_hit, tt_hit, end_hit,func_hit,best_end_cost,end_costs,pruned_costs,end_paths, max_depth, practicallyINF, done_acts
    func_hit += 1

    if cumul_cost > best_cost :
        prune_hit += 1
        return (float("inf"),[],[])

    if len(assignment.keys()) == len(csp.get_nodes()):
        end_hit += 1
        end_costs.append(cumul_cost)
        done_acts.append(list(done_act))
        return (0,[],[])

    global transposition_table
    global cost_dict


    assignment_hash = ';'.join(["{}:{}".format(k, v) for k, v in sorted(assignment.items())])
    if assignment_hash in transposition_table:
        tt_hit += 1
        val = transposition_table[assignment_hash]
        end_costs.append(val[0]+cumul_cost)
        done_acts.append(list(done_act))
        return val

    move_list = get_actions(csp, s)
    available_moves_list = get_all_actions(csp, s)

    best_move = None
    best_subtree_path = []
    best_subtree_available_move_list = []
    added_move = False

    if not move_list:
        return (float("inf"),[],[])

    for move in move_list:
        added_move = False
        curr_cost = 0

        (application_cost, applied_actions, total_inferred) = apply_action(move, csp, assignment, s)
        curr_cost += application_cost

        if move not in done_act:
            curr_cost += 10 * cost_dict[move]
            done_act.add(move)
            added_move = True

        cost_for_move, subtree_path, subtree_avilable_moves_list = Search(assignment, csp, s, cumul_cost + curr_cost, best_cost, depth+1, done_act)

        if added_move:
            done_act.remove(move)

        curr_cost += cost_for_move

        if curr_cost < best_cost:
            best_cost = curr_cost
            best_move = f"{move}( {len(applied_actions)} )"
            best_subtree_path = subtree_path[:]
            best_subtree_available_move_list = subtree_avilable_moves_list[:]

        undo_action(applied_actions, total_inferred, assignment)

    # if not best_move:
        # return (float("inf"), [], [])

    transposition_table[assignment_hash] = (best_cost,best_subtree_path, best_subtree_available_move_list)
    best_subtree_path.insert(0, best_move)
    best_subtree_available_move_list.insert(0, available_moves_list)
    return (best_cost,best_subtree_path, best_subtree_available_move_list)


def SudokuSearch(csp: CSP, s: Sudoku_Vars):

    global prune_hit, tt_hit, end_hit,func_hit,best_end_cost,end_costs,pruned_costs,end_paths,max_depth,done_acts, transposition_table
    transposition_table.clear()
    prune_hit = 0
    tt_hit = 0
    end_hit = 0
    max_depth = 0
    end_costs = []
    done_acts = []
    end_paths = []
    pruned_costs = []
    func_hit = 0
    best_end_cost = float("inf")
    transposition_table.clear()


    total_queue = csp.get_arcs()
    MAC(total_queue, csp)

    assignment = {}
    for v in csp.get_nodes():
        if v.status:
            assignment[v.get_name()] = v.value
    steps = []

    out,path,available_moves = Search(assignment, csp, s, 0)

    print("-------------------------------")
    print("the cost was ", out)
    print()
    print("with path ",path)
    print()
    print("with available moves",available_moves)
    print("-------------------------------")
    print()
    return out, end_costs, available_moves,func_hit,path, done_acts
