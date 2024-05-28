from collections import defaultdict
from copy import copy
import operator

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


char_to_num = dict(zip(string.ascii_uppercase, range(26)))

def var_to_colum_row(varname):
    return (char_to_num[varname[0]], int(varname[1])-1)


def get_one_directional_unique(s: Sudoku_Vars, vars: list, axis = 0):
    
    move_name = 'h_unique' if axis == 0 else 'v_unique'
    
    init_domains = []
    for var in vars:
        init_domains.append(var.original_domain)
    
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
                curr_vars.append((idx+1, value_pointer[idx], move_name))
            
    return curr_vars


def get_actions(csp: CSP, s: Sudoku_Vars, last_action: tuple) -> list:

    sole_vars = get_sole_candidate(csp.get_nodes())
    unique_vars = []
    curr_vars = []

    redo_last = False

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

    if last_action[2] not in ["switch","start"]:
        redo_last = True

    for row in range(9):
        curr_vars = get_unique_candidate(s.get_row(row))
        unique_vars.extend(curr_vars)

    for col in range(9):
        curr_vars = get_unique_candidate(s.get_col(col))
        unique_vars.extend(curr_vars)

    for x in range(3):
        for y in range(3):
            curr_vars = get_unique_candidate(
                s.get_box(x, y))
            unique_vars.extend(curr_vars)
#
    already_seen = set()
    pruned_unique_vars = []
    for i in unique_vars:
        if (i[0],i[1]) not in already_seen:
            already_seen.add((i[0], i[1]))
            pruned_unique_vars.append(i)

    # move_list =  pruned_unique_vars + sole_vars
    move_list =  one_dim_unique + sole_vars + pruned_unique_vars
    # move_list =  sole_vars
    if redo_last:
        move_list = list(filter(lambda x: x[0] == last_action[0], move_list))
        move_list += [(-1, None ,"switch")]

    # print("[MOVELIST]: ",move_list)
    return move_list



cost_dict = {
    "sole": 2,
    "switch": 3,
    "unique": 5,
    "h_unique": 2,
    "v_unique": 1,
    "switch_function": 25,
}

transposition_table = {}

def apply_action(curr_val, curr_var: Variable, assignment: dict):
    assignment[curr_var.get_name()] = curr_val
    curr_var.set_value(curr_val)


def undo_action(var: Variable, inferred_list: list, assignment: dict):
    for i in inferred_list:
        i[0].add_to_domain(i[1])
    del assignment[var.get_name()]
    var.unset_value()

prune_hit = 0
tt_hit = 0
end_hit = 0
end_costs = []
end_paths = []
pruned_costs = []
func_hit = 0
best_end_cost = float("inf")

def Search(assignment: dict, csp: CSP, s: Sudoku_Vars, cumul_cost=0, best_cost=float("inf"), depth=0, last_action=(-1,None,"start")) -> str:
    global prune_hit, tt_hit, end_hit,func_hit,best_end_cost,end_costs,pruned_costs,end_paths
    func_hit += 1

    if(cumul_cost > best_cost):
        prune_hit += 1
        return (float("inf"),[])

    if len(assignment.keys()) == len(csp.get_nodes()):
        end_hit += 1
        end_costs.append(cumul_cost)
        s.print_board()
        return (0,[])

    global transposition_table
    global cost_dict

    assignment_hash = ';'.join(["{}:{}".format(k, v) for k, v in sorted(assignment.items())])
    assignment_hash += "|"+str(last_action[0])
    if assignment_hash in transposition_table:

        tt_hit += 1
        val = transposition_table[assignment_hash]

        # print(f"[TRANSPOSITION TABLE HIT] returning {cumul_cost} + {val} = {cumul_cost+val}")
        return val

    move_list = get_actions(csp, s, last_action)

    # cost_for_move = float("inf")
    best_move = None
    best_subtree_path = []


    if move_list:
        for i, move in enumerate(move_list):
            # print(f"\nDepth: {depth} | Move# {i} | Move {move} | Best cost {best_cost}\n")
            curr_cost = 0
            if move[2] != "switch":
                apply_action(move[0], move[1], assignment)
                curr_cost += cost_dict[move[2]]

                MAC_queue = [(ni[0], move[1], ni[1]) for ni in csp.get_neighbours_of_v(move[1])]
                _, inferred = MAC(MAC_queue, csp)
                # curr_cost += len(inferred)
                # print("len inf ",len(inferred))

            else:
                curr_cost += cost_dict[move[2]]
            cost_for_move, subtree_path = Search(assignment, csp, s, cumul_cost + curr_cost, best_cost, depth+1, move)
            curr_cost += cost_for_move
            # print("cost for applying ",move,"=", cost_for_move)
#            print("a before {}, afterP{}".format(a, str(move[0])+(move[1].get_name()if move[1] else "none")))
            if curr_cost < best_cost:
                best_cost = curr_cost
                best_move = move[2] if move[2] == 'switch' else f"{move[2]}({move[1].get_name()}:{move[0]})"
                best_subtree_path = subtree_path


            if move[2] != "switch":
                undo_action(move[1], inferred, assignment)

        transposition_table[assignment_hash] = (best_cost,best_subtree_path[:]) # subtree
        best_subtree_path.insert(0,best_move)
        return (best_cost,best_subtree_path)
    return (float("inf"),[])


def SudokuSearch(csp: CSP, s: Sudoku_Vars):

    global prune_hit, tt_hit, end_hit,func_hit,best_end_cost,end_costs,pruned_costs,end_paths
    total_queue = csp.get_arcs()
    MAC(total_queue, csp)

    assignment = {}
    for v in csp.get_nodes():
        if v.status:
            assignment[v.get_name()] = v.value
    steps = []

    out,path = Search(assignment, csp, s, 0)
    print("-------------------------------")
    print("prunehit:",prune_hit)
    print("tthit:",tt_hit)
    print("endhit:",end_hit)
    print("end cost (incl tt sums):",out)
    print("endcosts",end_costs)
    # print("prunedcosts",pruned_costs)
    print("endpaths",end_paths)
    print("funchit",func_hit)
    print("-------------------------------")
    print()
    print("the cost was ", out)
    print("with path ",path)
    if out == "success":
        return True, steps
    else:
        return False, steps
