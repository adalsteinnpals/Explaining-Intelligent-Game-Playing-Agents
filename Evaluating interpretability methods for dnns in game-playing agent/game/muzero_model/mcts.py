import time, copy
import numpy as np


class Node:
    '''Search result of one abstruct (or root) state'''
    def __init__(self, p, v):
        self.p, self.v = p, v
        self.n, self.q_sum = np.zeros_like(p), np.zeros_like(p)
        self.n_all, self.q_sum_all = 1, v / 2 # prior

    def update(self, action, q_new):
        # Update
        self.n[action] += 1
        self.q_sum[action] += q_new

        # Update overall stats
        self.n_all += 1
        self.q_sum_all += q_new
        

class Tree:
    '''Monte Carlo Tree'''
    def __init__(self, nets):
        self.nets = nets
        self.nodes = {}

    def search(self, state, path, rp, depth):
        # Return predicted value from new state
        """
        Simulate recursively until a new state is reached:
            Pick best action on current node defined by UCB
            
        """

        if self.nets.model_name == 'alphazero':
            if state.terminal():
                return -1

        key = state.record_string()
        if len(path) > 0:
            key += '|' + ' '.join(map(state.action2str, path))
        if key not in self.nodes:
            p, v = self.nets.prediction.inference(rp)                
            self.nodes[key] = Node(p, v)
            return v
        

        # State transition by an action selected from bandit
        node = self.nodes[key]
        p = node.p
        mask = np.zeros_like(p)
        if depth == 0 or self.nets.model_name == 'alphazero':
            # Add noise to policy on the root node
            if depth == 0:
                p = 0.75 * p + 0.25 * np.random.dirichlet([0.3] * len(p))  # dirichlet noise 0.3 as in paper [alphazero]
            
            # On the root node, we choose action only from legal actions            
            mask[state.legal_actions()] = 1
            p *= mask
            p /= p.sum() + 1e-16

        n, q_sum = 1 + node.n, node.q_sum_all / node.n_all + node.q_sum
        ucb = q_sum / n + 2.0 * np.sqrt(node.n_all) * p / n + mask * 4 # PUCB formula
        best_action = np.argmax(ucb)

        # Search next state by recursively calling this function
        if self.nets.model_name == 'muzero':
            representation = self.nets.dynamics.inference(rp, 
                                                          state.action_feature(best_action))
        else:
            state.play(best_action)
            representation = state.feature()
            # TODO once found invalid move, maybe a bug
            
        path.append(best_action)
        
        q_new = -self.search(state, path, representation, depth + 1) # With the assumption of changing player by turn
        node.update(best_action, q_new)

        return q_new

    def think(self, state, num_simulations, temperature = 1, show=False):
        # End point of MCTS
        if show:
            print(state)
        start, prev_time = time.time(), 0
        
        """
        BOTH WHITE AND BLACK TRY TO MAXIMIZE THE REWARD
        """
        if self.nets.model_name == 'muzero':
            state_rp = self.nets.representation.inference(state.feature())
        else:
            state_rp = state.feature()
        for _ in range(num_simulations):
            state_ = copy.deepcopy(state)
            self.search(state_, [], state_rp, depth=0)

            # Display search result on every second
            if show:
                tmp_time = time.time() - start
                if int(tmp_time) > int(prev_time):
                    prev_time = tmp_time
                    root, pv = self.nodes[state.record_string()], self.pv(state)
                    print('%.2f sec. best %s. q = %.4f. n = %d / %d. pv = %s'
                          % (tmp_time, state.action2str(pv[0]), root.q_sum[pv[0]] / root.n[pv[0]],
                             root.n[pv[0]], root.n_all, ' '.join([state.action2str(a) for a in pv])))

        #  Return probability distribution weighted by the number of simulations
        n = root = self.nodes[state.record_string()].n + 1
        n = (n / np.max(n)) ** (1 / (temperature + 1e-8))
        return n / n.sum()

    def pv(self, state):
        # Return principal variation (action sequence which is considered as the best)
        s, pv_seq = copy.deepcopy(state), []
        while True:
            key = s.record_string()
            if key not in self.nodes or self.nodes[key].n.sum() == 0:
                break
            best_action = sorted([(a, self.nodes[key].n[a]) for a in s.legal_actions()], key=lambda x: -x[1])[0][0]
            pv_seq.append(best_action)
            s.play(best_action)
        return pv_seq
    
    
if __name__ == '__main__':
    
    
    from models import Nets
    from utils import load_saved_model, save_model, load_episodes_only
    
    net = Nets()
    
    
    
    
    
    
    
    
    