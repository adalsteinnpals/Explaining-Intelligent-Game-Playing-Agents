


 


def coo_to_action(queen_mv,row,col,state):
    rows, columns = state.board.grid.shape
    action = col + columns*row + columns*rows*queen_mv 
    return action


state = State(player_to_move = 1)
print(state)
while True:
    input()
    p, _ = nets.predict_all(state, [])[-1]
    action = sorted([(a, p[a]) for a in state.legal_actions()], key=lambda x:-x[1])[0][0]
    state.play(action)
    print(state)
    print('Row: (q to break)')
    r = input()
    if r == 'q':
        break
    print('Column:')
    c = input()
    print('Compass direction:')
    q = input()
    action_human = coo_to_action(int(q),int(r),int(c), state)
    state.play(action_human)
    print(state)
    


#%%
    
a = np.random.rand(3,4,5)
print(a)

q,r,c = 2,1,3

print(a[q,r,c])

print(a.reshape(-1)[c+5*r+4*5*q])


#%%

# Show outputs from trained nets

print('initial state')
show_net(nets, State())

#%%
# Search with trained nets

tree = Tree(nets)
tree.think(State().play('B1 A3'), 100000, show=True)