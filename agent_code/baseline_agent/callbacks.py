import os
import pickle
import numpy as np
from collections import deque

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup(self):
    if self.train or not os.path.isfile("my-saved-model.pt"):
        # Exactly 4 weights mapped to the 4 priorities
        self.model = np.zeros(4) 
    else:
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)

def is_dangerous(game_state, tx, ty):
    if game_state['explosion_map'][tx, ty] != 0: return True
    field = game_state['field']
    for (bx, by), timer in game_state['bombs']:
        if tx == bx and ty == by: return True
        if tx == bx and 1 <= abs(ty - by) <= 3:
            step = 1 if ty > by else -1
            blocked = False
            for i in range(step, ty - by, step):
                if field[bx, by + i] == -1: 
                    blocked = True; break
            if not blocked: return True
        if ty == by and 1 <= abs(tx - bx) <= 3:
            step = 1 if tx > bx else -1
            blocked = False
            for i in range(step, tx - bx, step):
                if field[bx + i, by] == -1: 
                    blocked = True; break
            if not blocked: return True
    return False

def get_bfs_action(game_state, mode):
    start_x, start_y = game_state['self'][3]
    field = game_state['field']
    coins = [(c[0], c[1]) for c in game_state['coins']]
    others = [xy for (n, s, b, xy) in game_state['others']]
    
    queue = deque([(start_x, start_y, None)])
    visited = set([(start_x, start_y)])
    
    while queue:
        cx, cy, first_action = queue.popleft()
        
        #  Escape Check
        if mode == 'escape' and not is_dangerous(game_state, cx, cy):
            return first_action
            
        if mode == 'hunt' and not is_dangerous(game_state, cx, cy):
            #  Check if agent is around to bomb
            if (cx, cy) in others: return first_action
            #  Explore unbombed crates and coins
            if (cx, cy) in coins: return first_action
            if field[cx, cy] == 1: return first_action
                
        for move, dx, dy in [('UP', 0, -1), ('DOWN', 0, 1), ('LEFT', -1, 0), ('RIGHT', 1, 0)]:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) not in visited and field[nx, ny] >= 0:
                visited.add((nx, ny))
                if field[nx, ny] == 0 and (nx, ny) not in others:
                    queue.append((nx, ny, first_action if first_action else move))
                elif mode == 'hunt' and (field[nx, ny] == 1 or (nx, ny) in others):
                    return first_action if first_action else 'BOMB'
    return None

def get_safe_actions(game_state):
    x, y = game_state['self'][3]
    field = game_state['field']
    bombs = [b[0] for b in game_state['bombs']]
    others = [xy for (n, s, b, xy) in game_state['others']]
    
    safe_actions = []
    for action in ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']:
        dx, dy = 0, 0
        if action == 'UP': dy = -1
        elif action == 'DOWN': dy = 1
        elif action == 'LEFT': dx = -1
        elif action == 'RIGHT': dx = 1
        nx, ny = x + dx, y + dy
        
        if field[nx, ny] != 0 or (nx, ny) in others or (nx, ny) in bombs: continue
        if is_dangerous(game_state, nx, ny): continue
        safe_actions.append(action)
        
    if game_state['self'][2]: 
        game_state['bombs'].append(((x, y), 4)) 
        escape = get_bfs_action(game_state, 'escape')
        game_state['bombs'].pop() 
        if escape is not None:
            safe_actions.append('BOMB')
            
    return safe_actions if safe_actions else ['WAIT']

def act(self, game_state: dict) -> str:
    x, y = game_state['self'][3]
    
    # Immediate Escape
    if is_dangerous(game_state, x, y):
        escape = get_bfs_action(game_state, 'escape')
        if escape: return escape
        
    safe_actions = get_safe_actions(game_state)
    epsilon = 0.15 
    
    if self.train and np.random.rand() < epsilon:
        return np.random.choice(safe_actions)
        
    q_values = []
    for action in safe_actions:
        features = state_to_features(game_state, action)
        q_values.append(np.dot(self.model, features))
        
    best_action = safe_actions[np.argmax(q_values)]
    
    #  Anti-Stuck Override for early learning phases
    if np.max(q_values) == 0:
        hunt_action = get_bfs_action(game_state, 'hunt')
        #  Exit initially by safely bombing crates
        if hunt_action == 'BOMB' and 'BOMB' in safe_actions:
            return 'BOMB'
        #  Explore
        if hunt_action in safe_actions:
            return hunt_action
            
        safe_moves = [a for a in safe_actions if a not in ['WAIT', 'BOMB']]
        if safe_moves: return np.random.choice(safe_moves)
        
    return best_action

def state_to_features(game_state: dict, action: str) -> np.array:
    if game_state is None: return np.zeros(4)
    
    x, y = game_state['self'][3]
    field = game_state['field']
    others = [xy for (n, s, b, xy) in game_state['others']]
    features = np.zeros(4)
    
    # Priority 1 Feature: Crate Bombing
    if action == 'BOMB':
        if 1 in [field[x+1, y], field[x-1, y], field[x, y+1], field[x, y-1]]:
            features[0] = 1.0
            
    # Priority 2 Feature: Enemy Bombing 
    if action == 'BOMB':
        if any(abs(ox - x) <= 2 and abs(oy - y) <= 2 for ox, oy in others):
            features[1] = 1.0
            
    # Priority 3 Feature: Explore/Loot Pathfinding
    if action == get_bfs_action(game_state, 'hunt'):
        features[2] = 1.0
        
    # Priority 4 Feature: Enemy Hunting (Closing distance to prevent freezing)
    dx, dy = 0, 0
    if action == 'UP': dy = -1
    elif action == 'DOWN': dy = 1
    elif action == 'LEFT': dx = -1
    elif action == 'RIGHT': dx = 1
    nx, ny = x + dx, y + dy
    
    if action in ['UP', 'DOWN', 'LEFT', 'RIGHT'] and others:
        current_dist = min(abs(ox - x) + abs(oy - y) for ox, oy in others)
        next_dist = min(abs(ox - nx) + abs(oy - ny) for ox, oy in others)
        if next_dist < current_dist:
            features[3] = 1.0
            
    return features