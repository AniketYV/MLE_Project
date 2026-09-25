import pickle
import numpy as np
import events as e
from .callbacks import state_to_features, get_safe_actions

REWARD_MAP = {
    e.COIN_COLLECTED: 20,
    e.CRATE_DESTROYED: 10,
    e.KILLED_OPPONENT: 50,
    e.BOMB_DROPPED: 2, 
    e.KILLED_SELF: -50,
    e.GOT_KILLED: -50,
    e.INVALID_ACTION: -5,
    e.WAITED: -1
}

def setup_training(self):
    pass

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: list):
    if old_game_state is None: return
        
    if e.BOMB_DROPPED in events:
        x, y = old_game_state['self'][3]
        field = old_game_state['field']
        others = [xy for (n, s, b, xy) in old_game_state['others']]
        
        crate_nearby = 1 in [field[x+1, y], field[x-1, y], field[x, y+1], field[x, y-1]]
        enemy_nearby = any(abs(ox - x) <= 2 and abs(oy - y) <= 2 for ox, oy in others)
        
        if not crate_nearby and not enemy_nearby:
            events.append(e.INVALID_ACTION) 
            
    step_reward = sum(REWARD_MAP.get(event, 0) for event in events)
    old_features = state_to_features(old_game_state, self_action)
    
    gamma = 0.9
    alpha = 0.1
    
    if new_game_state is None:
        max_next_q = 0
    else:
        safe_actions = get_safe_actions(new_game_state)
        next_q_values = [np.dot(self.model, state_to_features(new_game_state, a)) for a in safe_actions]
        max_next_q = np.max(next_q_values) if next_q_values else 0
        
    current_q = np.dot(self.model, old_features)
    td_error = step_reward + (gamma * max_next_q) - current_q
    self.model += alpha * td_error * old_features

def end_of_round(self, last_game_state: dict, last_action: str, events: list):
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)