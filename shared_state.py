import json
import os

STATE_FILE = 'shared_state.json'

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {}

def update_state(key, value):
    state = load_state()
    state[key] = value
    save_state(state)

def get_state_version():
    state = load_state()
    return state.get('version', 0)
