"""Tabular Q-learning agent for Hangman."""
import numpy as np
import pickle
from collections import defaultdict
from typing import Dict, Set
from pathlib import Path

class QLearningAgent:
    """Tabular Q-learning agent."""
    
    def __init__(self, learning_rate=0.1, discount_factor=0.95, 
                 epsilon=0.2, epsilon_decay=0.995, epsilon_min=0.01):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: dict of {state_key: {action: q_value}}
        self.q_table = defaultdict(lambda: defaultdict(float))
        
    def _get_state_key(self, state: Dict, hmm_probs: Dict[str, float]) -> str:
        """Create hashable state key."""
        mask = state['mask']
        guessed = ''.join(sorted(state['guessed_letters']))
        lives = state['lives']
        
        # Include top-3 HMM recommended letters
        top_letters = sorted(hmm_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        hmm_hint = ''.join([letter for letter, _ in top_letters])
        
        return f"{mask}|{guessed}|{lives}|{hmm_hint}"
    
    def select_action(self, state: Dict, hmm_probs: Dict[str, float], 
                     valid_actions: Set[str], training=True) -> str:
        """Select action using epsilon-greedy policy."""
        state_key = self._get_state_key(state, hmm_probs)
        
        # Exploration
        if training and np.random.random() < self.epsilon:
            return np.random.choice(list(valid_actions))
        
        # Exploitation: choose best valid action
        q_values = self.q_table[state_key]
        valid_q = {action: q_values[action] for action in valid_actions}
        
        if not valid_q:
            return np.random.choice(list(valid_actions))
        
        # Combine Q-values with HMM probs for better guidance
        combined_scores = {}
        for action in valid_actions:
            q_val = valid_q.get(action, 0.0)
            hmm_val = hmm_probs.get(action, 0.0)
            combined_scores[action] = q_val + 0.3 * hmm_val  # Weight HMM guidance
        
        return max(combined_scores, key=combined_scores.get)
    
    def update(self, state: Dict, action: str, reward: float, 
               next_state: Dict, done: bool, hmm_probs: Dict[str, float],
               next_hmm_probs: Dict[str, float]):
        """Update Q-value using Q-learning update rule."""
        state_key = self._get_state_key(state, hmm_probs)
        next_state_key = self._get_state_key(next_state, next_hmm_probs)
        
        # Current Q-value
        current_q = self.q_table[state_key][action]
        
        # Max Q-value for next state
        if done:
            max_next_q = 0.0
        else:
            next_q_values = self.q_table[next_state_key]
            max_next_q = max(next_q_values.values()) if next_q_values else 0.0
        
        # Q-learning update
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_key][action] = new_q
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath='models/rl/q_table.pkl'):
        """Save Q-table."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert defaultdict to regular dict for pickling
        q_table_dict = {k: dict(v) for k, v in self.q_table.items()}
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': q_table_dict,
                'epsilon': self.epsilon
            }, f)
        
        print(f"Q-table saved to {filepath}")
    
    def load(self, filepath='models/rl/q_table.pkl'):
        """Load Q-table."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Convert back to defaultdict
        self.q_table = defaultdict(lambda: defaultdict(float))
        for state_key, actions in data['q_table'].items():
            self.q_table[state_key] = defaultdict(float, actions)
        
        self.epsilon = data.get('epsilon', self.epsilon_min)
        print(f"Q-table loaded from {filepath}")
