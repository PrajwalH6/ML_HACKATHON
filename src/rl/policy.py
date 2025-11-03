"""Exploration policies."""
import numpy as np

class EpsilonGreedy:
    """Epsilon-greedy exploration."""
    
    def __init__(self, epsilon=0.3, epsilon_decay=0.995, epsilon_min=0.01):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
    
    def select_action(self, q_values, valid_actions):
        """Select action using epsilon-greedy."""
        if np.random.random() < self.epsilon:
            return np.random.choice(list(valid_actions))
        
        # Greedy: select best valid action
        valid_indices = [ord(a) - ord('a') for a in valid_actions]
        masked_q = np.full(26, -np.inf)
        masked_q[valid_indices] = q_values[valid_indices]
        
        action_idx = np.argmax(masked_q)
        return chr(ord('a') + action_idx)
    
    def decay(self):
        """Decay epsilon."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
