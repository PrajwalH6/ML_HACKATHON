"""Deep Q-Network for Hangman."""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from typing import Dict, Set
from collections import deque
import random

class DQNetwork(nn.Module):
    """Deep Q-Network architecture."""
    
    def __init__(self, state_dim: int, action_dim: int = 26, hidden_dims=[256, 128]):
        super(DQNetwork, self).__init__()
        
        layers = []
        in_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class DQNAgent:
    """DQN Agent for Hangman."""
    
    def __init__(self, state_dim: int, action_dim: int = 26,
                 learning_rate=1e-3, gamma=0.95, epsilon=0.3,
                 epsilon_decay=0.995, epsilon_min=0.01,
                 buffer_size=10000, batch_size=64):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # Networks
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        
        # Replay buffer
        self.memory = deque(maxlen=buffer_size)
        
        # Alphabet
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
    
    def select_action(self, state: np.ndarray, valid_actions: Set[str], training=True) -> str:
        """Select action using epsilon-greedy."""
        if training and random.random() < self.epsilon:
            return random.choice(list(valid_actions))
        
        # Get Q-values
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]
        
        # Mask invalid actions
        valid_indices = [ord(a) - ord('a') for a in valid_actions]
        masked_q = np.full(26, -np.inf)
        masked_q[valid_indices] = q_values[valid_indices]
        
        action_idx = np.argmax(masked_q)
        return self.alphabet[action_idx]
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        action_idx = ord(action) - ord('a')
        self.memory.append((state, action_idx, reward, next_state, done))
    
    def update(self):
        """Update network using batch from replay buffer."""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss and update
        loss = self.loss_fn(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Copy weights from policy to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath='models/rl/dqn_weights.pth'):
        """Save model weights."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        
        print(f"DQN weights saved to {filepath}")
    
    def load(self, filepath='models/rl/dqn_weights.pth'):
        """Load model weights."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        
        print(f"DQN weights loaded from {filepath}")
