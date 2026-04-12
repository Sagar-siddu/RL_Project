"""
PyTorch PPO Agent for Dynamic Pricing Environment
Meta PyTorch Hackathon Entry
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from typing import Tuple, List, Dict
from collections import deque
import json

class PricingPolicyNetwork(nn.Module):
    """Policy Network for Dynamic Pricing (Actor-Critic)"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Actor head (mean and log std of prices)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softplus()  # Prices must be positive
        )
        
        self.actor_log_std = nn.Parameter(
            torch.zeros(action_dim)
        )
        
        # Critic head (value estimation)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning mean, log_std, and value"""
        features = self.backbone(state)
        
        mean = self.actor_mean(features)
        log_std = self.actor_log_std.expand_as(mean)
        value = self.critic(features)
        
        return mean, log_std, value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy"""
        mean, log_std, value = self.forward(state)
        
        if deterministic:
            return mean, value
        
        std = log_std.exp()
        dist = Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Clamp prices to reasonable range
        action = torch.clamp(action, min=0.1, max=1000.0)
        
        return action, log_prob, value

class PPOAgent:
    """PPO Agent for Dynamic Pricing Environment"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.policy = PricingPolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = 0.2
        self.epochs_per_update = 5
        self.batch_size = 64
        
        self.memory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': [],
            'episode_returns': []
        }
        
        self.episode_rewards = []
        self.episode_lengths = []
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        """Select action from state"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if deterministic:
                action, value = self.policy.get_action(state_tensor, deterministic=True)
                log_prob = torch.zeros(1).to(self.device)
            else:
                action, log_prob, value = self.policy.get_action(state_tensor, deterministic=False)
        
        return action.cpu().numpy().squeeze(), log_prob.item(), value.item()
    
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """Store transition in memory"""
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['rewards'].append(reward)
        self.memory['values'].append(value)
        self.memory['log_probs'].append(log_prob)
        self.memory['dones'].append(done)
    
    def compute_returns_and_advantages(self) -> Tuple[List[float], List[float]]:
        """Compute returns and GAE advantages"""
        returns = []
        advantages = []
        
        next_value = 0
        gae = 0
        
        for t in reversed(range(len(self.memory['rewards']))):
            if t == len(self.memory['rewards']) - 1:
                next_value = 0 if self.memory['dones'][t] else next_value
            else:
                next_value = self.memory['values'][t + 1]
            
            delta = (
                self.memory['rewards'][t] +
                self.gamma * next_value * (1 - self.memory['dones'][t]) -
                self.memory['values'][t]
            )
            
            gae = delta + self.gamma * self.gae_lambda * (1 - self.memory['dones'][t]) * gae
            
            returns.insert(0, gae + self.memory['values'][t])
            advantages.insert(0, gae)
        
        return returns, advantages
    
    def update(self) -> Dict[str, float]:
        """PPO update step"""
        if len(self.memory['states']) == 0:
            return {'loss': 0.0}
        
        returns, advantages = self.compute_returns_and_advantages()
        
        # Normalize advantages
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        states = torch.FloatTensor(np.array(self.memory['states'])).to(self.device)
        actions = torch.FloatTensor(np.array(self.memory['actions'])).to(self.device)
        old_log_probs = torch.FloatTensor(self.memory['log_probs']).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        
        total_loss = 0
        
        for epoch in range(self.epochs_per_update):
            # Shuffle data
            indices = np.random.permutation(len(self.memory['states']))
            
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i+self.batch_size]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                
                # Forward pass
                mean, log_std, values = self.policy(batch_states)
                
                # New log probs
                std = log_std.exp()
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                
                # PPO loss
                ratio = (new_log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * ((values.squeeze() - batch_returns) ** 2).mean()
                entropy = dist.entropy().mean()
                
                # Total loss with entropy bonus
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                
                total_loss += loss.item()
        
        # Clear memory
        self.memory = {k: [] for k in self.memory}
        
        return {'loss': total_loss / (self.epochs_per_update * len(range(0, len(indices), self.batch_size)))}
    
    def save_checkpoint(self, path: str):
        """Save agent checkpoint"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load agent checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def get_stats(self) -> Dict:
        """Get training statistics"""
        if not self.episode_rewards:
            return {
                'mean_episode_reward': 0,
                'max_episode_reward': 0, 
                'mean_episode_length': 0
            }
        
        return {
            'mean_episode_reward': float(np.mean(self.episode_rewards[-100:])),
            'max_episode_reward': float(np.max(self.episode_rewards[-100:])),
            'mean_episode_length': float(np.mean(self.episode_lengths[-100:]))
        }
