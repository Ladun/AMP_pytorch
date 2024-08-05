
import torch
from torch import nn

from models.functions import *

class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()

        hidden_dim = config.gail.hidden_dim
        self.m = nn.Sequential(
            nn.Linear(config.env.state_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )


    def forward(self, state, next_state):
        state_action = torch.cat([state, next_state], dim=1)

        prob = self.m(state_action)

        return prob
    
    def get_reward(self, state, next_state):
        prob = self(state, next_state)
        
        reward = torch.max(0, 1 - 0.25 * torch.pow(prob - 1, 2))
        
        return reward