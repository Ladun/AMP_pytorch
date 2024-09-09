
import torch
from torch import nn

from .functions import *

class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()

        hidden_dim = config.train.gail.hidden_dim
        activation_func = nn.ReLU
        self.m = nn.Sequential(
            nn.Linear(config.env.state_dim * 2, hidden_dim),
            activation_func(),
            nn.Linear(hidden_dim, hidden_dim),
            activation_func(),
            nn.Linear(hidden_dim, 1),
        )


    def forward(self, concated_state):
        prob = self.m(concated_state)

        return prob
    
    def get_reward(self, state, next_state):
        concat_state = torch.concat([state, next_state], dim=1)
        prob = self(concat_state)
        
        reward = torch.clamp(1 - 0.25 * torch.pow(prob - 1, 2), min=0)
        
        return reward
    
    def compute_gradient_penalty(self, samples):
        
        # Compute discriminator output at these intermediate points
        outputs = self.forward(samples)
        
        # Compute gradients at the intermediate points
        gradients = torch.autograd.grad(
            outputs=outputs,
            inputs=samples,
            create_graph=True,
            retain_graph=True,
        )[0]
        
        # Calculate L2 norm of the gradients
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = (gradients.norm(2, dim=1)** 2).mean()
        return gradient_penalty