
import torch
from torch import nn

from .functions import *

class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()

        hidden_dim = config.train.gail.hidden_dim
        activation_func = nn.ReLU
        in_dim = config.env.state_dim * 2
        self.m = []
        for hidden in config.actor.hidden_dim:
            self.m.append(nn.Linear(in_dim, hidden))
            self.m.append(activation_func())
            in_dim = hidden
        self.m.append(nn.Linear(in_dim, 1))        
        self.m = nn.Sequential(*self.m)  
        
        self.apply(lambda m: init_orthogonal_weights(m, config.actor.init_scaling))   


    def forward(self, concated_state):
        prob = self.m(concated_state)

        return prob
    
    def get_reward(self, state, next_state):
        concat_state = torch.concat([state, next_state], dim=1)
        prob = self(concat_state)
        
        reward = torch.clamp(1 - 0.25 * torch.square(1 - prob), min=0)
        
        return reward
    
    def compute_gradient_penalty(self, samples):
        
        samples = samples.requires_grad_(True)
        
        outputs = self(samples)
        
        # Fake output tensor
        fake = torch.ones(samples.size(0), 1, device=samples.device, requires_grad=False)
        
        # Compute gradients at the intermediate points
        gradients = torch.autograd.grad(
            outputs=outputs,
            inputs=samples,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        # Calculate L2 norm of the gradients
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = (gradients.norm(2, dim=1) ** 2).mean()
        return gradient_penalty