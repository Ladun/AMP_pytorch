
import torch

from torch import nn
from torch.distributions import MultivariateNormal, Normal

from .functions import *

class Critic(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        # -------- Initialize variables --------

        self.device     = device    
        
        activation_func = nn.ReLU        
        in_dim = config.env.state_dim + config.env.goal_dim
        self.m = []
        for hidden in config.actor.hidden_dim:
            self.m.append(nn.Linear(in_dim, hidden))
            self.m.append(activation_func())
            in_dim = hidden
        self.m.append(nn.Linear(in_dim, 1))        
        self.m = nn.Sequential(*self.m)       

        self.apply(lambda m: init_xavier_uniform(m, config.critic.init_scaling))
        
    def forward(self, state):
        return self.m(state)
    
class Actor(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        # -------- Initialize variables --------

        self.device     = device
        self.state_dim  = config.env.state_dim
        self.goal_dim   = config.env.goal_dim

        # if action space is defined as continuous, make variance
        self.action_dim = config.env.action_dim
        
        self.action_std = config.actor.action_std_init
        if "min_action_std" in config.actor:
            self.learnable_std = False
        else:
            # learnable std
            self.learnable_std = True
            self.actor_logstd = nn.Parameter(torch.log(torch.ones(1, config.env.action_dim) * config.actor.action_std_init))

        activation_func = nn.ReLU
        in_dim = self.state_dim + self.goal_dim
        self.m = []
        for hidden in config.actor.hidden_dim:
            self.m.append(nn.Linear(in_dim, hidden))
            self.m.append(activation_func())
            in_dim = hidden
        self.m.append(nn.Linear(in_dim, self.action_dim))        
        self.m = nn.Sequential(*self.m)    
        
        self.apply(lambda m: init_xavier_uniform(m, config.actor.init_scaling) )


    def action_decay(self, action_std_decay_rate, min_action_std):
        # Change the action variance
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if (self.action_std <= min_action_std):
            self.action_std = min_action_std

    def set_action_std(self, action_std):
        self.action_std = action_std

    def forward(self, state, action=None):

        # continuous space action 
        action_mean = self.m(state)
        
        if not self.learnable_std:
            dist = Normal(action_mean, self.action_std)
        else:
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            dist = Normal(action_mean, action_std)

        # Get (action, action's log probs, estimated Value)
        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1)