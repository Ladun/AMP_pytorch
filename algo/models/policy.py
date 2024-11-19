
import torch
from torch import nn
from torch.distributions import MultivariateNormal

from .functions import *

class Critic(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        # -------- Initialize variables --------

        self.device     = device    
        
        activation_func = nn.ReLU        
        in_dim = config.env.state_dim
        self.m = []
        for hidden in config.actor.hidden_dim:
            self.m.append(nn.Linear(in_dim, hidden))
            self.m.append(activation_func())
            in_dim = hidden
        self.m.append(nn.Linear(in_dim, 1))        
        self.m = nn.Sequential(*self.m)       

        self.apply(lambda m: init_orthogonal_weights(m, config.critic.init_scaling))
        
    def forward(self, state):
        return self.m(state)

class Actor(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        # -------- Initialize variables --------

        self.device     = device

        # if action space is defined as continuous, make variance
        self.action_dim = config.env.action_dim
        self.action_std = config.actor.action_std_init
        self.action_var = torch.full((self.action_dim, ), config.actor.action_std_init ** 2).to(self.device)
        # learnable std
        # self.actor_logstd = nn.Parameter(torch.log(torch.ones(1, config.env.action_dim) * config.network.action_std_init))

        activation_func = nn.ReLU
        in_dim = config.env.state_dim
        self.m = []
        for hidden in config.actor.hidden_dim:
            self.m.append(nn.Linear(in_dim, hidden))
            self.m.append(activation_func())
            in_dim = hidden
        self.m.append(nn.Linear(in_dim, self.action_dim))        
        self.m = nn.Sequential(*self.m)    
        
        self.apply(lambda m: init_orthogonal_weights(m, config.actor.init_scaling))


    def action_decay(self, action_std_decay_rate, min_action_std):
        # Change the action variance
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if (self.action_std <= min_action_std):
            self.action_std = min_action_std

        self.action_var = torch.full((self.action_dim, ), self.action_std ** 2).to(self.device)

    def set_action_std(self, action_std):
        self.action_std = action_std
        self.action_var = torch.full((self.action_dim, ), self.action_std ** 2).to(self.device)

    def forward(self, state, action=None):

        # continuous space action 
        action_mean = self.m(state)

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)

        # for learnable std
        # action_logstd = self.actor_logstd.expand_as(action_mean)
        # action_std = torch.exp(action_logstd)
        # cov_mat = torch.diag_embed(action_std)
        # dist = MultivariateNormal(action_mean, cov_mat)

        # Get (action, action's log probs, estimated Value)
        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy()