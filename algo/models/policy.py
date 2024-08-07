
import torch
from torch import nn
from torch.distributions import MultivariateNormal

from .functions import *

class ActorCritic(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        # -------- Initialize variables --------

        self.device     = device
        self.shared_layer = config.network.shared_layer

        # if action space is defined as continuous, make variance
        self.action_dim = config.env.action_dim
        self.action_std = config.network.action_std_init
        self.action_var = torch.full((self.action_dim, ), config.network.action_std_init ** 2).to(self.device)
        # learnable std
        # self.actor_logstd = nn.Parameter(torch.log(torch.ones(1, config.env.action_dim) * config.network.action_std_init))

        if self.shared_layer:
            self.shared_net = nn.Sequential(
                nn.Linear(config.env.state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh()
            )
            self.actor = nn.Sequential(
                nn.Linear(64, config.env.action_dim),
                nn.Tanh()
            )
            self.critic = nn.Linear(64, 1)

        else:
            self.actor = nn.Sequential(
                nn.Linear(config.env.state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, config.env.action_dim),
                nn.Tanh()
            )
            self.critic = nn.Sequential(
                nn.Linear(config.env.state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )          
        
        self.apply(init_orthogonal_weights)


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
        if self.shared_layer:
            state = self.shared_net(state)

        # continuous space action 
        action_mean = self.actor(state)

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

        return action, dist.log_prob(action), dist.entropy(), self.critic(state)