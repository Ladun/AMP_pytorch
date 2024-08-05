
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, gamma, tau, device):
        self.temp_memory = {
            "state": [],
            "action": [],
            "reward": [],
            "done": [],
            "value": [],
            "logprob": []
        }

        self.gamma = gamma
        self.tau = tau
        self.device = device

    def store(self, **kwargs):

        for k, v in kwargs.items():
            if k not in self.temp_memory:
                print("[Warning] wrong data insertion")
            else:
                self.temp_memory[k].append(v)

    def compute_gae_and_get(self, ns, v, d):
        """
        parameters:
            ===========  ==========================  ==================
            Symbol       Shape                       Type
            ===========  ==========================  ==================
            v            (num_envs,)                 torch.Tensor       // value in the next state
            d            (num_envs,)                 numpy.ndarray      // done in the next state

        reference from:
            https://github.com/openai/baselines/blob/master/baselines/ppo1/pposgd_simple.py line 64

        desc:
            Information about the value in storage
            ===========  ==================================  ==================
            Symbol       Shape                               Type
            ===========  ==================================  ==================
            state        list of (num_envs, (obs_space))     numpy.ndarray
            reward       list of (num_envs,)                 numpy.ndarray
            done         list of (num_envs,)                 numpy.ndarray
            action       list of (num_envs,)                 torch.Tensor
            logprob      list of (num_envs,)                 torch.Tensor
            value        list of (num_envs,)                 torch.Tensor
            ===========  ==================================  ==================
        """
        storage = {k: torch.stack(v)
                      if isinstance(v[0], torch.Tensor)
                      else torch.from_numpy(np.stack(v)).to(self.device)
                   for k, v in self.temp_memory.items()}
        
        storage['state'] = torch.cat([storage['state'], torch.from_numpy(ns).to(self.device).unsqueeze(0)], dim=0)
        storage['value'] = torch.cat([storage['value'], v.unsqueeze(0)], dim=0)
        storage['done']  = torch.cat([storage['done'], torch.from_numpy(d).to(self.device).unsqueeze(0)], dim=0).float()
                
        storage = self.__compute_gae(**storage) 
        self.clear_memory()
        return storage

                
    def __compute_gae(self, state, action, reward, done, value, logprob):
        
        steps, num_envs = reward.size()

        gae_t       = torch.zeros(num_envs).to(self.device)
        advantage   = torch.zeros((steps, num_envs)).to(self.device)

        # Each episode is calculated separately by done.
        for t in reversed(range(steps)):
            # delta(t)   = reward(t) + γ * value(t+1) - value(t)
            delta_t      = reward[t] + self.gamma * value[t+1] * (1 - done[t + 1]) - value[t]

            # gae(t)     = delta(t) + γ * τ * gae(t + 1)
            gae_t        = delta_t + self.gamma * self.tau * gae_t * (1 - done[t + 1])
            advantage[t] = gae_t
        
        # Remove value in the next state
        v_target = advantage + value[:steps]
        
        storage = {
            "state"     : state,
            "reward"    : reward,
            "action"    : action,
            "logprob"   : logprob,
            "done"      : done,
            "value"     : value,
            "advant"    : advantage,
            "v_target"  : v_target
        }
        # The first two values ​​refer to the trajectory length and number of envs.
        return {k: v[:steps].reshape(-1, *v.size()[2:]) for k, v in storage.items()}

    def clear_memory(self):
        self.temp_memory = {k: [] for k, v in self.temp_memory.items()}

    def __len__(self):
        return len(self.temp_memory['state']) * self.temp_memory['state'][0].shape[0]
