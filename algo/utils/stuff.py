import os
import numpy as np
import torch

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class RunningMeanStd:
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

    def save_variables(self):
        return {
            "mean": self.mean,
            "var": self.var,
            "count": self.count
        }

    def load_variables(self, saved_dict):
        self.mean = saved_dict['mean']
        self.var = saved_dict['var']
        self.count = saved_dict['count']



class Normalizer:
    def __init__(self, shape, name):
        self.rms = RunningMeanStd(shape=shape)
        self.name = name

    def __call__(self, obs, update=True):
        if update:
            self.rms.update(obs)
        return np.clip((obs - self.rms.mean) / np.sqrt(self.rms.var + 1e-8), -10, 10)

    def save(self, base_path):
        save_path = os.path.join(base_path, f"{self.name}_normalizer.pth")
        torch.save(self.rms.save_variables(), save_path)

    def load(self, base_path):
        load_path = os.path.join(base_path, f"{self.name}_normalizer.pth")
        self.rms.load_variables(torch.load(load_path, weights_only=False))


# Reward scaler
class RewardScaler:
    def __init__(self, num_envs, gamma=1.):
        self.rms = RunningMeanStd(shape=())
        self.gamma = gamma
        self.ret = np.zeros(num_envs)

    def __call__(self, agent_id, reward, dones, update=True):
        self.ret[agent_id] = self.ret[agent_id] * self.gamma + reward
        if update:
            self.rms.update(self.ret[agent_id])
        reward = reward / np.sqrt(self.rms.var + 1e-8)

        self.ret[agent_id][dones] = 0
        return np.clip(reward, -10, 10)

    def save(self, base_path):
        save_path = os.path.join(base_path, "reward_scaler.pth")
        torch.save(self.rms.save_variables(), save_path)

    def load(self, base_path):
        load_path = os.path.join(base_path, "reward_scaler.pth")
        self.rms.load_variables(torch.load(load_path, weights_only=False))
        