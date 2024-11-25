
import os
import glob
import numpy as np
import logging
import shutil
import math
from datetime import datetime
from collections import defaultdict

from omegaconf import OmegaConf
import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from .models.policy import Actor, Critic
from .models.discriminator import Discriminator
from .data.motion_dataset import MotionDataset
from .scheduler import WarmupLinearSchedule
from .replay_buffer import ReplayBuffer, UnityReplayBuffer
from .utils.general import (
    set_seed, pretty_config, get_cur_time_code,
    get_config, get_rng_state, set_rng_state,
    TimerManager, get_device
)
from .utils.stuff import RewardScaler, Normalizer
from .data.preprocess_deepmimic_data import parse_skeleton_file, DOFS, get_disc_motion_dim

logger = logging.getLogger(__name__)


def data_iterator(batch_size, given_data):
    # Simple mini-batch spliter

    total_size = len(given_data[0])
    indices = np.arange(total_size)
    np.random.shuffle(indices)
    n_batches = total_size // batch_size
    for nb in range(n_batches):
        ind = indices[batch_size * nb : batch_size * (nb + 1)]
        yield (d[ind] for d in given_data) 


class AMPAgent:
    def __init__(self, config, eval=False):
        
        self.config = config
        self.device = get_device(config.device)
        
        set_seed(self.config.seed)
               
        # -------- Define gail dataset --------            
                     
        if self.config.train.gail.use:
            # Disriminator dataset
            if not eval:
                self.motion_dataset = MotionDataset(self.config.train.gail.dataset_file, 
                                                    self.config.train.gail.skeleton_file)
            self.skeleton_info = parse_skeleton_file(self.config.train.gail.skeleton_file)
            
        # -------- Define models --------
        
        self.actor = Actor(config, self.device).to(self.device) 
        self.critic = Critic(config, self.device).to(self.device) 
        if self.config.train.gail.use:
            self.disc = Discriminator(config, get_disc_motion_dim(self.skeleton_info) * 2).to(self.device) 
        
        # -------- Define train supporter ---------
               
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            **self.config.actor.optimizer
        )  
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            **self.config.critic.optimizer
        )
                     
        if self.config.train.gail.use:
            self.disc_optimizer  = torch.optim.Adam([
                {'params': self.disc.parameters(),
                **config.train.gail.optimizer}
            ])
        
        if self.config.train.scheduler:
            self.actor_scheduler = WarmupLinearSchedule(optimizer=self.actor_optimizer,
                                                        warmup_steps=0,
                                                        max_steps=self.config.train.total_timesteps // (self.config.train.max_episode_len * self.config.env.num_envs))
            
            self.critic_scheduler = WarmupLinearSchedule(optimizer=self.critic_optimizer,
                                                         warmup_steps=0,
                                                         max_steps=self.config.train.total_timesteps // (self.config.train.max_episode_len * self.config.env.num_envs))
                         
            if self.config.train.gail.use:
                self.disc_scheduler = WarmupLinearSchedule(optimizer=self.disc_optimizer,
                                                           warmup_steps=0, 
                                                           max_steps=self.config.train.total_timesteps // (self.config.train.max_episode_len * self.config.env.num_envs))
                    
        # reward scaler: r / rs.std()
        if self.config.train.reward_scaler:
            self.reward_scaler = RewardScaler(self.config.env.num_envs, gamma=self.config.train.gamma)

        # observation scaler: (ob - ob.mean()) / (ob.std())
        if self.config.train.observation_normalizer:
            sp = (config.env.state_dim, ) if isinstance(config.env.state_dim, int) else list(config.env.state_dim)
            self.obs_normalizer = Normalizer(self.config.env.num_envs, sp, "obs")
            
        
        if self.config.train.goal_normalizer:
            sp = (config.env.goal_dim, ) 
            self.goal_normalizer = Normalizer(self.config.env.num_envs, sp, "goal")
        
        
        self.timer_manager  = TimerManager()
        self.writer         = None
        self.memory         = None
        self.timesteps      = 0
        
        self.w_g = self.config.train.task_reward_lerp
        self.w_s = (1 - self.w_g)
        
        self.episodic_reward = None
        self.style_reward = None
        self.duration = None
        
        logger.info("----------- Config -----------")
        pretty_config(config, logger=logger)
        logger.info(f"Device: {self.device}")
    
    def save(self, postfix):
        '''
        ckpt_root
            exp_name
                config.yaml
                checkpoints
                    1
                    2
                    ...
                
        '''

        ckpt_path = os.path.join(self.config.experiment_path, "checkpoints")
        if os.path.exists(ckpt_path):
            # In order to save only the maximum number of checkpoints as max_save_store,
            # checkpoints exceeding that number are deleted. (exclude 'best')
            current_ckpt = [f for f in os.listdir(ckpt_path) if f.startswith('timesteps')]
            current_ckpt.sort(key=lambda x: int(x[9:]))
            # Delete exceeded checkpoints
            if self.config.train.max_ckpt_count > 0 and self.config.train.max_ckpt_count <= len(current_ckpt):
                for ckpt in current_ckpt[:len(current_ckpt) - self.config.train.max_ckpt_count - 1]:
                    shutil.rmtree(os.path.join(self.config.experiment_path, "checkpoints", ckpt), ignore_errors=True)

        # Save configuration file
        os.makedirs(self.config.experiment_path, exist_ok=True)
        with open(os.path.join(self.config.experiment_path, "config.yaml"), 'w') as fp:
            OmegaConf.save(config=self.config, f=fp)

        # postfix is ​​a variable for storing each episode or the best model
        ckpt_path = os.path.join(ckpt_path, postfix)
        os.makedirs(ckpt_path, exist_ok=True)
        
        # save model and optimizers
        torch.save(self.actor.state_dict(), os.path.join(ckpt_path, "actor.pt"))
        torch.save(self.actor_optimizer.state_dict(), os.path.join(ckpt_path, "actor_optimizer.pt"))
        torch.save(self.critic.state_dict(), os.path.join(ckpt_path, "critic.pt"))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(ckpt_path, "critic_optimizer.pt"))
                     
        if self.config.train.gail.use:
            torch.save(self.disc.state_dict(), os.path.join(ckpt_path, "discriminator.pt"))
            torch.save(self.disc_optimizer.state_dict(), os.path.join(ckpt_path, "disc_optimizer.pt"))
        
        if self.config.train.scheduler:
            torch.save(self.actor_scheduler.state_dict(), os.path.join(ckpt_path, "actor_scheduler.pt"))
            torch.save(self.critic_scheduler.state_dict(), os.path.join(ckpt_path, "critic_scheduler.pt"))
                         
            if self.config.train.gail.use:
                torch.save(self.disc_scheduler.state_dict(), os.path.join(ckpt_path, "disc_scheduler.pt"))

        if self.config.train.reward_scaler:
            self.reward_scaler.save(ckpt_path)
        if self.config.train.observation_normalizer:
            self.obs_normalizer.save(ckpt_path)
        if self.config.train.goal_normalizer:
            self.goal_normalizer.save(ckpt_path)

        # save random state
        torch.save(get_rng_state(), os.path.join(ckpt_path, 'rng_state.pkl'))

        with open(os.path.join(ckpt_path, "appendix"), "w") as f:
            f.write(f"{self.timesteps}\n")
    
    
    @classmethod
    def load(cls, experiment_path, postfix, resume=True, eval=False):

        config = get_config(os.path.join(experiment_path, "config.yaml"))
        config.actor.action_std_init = config.actor.min_action_std
        amp_algo = AMPAgent(config, eval)
        
        # Create a variable to indicate which path the model will be read from
        ckpt_path = os.path.join(experiment_path, "checkpoints", postfix)
        print(f"Load pretrained model from {ckpt_path}")

        amp_algo.actor.load_state_dict(torch.load(os.path.join(ckpt_path, "actor.pt")))
        amp_algo.actor_optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, "actor_optimizer.pt")))
        amp_algo.critic.load_state_dict(torch.load(os.path.join(ckpt_path, "critic.pt")))
        amp_algo.critic_optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, "critic_optimizer.pt")))
        
        
        if amp_algo.config.train.gail.use:
            amp_algo.disc.load_state_dict(torch.load(os.path.join(ckpt_path, "discriminator.pt")))
            amp_algo.disc_optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, "disc_optimizer.pt")))
        
        if amp_algo.config.train.scheduler:
            amp_algo.actor_scheduler.load_state_dict(torch.load(os.path.join(ckpt_path, "actor_scheduler.pt")))
            amp_algo.critic_scheduler.load_state_dict(torch.load(os.path.join(ckpt_path, "critic_scheduler.pt")))
            
            if amp_algo.config.train.gail.use:
                amp_algo.disc_scheduler.load_state_dict(torch.load(os.path.join(ckpt_path, "disc_scheduler.pt")))
        
        if amp_algo.config.train.reward_scaler:
            amp_algo.reward_scaler.load(ckpt_path)
        if amp_algo.config.train.observation_normalizer:
            amp_algo.obs_normalizer.load(ckpt_path)
        if amp_algo.config.train.goal_normalizer:
            amp_algo.goal_normalizer.load(ckpt_path)

        # load random state
        set_rng_state(torch.load(os.path.join(ckpt_path, 'rng_state.pkl'), map_location='cpu'))

        with open(os.path.join(ckpt_path, "appendix"), "r") as f:
            lines = f.readlines()

        if resume:
            amp_algo.timesteps = int(lines[0])

        return amp_algo
     
    def optimize_policy(self, data):
        data_loader = data_iterator(self.config.train.ppo.batch_size, data)   

        policy_losses   = []
        entropy_losses  = []
        value_losses    = []
        total_losses    = []
        
        self.actor.train()
        self.critic.train()
        for batch in data_loader:
            ob, _, ac, old_logp, adv, vtarg, old_v = batch
            adv = (adv - adv.mean()) / (adv.std() + 1e-7)

            # -------- Loss calculate --------

            # --- policy loss
            _, cur_logp, cur_ent = self.actor(ob, action=ac)
            cur_v = self.critic(ob)
            cur_v = cur_v.reshape(-1)

            ratio = torch.exp(cur_logp - old_logp)
            surr1 = ratio * adv

            if self.config.train.ppo.loss_type == "clip":
                # clipped loss
                clipped_ratio = torch.clamp(ratio, 1. - self.config.train.ppo.eps_clip, 1. + self.config.train.ppo.eps_clip)
                surr2 = clipped_ratio * adv

                policy_surr = torch.min(surr1, surr2)
                
            elif self.config.train.ppo.loss_type == "kl":
                # kl-divergence loss
                policy_surr = surr1 - 0.01 * torch.exp(old_logp) * (old_logp - cur_logp)
            else:
                # simple ratio loss
                policy_surr = surr1
            
            policy_surr = -policy_surr.mean()

            # --- entropy loss

            policy_ent = -cur_ent.mean()
            
            # -------- Actor Backward process --------
            actor_loss = policy_surr + self.config.train.ppo.coef_entropy_penalty * policy_ent            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.config.train.clipping_gradient:
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
            self.actor_optimizer.step()
            

            # --- value loss

            if self.config.train.ppo.value_clipping:
                cur_v_clipped = old_v + (cur_v - old_v).clamp(-self.config.train.ppo.eps_clip, self.config.train.ppo.eps_clip)
                vloss1 = (cur_v - vtarg) ** 2
                vloss2 = (cur_v_clipped - vtarg) ** 2
                vf_loss = torch.max(vloss1, vloss2)
            else:
                vf_loss = (cur_v - vtarg) ** 2 
                
            vf_loss = 0.5 * vf_loss.mean()

            # -------- Critic Backward process --------
            
            critic_loss = self.config.train.ppo.coef_value_function * vf_loss
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.config.train.clipping_gradient:
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
            self.critic_optimizer.step()

            # ---------- Record training loss data ----------
            total_loss = actor_loss + critic_loss
    
            policy_losses.append(policy_surr.item())
            entropy_losses.append(policy_ent.item())
            value_losses.append(vf_loss.item())
            total_losses.append(total_loss.item())
            
        return {
            "train/policy_loss": policy_losses,
            "train/entropy_loss": entropy_losses,
            "train/value_loss": value_losses,
            "train/total_loss": total_losses
        }


    def optimize_discriminator(self, data):
        
        weight_sum = sum(set(self.motion_dataset.weights))
        weights = weight_sum / np.array(self.motion_dataset.weights)
        sampler = WeightedRandomSampler(weights, num_samples=len(data[0]), replacement=True)
        
        expert_loader = DataLoader(self.motion_dataset, 
                                   batch_size=self.config.train.gail.batch_size,
                                   drop_last=True,
                                   sampler=sampler)  
        expert_iter = iter(expert_loader) 
        agent_iter  = data_iterator(self.config.train.gail.batch_size, data)    
        
        iter_len = len(data[0]) // self.config.train.gail.batch_size
    
        loss_fn = nn.MSELoss()
        discriminator_losses = []
        agent_accuracies = []
        expert_accuracies = []
        gradient_penalties = []
        
        self.disc.train()
        for _ in range(iter_len):
            try:
                agent_data = next(agent_iter)
            except StopIteration as e:
                agent_iter = data_iterator(self.config.train.gail.batch_size, data)    
                agent_data = next(agent_iter)
            agent_data = torch.concat([d.to(self.device) for d in agent_data], dim=1)
        
            try:
                expert_data = next(expert_iter)
            except StopIteration as e:
                expert_iter = iter(expert_loader)   
                expert_data = next(expert_iter)
            expert_data = torch.concat([d.to(self.device) for d in expert_data], dim=1)      

            # Train
            agent_prob = self.disc(agent_data)
            expert_prob = self.disc(expert_data)

            agent_loss = 0.5 * loss_fn(agent_prob, -torch.ones_like(agent_prob))
            expert_loss = 0.5 * loss_fn(expert_prob, torch.ones_like(expert_prob))
        
            # minimize E_expert [ (D(s, s') - 1)^2] + E_agent [(D(s, s') + 1)^2]
            loss = 0.5 * (agent_loss + expert_loss)
            
            if self.config.train.gail.gradient_penalty_weight != 0:            
                # calculate gradient penalty
                gradient_penalty = self.disc.compute_gradient_penalty(expert_data)                
                loss += self.config.train.gail.gradient_penalty_weight * gradient_penalty
            

            self.disc_optimizer.zero_grad()
            loss.backward()
            self.disc_optimizer.step()

            agent_acc  = ((agent_prob < 0).float().mean().item())
            expert_acc = ((expert_prob > 0).float().mean().item())

            discriminator_losses.append(loss.item())
            agent_accuracies.append(agent_acc)
            expert_accuracies.append(expert_acc)
            gradient_penalties.append(gradient_penalty.item())
        
        return {
            
            "train_gail/discrim_loss": discriminator_losses,
            "train_gail/agent_accuracy": agent_accuracies,
            "train_gail/expert_accuracy": expert_accuracies,
            "train_gail/gradient_penalty": gradient_penalties,
        }
    
    def prepare_data(self, next_state, done):

        # Estimate next state value for gae
        with torch.no_grad():
            if self.config.train.observation_normalizer:
                next_state[:,  :self.config.env.state_dim] = self.obs_normalizer(next_state[:,  :self.config.env.state_dim])
            if self.config.train.goal_normalizer:
                next_state[:, self.config.env.state_dim: ] = self.goal_normalizer(next_state[:, self.config.env.state_dim: ])
            next_value = self.critic(torch.Tensor(next_state).to(self.device))
            next_value = next_value.flatten()
        
        data = self.memory.compute_gae_and_get(next_state, next_value, done)
          
        s        = data['state'].float()
        ns       = data['next_state'].float()
        a        = data['action']
        logp     = data['logprob'].float()
        v_target = data['v_target'].float()
        adv      = data['advant'].float()
        v        = data['value'].float()
                
        return s, ns, a, logp, adv, v_target, v
    
    def convert_data_in_disc_form(self, data):
        
        def get_data_by_id(d, id):
            p = 5 * 3 * id
            return d[:, p: p + 15]
        
        obs = []
        end_effector_id = self.skeleton_info["end_effector_id"]
        dofs = DOFS["Humanoid"]
        
        # 1. root linear velocity
        obs.append(get_data_by_id(data, 0)[:, 9:12])
        # 2. root angular velocity
        obs.append(get_data_by_id(data, 0)[:, 12:15])
        
        for key, _ in dofs[2:]:
            # 3. normal vector
            obs.append(get_data_by_id(data, key)[:, 3:6])
            # 4. tangent vector
            obs.append(get_data_by_id(data, key)[:, 6:9])
            # 5. linear velocity
            obs.append(get_data_by_id(data, key)[:, 9:12])
            # 6. angular velocity
            obs.append(get_data_by_id(data, key)[:, 12:15]) 
            
        # 7. end effector position
        for key in end_effector_id:
            obs.append(get_data_by_id(data, key)[:, 0:3])   
        
        obs = torch.concat(obs, dim=1)
        return obs
    def prepare_data_for_discriminator(self, data):
        
        # shape of observastion from env
        # [[pos_i, nor_i, tan_i, lin_vel_i, ang_vel_i] for i in range(15)]
        s, ns = data[:2]        
        s = self.convert_data_in_disc_form(s)
        ns = self.convert_data_in_disc_form(ns)
                
        return s, ns
    
    def optimize(self, next_state, done):     
                    
        # ------------- Preprocessing for optimizing-------------
        
        with self.timer_manager.get_timer("Prepare data"):
            data = self.prepare_data(next_state, done)
            
        # ------------- Optimizing-------------
        
        metrics = defaultdict(list)
        
        with self.timer_manager.get_timer("Backpropagation"):                
            if self.config.train.gail.use:
                for _ in range(self.config.train.gail.epochs):
                    with self.timer_manager.get_timer("Prepare data for discriminator"):  
                        disc_data = self.prepare_data_for_discriminator(data)
                    
                    with self.timer_manager.get_timer("Discrimniator optimize per epoch"):  
                        disc_metric = self.optimize_discriminator(disc_data)
                    
                    for k, v in disc_metric.items():
                        metrics[k].extend(v)
                    
            for _ in range(self.config.train.ppo.epochs):
                with self.timer_manager.get_timer("Policy optimize per epoch"):  
                    policy_metric = self.optimize_policy(data)
                
                for k, v in policy_metric.items():
                    metrics[k].extend(v)           
        
        # ------------- Update other parameters
        
        # action std decaying
        while self.timesteps > self.next_action_std_decay_step:
            self.next_action_std_decay_step += self.config.actor.action_std_decay_freq
            self.actor.action_decay(
                self.config.actor.action_std_decay_rate,
                self.config.actor.min_action_std
            )
            
        # scheduling learning rate
        if self.config.train.scheduler:
            self.actor_scheduler.step()
            self.critic_scheduler.step()
            if self.config.train.gail.use:
                self.disc_scheduler.step()
            
        return metrics
    
    def add_to_buffer(self, agent_id, state, next_state, action, reward, value, logprob, done):
        
        episodic_rewards = []
        style_rewards = []
        durations = []
        
        # update episodic_reward
        self.episodic_reward[agent_id] += reward
        self.duration[agent_id] += 1
        
        if self.config.train.gail.use:
            with torch.no_grad():
                d_state = self.convert_data_in_disc_form(state)
                d_next_state = self.convert_data_in_disc_form(torch.from_numpy(next_state).to(self.device, dtype=torch.float))
                s_reward = self.disc.get_reward(d_state, d_next_state)
                s_reward = s_reward.squeeze(1).cpu().numpy()    
                s_reward = s_reward * self.config.train.style_reward_scale   
            self.style_reward[agent_id] += s_reward
            
            total_reward = self.w_g * reward + self.w_s * s_reward
        else:
            total_reward = reward

        if self.config.train.reward_scaler:
            total_reward = self.reward_scaler(agent_id, total_reward, done)
            
        self.memory.store(
            agent_id=agent_id,
            state=state,
            action=action,
            reward=total_reward,
            done=done,
            value=value,
            logprob=logprob
        )
        
        for idx, d in zip(agent_id, done):
            if d:
                episodic_rewards.append(self.episodic_reward[idx])
                style_rewards.append(self.style_reward[idx])
                durations.append(self.duration[idx])    

                self.episodic_reward[idx] = 0
                self.style_reward[idx] = 0
                self.duration[idx] = 0   
                
        return episodic_rewards, style_rewards, durations 
    
        
    def collect_trajectory(self, envs, state, done):
        
        episodic_rewards = []
        style_rewards = []
        durations = []
        
        for _ in range(0, self.config.train.max_episode_len):                        
            # ------------- Collect Trajectories -------------
            '''
            Actor-Critic symbol's information
            ===========  ==========================  ==================
            Symbol       Shape                       Type
            ===========  ==========================  ==================
            action       (num_envs,)                 torch.Tensor
            logprobs     (num_envs,)                 torch.Tensor
            ent          (num_envs,)                 torch.Tensor
            values       (num_envs, 1)               torch.Tensor
            ===========  ==========================  ==================
            '''

                
            with torch.no_grad():
                if self.config.train.observation_normalizer:
                    state[:,  :self.config.env.state_dim] = self.obs_normalizer(state[:,  :self.config.env.state_dim])
                if self.config.train.goal_normalizer:
                    state[:, self.config.env.state_dim: ] = self.goal_normalizer(state[:, self.config.env.state_dim: ])
                state = torch.from_numpy(state).to(self.device, dtype=torch.float)
                action, logprobs, _ = self.actor(state)
                values = self.critic(state)
                values = values.flatten() # reshape shape of the value to (num_envs,)
                
            next_state, reward, terminated, truncated, info = envs.step(np.clip(action.cpu().numpy(), envs.action_space.low, envs.action_space.high))

            if len(info['terminal_steps']) > 0:
                agent_id =info['terminal_steps'].agent_id
                ns, r, te, tr = info['terminal_state']
                
                e, s, d = self.add_to_buffer(agent_id=info['terminal_steps'].agent_id,
                                             state=state[agent_id],
                                             next_state=ns,
                                             action=action[agent_id],
                                             reward=r,
                                             value=values[agent_id],
                                             logprob=logprobs[agent_id],
                                             done=te + tr)
                episodic_rewards.extend(e)
                style_rewards.extend(s)
                durations.extend(d)
            
            while len(info['decision_steps']) == 0:                
                empty_ac = np.empty((0, *envs.action_space.shape))   
                next_state, reward, terminated, truncated, info = envs.step(empty_ac)

                if len(info['terminal_steps']) > 0:
                    agent_id = info['terminal_steps'].agent_id
                    ns, r, te, tr = info['terminal_state']
                
                    e, s, d = self.add_to_buffer(agent_id=info['terminal_steps'].agent_id,
                                                state=state[agent_id],
                                                next_state=ns,
                                                action=action[agent_id],
                                                reward=r,
                                                value=values[agent_id],
                                                logprob=logprobs[agent_id],
                                                done=te + tr)
                    episodic_rewards.extend(e)
                    style_rewards.extend(s)
                    durations.extend(d)


            self.timesteps += self.config.env.num_envs
            
            # add experience to the memory   
            e, s, d = self.add_to_buffer(agent_id=info['decision_steps'].agent_id,
                               state=state,
                               next_state=next_state,
                               action=action,
                               reward=reward,
                               value=values,
                               logprob=logprobs,
                               done=terminated + truncated)  
            episodic_rewards.extend(e)
            style_rewards.extend(s)
            durations.extend(d)
            
            done = terminated + truncated
            
            # update state
            state = next_state  
            
        return {
            "train/score": episodic_rewards,
            "train/style_reward": style_rewards,
            "train/duration": durations
        }, next_state, done
        
    def train(self, envs, exp_name=None):
    
        start_time = datetime.now().replace(microsecond=0)
        
        # ==== Create an experiment directory to record training data ============
        
        self.config.experiment_name = f"exp{get_cur_time_code()}" if exp_name is None else exp_name
        self.config.experiment_path = os.path.join(self.config.checkpoint_path, self.config.experiment_name)

        # If an existing experiment has the same name, add a number to the end of the path.
        while os.path.exists(self.config.experiment_path):
            exp_name  = self.config.experiment_path[len(self.config.checkpoint_path) + 1:]
            exp_split = exp_name.split("_")

            try:
                exp_num  = int(exp_split[-1]) + 1
                exp_name = f"{'_'.join(exp_split[:max(1, len(exp_split) - 1)])}_{str(exp_num)}"
            except:
                exp_name = f"{exp_name}_0"

            self.config.experiment_name = exp_name
            self.config.experiment_path = os.path.join(self.config.checkpoint_path, self.config.experiment_name)
        os.makedirs(self.config.experiment_path, exist_ok=True)
        logger.addHandler( logging.FileHandler(os.path.join(self.config.experiment_path, f"running_train_log.log")))
        
                
        # ===== For logging training state ============
        
        writer_path     = os.path.join( self.config.experiment_path, 'runs')
        self.writer     = SummaryWriter(writer_path)

        self.episodic_reward = np.zeros(self.config.env.num_envs)
        self.duration        = np.zeros(self.config.env.num_envs)
        self.style_reward    = np.zeros(self.config.env.num_envs)
        best_score      = -1e9
        
        # ==== Make rollout buffer
        self.memory = UnityReplayBuffer(   
            num_of_agents=self.config.env.num_envs,
            gamma=self.config.train.gamma,
            tau=self.config.train.tau,
            device=self.device,
        )     

        '''
        Environment symbol's information
        ===========  ==========================  ==================
        Symbol       Shape                       Type
        ===========  ==========================  ==================
        state        (num_envs, (obs_space))     numpy.ndarray
        reward       (num_envs,)                 numpy.ndarray
        term         (num_envs,)                 numpy.ndarray
        trunc        (num_envs,)                 numpy.ndarray
        done         (num_envs,)                 numpy.ndarray
        ===========  ==========================  ==================
        '''
        next_state, _  = envs.reset() #envs.reset(seed=self.config.seed)
        done = np.zeros(self.config.env.num_envs)
        self.next_action_std_decay_step = self.config.actor.action_std_decay_freq        
        
        print(f"================ Start training ================")
        print(f"========= Exp name: {self.config.experiment_name} ==========")
        while self.timesteps < self.config.train.total_timesteps:
            
            with self.timer_manager.get_timer("Total"):
                with self.timer_manager.get_timer("Collect Trajectory"):
                    step_metrics, next_state, done = self.collect_trajectory(envs, next_state, done)
                        
                with self.timer_manager.get_timer("Optimize"):  
                    optimize_metrics = self.optimize(next_state, done)  
                    
            step_metrics.update(optimize_metrics)
            
            # ------------- Logging training state -------------
            
            # Writting for tensorboard    
            for k, v in step_metrics.items():
                avg = np.mean(v)
                self.writer.add_scalar(k, avg, self.timesteps)                  
        
            if self.config.train.scheduler:
                self.writer.add_scalar(f"train/actor_learning_rate", self.actor_scheduler.get_lr()[0], self.timesteps)
                self.writer.add_scalar(f"train/critic_learning_rate", self.critic_scheduler.get_lr()[0], self.timesteps)
                if self.config.train.gail.use:
                    self.writer.add_scalar(f"train_gail/learning_rate", self.disc_scheduler.get_lr()[0], self.timesteps)

            # Printing for console
            remaining_num_of_optimize   = int(math.ceil((self.config.train.total_timesteps - self.timesteps) /
                                                        (self.config.env.num_envs * self.config.train.max_episode_len)))
            remaining_training_time_min = int(self.timer_manager.get_timer('Total').get() * remaining_num_of_optimize // 60)
            remaining_training_time_sec = int(self.timer_manager.get_timer('Total').get() * remaining_num_of_optimize % 60)                                      

            avg_score       = np.round(np.mean(step_metrics['train/score']), 4)
            std_score       = np.round(np.std(step_metrics['train/score']), 4)
            avg_duration    = np.round(np.mean(step_metrics['train/duration']), 4)
            
            logger.info(f"[{datetime.now().replace(microsecond=0) - start_time}] {self.timesteps}/{self.config.train.total_timesteps} - score: {avg_score} +-{std_score} \t duration: {avg_duration}")
            for k, v in self.timer_manager.timers.items():
                logger.info(f"\t {k} time: {round(v.clear(), 5)} sec")
            logger.info(f"\t Estimated training time remaining: {remaining_training_time_min} min {remaining_training_time_sec} sec")

            # Save best model
            if avg_score >= best_score:
                self.save(f'best')
                best_score = avg_score

            self.save(f"timesteps{self.timesteps}")
        
    
    def eval(self, env, max_ep_len, num_episodes=10):  
        
        self.episodic_reward = np.zeros(self.config.env.num_envs)
        self.duration        = np.zeros(self.config.env.num_envs)

        rewards = []
        durations = []

        for episode in range(num_episodes):

            episodic_rewards = []
            durations = []
            state, _ = env.reset()

            for t in range(max_ep_len):

                # ------------- Collect Trajectories -------------

                with torch.no_grad():
                    if self.config.train.observation_normalizer:
                        state[:,  :self.config.env.state_dim] = self.obs_normalizer(state[:,  :self.config.env.state_dim])
                    if self.config.train.goal_normalizer:
                        state[:, self.config.env.state_dim: ] = self.goal_normalizer(state[:, self.config.env.state_dim: ])
                    state = torch.from_numpy(state).to(self.device, dtype=torch.float)
                    action, _, _ = self.actor(state)
                    
                next_state, reward, terminated, truncated, info = env.step(np.clip(action.cpu().numpy(), env.action_space.low, env.action_space.high))

                
                if len(info['terminal_steps']) > 0:
                    agent_id = info['terminal_steps'].agent_id
                    _, r, te, tr = info['terminal_state']
                    print(agent_id, r.shape)
                    self.episodic_reward[agent_id] += r
                    self.duration[agent_id] += 1
        
                    for idx, d in zip(agent_id, te + tr):
                        if d:
                            episodic_rewards.append(self.episodic_reward[idx])
                            durations.append(self.duration[idx])    

                            self.episodic_reward[idx] = 0
                            self.duration[idx] = 0  
                           
                while len(info['decision_steps']) == 0:    
                    empty_ac = np.empty((0, *env.action_space.shape))   
                    next_state, reward, terminated, truncated, info = env.step(empty_ac)                        
                
                    if len(info['terminal_steps']) > 0:
                        agent_id =info['terminal_steps'].agent_id
                        ns, r, te, tr = info['terminal_state']
                        self.episodic_reward[agent_id] += r
                        self.duration[agent_id] += 1
            
                        for idx, d in zip(agent_id, terminated + truncated):
                            if d:
                                episodic_rewards.append(self.episodic_reward[idx])
                                durations.append(self.duration[idx])    

                                self.episodic_reward[idx] = 0
                                self.duration[idx] = 0  

                self.episodic_reward += reward
                self.duration += 1
                # update state
                state = next_state

            rewards.append(episodic_rewards)
            durations.append(durations)
            logger.info(f"Episode {episode}: score - {np.mean(episodic_rewards)} duration - {t}")

        avg_reward = np.mean(rewards)
        avg_duration = np.mean(durations)
        logger.info(f"Average score {avg_reward}, duration {avg_duration} on {num_episodes} games")               
        env.close()