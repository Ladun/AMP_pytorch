
import os
import glob
import numpy as np
import logging
import shutil
import math
from datetime import datetime
from PIL import Image
from collections import deque

from omegaconf import OmegaConf
import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.policy import ActorCritic
from models.discriminator import Discriminator
from data.motion_dataset import MotionDataset
from scheduler import WarmupLinearSchedule
from replay_buffer import ReplayBuffer
from utils.general import (
    set_seed, pretty_config, get_cur_time_code,
    TimerManager, get_device
)
from utils.stuff import RewardScaler, ObservationNormalizer

logger = logging.getLogger(__name__)


def data_iterator(batch_size, given_data):
    # Simple mini-batch spliter

    ob, n_ob, ac, oldpas, adv, tdlamret, old_v = given_data
    total_size = len(ob)
    indices = np.arange(total_size)
    np.random.shuffle(indices)
    n_batches = total_size // batch_size
    for nb in range(n_batches):
        ind = indices[batch_size * nb : batch_size * (nb + 1)]
        yield ob[ind], n_ob[ind], ac[ind], oldpas[ind], adv[ind], tdlamret[ind], old_v[ind]      


class AMPAgent:
    def __init__(self, config):
        
        self.config = config
        self.device = get_device(config.device)
        
        set_seed(self.config.seed)
        rng_state, _ = gym.utils.seeding.np_random(self.config.seed)
        self.env_rng_state = rng_state
        
        # -------- Define models --------
        
        self.network = ActorCritic(config, self.device).to(self.device) 
        self.disc = Discriminator(config)
        
        # -------- Define train supporter ---------
               
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            **self.config.network.optimizer
        )
        
        self.disc_optimizer  = torch.optim.Adam([
            {'params': self.disc.parameters(),
            **config.gail.optimizer}
        ])
        
        if self.config.train.scheduler:
            self.scheduler = WarmupLinearSchedule(optimizer=self.optimizer,
                                                  warmup_steps=0,
                                                  max_steps=self.config.train.total_timesteps // (self.config.train.max_episode_len * self.config.env.num_envs))
            self.disc_scheduler = WarmupLinearSchedule(optimizer=self.disc_optimizer,
                                                       warmup_steps=0, 
                                                       max_steps=self.config.train.total_timesteps // (self.config.train.max_episode_len * self.config.env.num_envs))
                    
        # reward scaler: r / rs.std()
        if self.config.train.reward_scaler:
            self.reward_scaler = RewardScaler(self.config.env.num_envs, gamma=self.config.train.gamma)

        # observation scaler: (ob - ob.mean()) / (ob.std())
        if self.config.train.observation_normalizer:
            sp = (config.env.state_dim, ) if isinstance(config.env.state_dim, int) else list(config.env.state_dim)
            self.obs_normalizer = ObservationNormalizer(self.config.env.num_envs, sp)
            
        # Disriminator dataset
        self.motion_dataset = MotionDataset(self.config.expert.data_dir, 
                                            self.config.expert.asf_file,
                                            self.config.expert.mujoco_xml_file)
        
        
        self.timer_manager  = TimerManager()
        self.writer         = None
        self.memory         = None
        self.timesteps      = 0
        
        logger.info("----------- Config -----------")
        pretty_config(config, logger=logger)
        logger.info(f"Device: {self.device}")
    
    def save(self, ckpt_path):
        pass
    
    def load(self, ckpt_path):
        pass
     
    def optimize_policy(self, data):
        data_loader = data_iterator(self.config.policy.batch_size, data)   

        policy_losses   = []
        entropy_losses  = []
        value_losses    = []
        total_losses    = []
        
        c2 = self.config.train.ppo.coef_entropy_penalty
        for batch in data_loader:
            ob, ac, old_logp, adv, vtarg, old_v = batch
            adv = (adv - adv.mean()) / (adv.std() + 1e-7)

            # -------- Loss calculate --------

            # --- policy loss
            _, cur_logp, cur_ent, cur_v = self.network(ob, action=ac)
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

            # --- value loss

            if self.config.train.ppo.value_clipping:
                cur_v_clipped = old_v + (cur_v - old_v).clamp(-self.config.train.ppo.eps_clip, self.config.train.ppo.eps_clip)
                vloss1 = (cur_v - vtarg) ** 2 # F.smooth_l1_loss(cur_v, vtarg, reduction='none')
                vloss2 = (cur_v_clipped - vtarg) ** 2 # F.smooth_l1_loss(cur_v_clipped, vtarg, reduction='none')
                vf_loss = torch.max(vloss1, vloss2)
            else:
                vf_loss = (cur_v - vtarg) ** 2 #F.smooth_l1_loss(cur_v, vtarg, reduction='none')
                
            vf_loss = 0.5 * vf_loss.mean()

            # -------- Backward process --------

            c1 = self.config.train.ppo.coef_value_function
            c2 = self.config.train.ppo.coef_entropy_penalty

            total_loss = policy_surr + c2 * policy_ent + c1 * vf_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            if self.config.train.clipping_gradient:
                nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
            self.optimizer.step()

            # ---------- Record training loss data ----------
    
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
        expert_loader = DataLoader(self.dataset, 
                                   batch_size=self.config.train.gail.batch_size,
                                   shuffle=True)    
        agent_loader  = data_iterator(self.config.train.gail.batch_size, data)    
    
        loss_fn = nn.MSELoss()
        discriminator_losses = []
        learner_accuracies = []
        expert_accuracies = []
        
        self.disc.train()
        for expert_ob, expert_next_ob in expert_loader:
            ob, next_ob = next(agent_loader)[:2]

            agent_prob = self.disc(ob, next_ob)
            expert_prob = self.disc(expert_ob, expert_next_ob)

            agent_loss = loss_fn(agent_prob, -torch.ones_like(agent_prob))
            expert_loss = loss_fn(expert_prob, torch.ones_like(expert_prob))
        
            # maximize E_expert [ (D(s, s') - 1)^2] + E_agent [(D(s, s') + 1)^2]
            # TODO: add gradient_penalty
            loss = agent_loss + expert_loss
            discriminator_losses.append(loss.item())

            self.disc_optimizer.zero_grad()
            loss.backward()
            self.disc_optimizer.step()

            learner_acc = ((agent_prob >= 0.5).float().mean().item())
            expert_acc = ((expert_prob < 0.5).float().mean().item())

            learner_accuracies.append(learner_acc)
            expert_accuracies.append(expert_acc)
        
        return {
            
            "train_gail/discrim_loss": discriminator_losses,
            "train_gail/learner_accuracy": learner_accuracies,
            "train_gail/expert_accuracy": expert_accuracies
        }
    
    def prepare_data(self, next_state, done):

        # Estimate next state value for gae
        with torch.no_grad():
            if self.config.train.observation_normalizer:
                next_state = self.obs_normalizer(next_state)
            _, _, _, next_value = self.network(torch.Tensor(next_state).to(self.device))
            next_value = next_value.flatten()
        
        data = self.memory.compute_gae_and_get(next_state, next_value, done)
          
        s        = data['state'].float()
        a        = data['action']
        logp     = data['logprob'].float()
        v_target = data['v_target'].float()
        adv      = data['advant'].float()
        v        = data['value'].float()
        
        print(s.shape)
        ns = torch.cat([s, torch.from_numpy(ns).to(self.device).unsqueeze(0)], dim=0)
        ns = ns[1:]
        print(ns.shape)
        
        return s, ns, a, logp, v_target, adv, v
    
    def optimize(self, next_state, done):     
                    
        # ------------- Preprocessing for optimizin-------------
        with self.timer_manager.get_timer("Prepare data"):
            data = self.prepare_data(next_state, done)
            
        # ------------- Optimizing-------------
        
        metrics = {
            "train/policy_loss": [],
            "train/entropy_loss": [],
            "train/value_loss": [],
            "train/total_loss": [],
            
            "train_gail/discrim_loss": [],
            "train_gail/learner_accuracy": [],
            "train_gail/expert_accuracy": []
        }
        
        with self.timer_manager.get_timer("Backpropagation"):                
            for _ in range(self.config.train.gail.epoch):
                disc_metric = self.optimize_discriminator(data)
                
                for k, v in disc_metric.items():
                    metrics[k].extend(v)
                    
            for _ in range(self.config.train.ppo.optim_epochs):
                policy_metric = self.optimize_policy(data)
                
                for k, v in policy_metric.items():
                    metrics[k].extend(v)           
        
        # ------------- Update other parameters
        
        # action std decaying
        while self.timesteps > self.next_action_std_decay_step:
            self.next_action_std_decay_step += self.config.network.action_std_decay_freq
            self.network.action_decay(
                self.config.network.action_std_decay_rate,
                self.config.network.min_action_std
            )
            
        # scheduling learning rate
        if self.config.train.scheduler:
            self.scheduler.step()
            self.disc_scheduler.step()
            
        return metrics
    
    
    def collect_trajectory(self, envs, state, episodic_reward, duration):
        
        episodic_rewards = []
        durations = []
        
        w_g = self.config.train.coef_task_specific
        w_s = self.config.train.coef_task_agnostic      
        for t in range(0, self.config.train.max_episode_len):                        
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
                    state = self.obs_normalizer(state)
                state = torch.from_numpy(state).to(self.device, dtype=torch.float)
                action, logprobs, _, values = self.network(state)
                values = values.flatten() # reshape shape of the value to (num_envs,)
                
            next_state, reward, terminated, truncated, _ = envs.step(np.clip(action.cpu().numpy(), envs.action_space.low, envs.action_space.high))

            self.timesteps += self.config.env.num_envs

            # update episodic_reward
            episodic_reward += reward
            duration += 1
            
            style_reward = self.disc.get_reward(state, torch.from_numpy(state).to(self.device, dtype=torch.float))            
            print(f"Reward: {reward.shape}, {style_reward.shape}")
            reward = w_g * reward + w_s * style_reward

            if self.config.train.reward_scaler:
                reward = self.reward_scaler(reward, terminated + truncated)
                
            # add experience to the memory                    
            self.memory.store(
                state=state,
                action=action,
                reward=reward,
                done=done,
                value=values,
                logprob=logprobs
            )
            done = terminated + truncated

            for idx, d in enumerate(done):
                if d:
                    episodic_rewards.append(episodic_reward[idx])
                    durations.append(duration[idx])    
                    # irl_score_queue.append(irl_episodic_reward[idx])

                    episodic_reward[idx] = 0
                    duration[idx] = 0            
                    # irl_episodic_reward[idx] = 0
            
            # update state
            state = next_state  
            
        return {
            "train/score": episodic_rewards,
            "train/duration": durations
        }, next_state, done
        
    def train(self, envs, exp_name=None):
    
        # Set random state for reproducibility
        envs.np_random = self.env_rng_state
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

        episodic_reward = np.zeros(self.config.env.num_envs)
        duration        = np.zeros(self.config.env.num_envs)
        irl_episodic_reward = np.zeros(self.config.env.num_envs)
        best_score      = -1e9
        
        # ==== Make rollout buffer
        self.memory = ReplayBuffer(            
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
        done         (num_envs,)                 numpy.ndarray
        ===========  ==========================  ==================
        '''
        next_state, _  = envs.reset()
        done = np.zeros(self.config.env.num_envs)
        self.next_action_std_decay_step = self.config.network.action_std_decay_freq        
        
        print(f"================ Start training ================")
        print(f"========= Exp name: {self.config.experiment_name} ==========")
        while self.timesteps < self.config.train.total_timesteps:
            
            with self.timer_manager.get_timer("Total"):
                with self.timer_manager.get_timer("Collect Trajectory"):
                    step_metrics, next_state, done = self.collect_trajectory(envs, next_state, episodic_reward, duration)
                        
                with self.timer_manager.get_timer("Optimize"):  
                    optimize_metrics = self.optimize(next_state, done)  
                    
            step_metrics.update(optimize_metrics)
            
            # ------------- Logging training state -------------
            
            # Writting for tensorboard    
            for k, v in step_metrics:
                avg = np.mean(v)
                self.writer.add_scalar(k, avg, self.timesteps)                  
        
            if self.config.train.scheduler:
                for idx, lr in enumerate(self.scheduler.get_lr()):
                    self.writer.add_scalar(f"train/learning_rate{idx}", lr, self.timesteps)
                for idx, lr in enumerate(self.disc_scheduler.get_lr()):
                    self.writer.add_scalar(f"train_gail/learning_rate{idx}", lr, self.timesteps)

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
                logger.info(f"\t\t {k} time: {v.get()} sec")
            logger.info(f"\t\t Estimated training time remaining: {remaining_training_time_min} min {remaining_training_time_sec} sec")

            # Save best model
            if avg_score >= best_score:
                self.save(f'best', envs)
                best_score = avg_score

            self.save(f"timesteps{self.timesteps}", envs)
            
        
    
    def eval(self):
        pass