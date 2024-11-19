
import logging
import itertools

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
from gymnasium import error, spaces

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import DecisionSteps, TerminalSteps, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel


logger = logging.getLogger(__name__)

class UnityGymException(error.Error):
    """
    Any error related to the gym wrapper of ml-agents.
    """

    pass

class ActionFlattener:
    """
    Flattens branched discrete action spaces into single-branch discrete action spaces.
    """

    def __init__(self, branched_action_space):
        """
        Initialize the flattener.
        :param branched_action_space: A List containing the sizes of each branch of the action
        space, e.g. [2,3,3] for three branches with size 2, 3, and 3 respectively.
        """
        self._action_shape = branched_action_space
        self.action_lookup = self._create_lookup(self._action_shape)
        self.action_space = spaces.Discrete(len(self.action_lookup))

    @classmethod
    def _create_lookup(self, branched_action_space):
        """
        Creates a Dict that maps discrete actions (scalars) to branched actions (lists).
        Each key in the Dict maps to one unique set of branched actions, and each value
        contains the List of branched actions.
        """
        possible_vals = [range(_num) for _num in branched_action_space]
        all_actions = [list(_action) for _action in itertools.product(*possible_vals)]
        # Dict should be faster than List for large action spaces
        action_lookup = {
            _scalar: _action for (_scalar, _action) in enumerate(all_actions)
        }
        return action_lookup

    def lookup_action(self, action):
        """
        Convert a scalar discrete action into a unique set of branched actions.
        :param action: A scalar value representing one of the discrete actions.
        :returns: The List containing the branched actions.
        """
        return self.action_lookup[action]


class VertorizedUnityEnv(gym.Env):
    
    def __init__(self, 
                 env_filename, worker_id=0,
                 flatten_branched = False,
                 action_space_seed = None,
                 no_graphics=False, width=80, height=80, 
                 time_scale=20.0):
        base_port = 5005
        if env_filename is None:
            base_port = UnityEnvironment.DEFAULT_EDITOR_PORT
            no_graphics = True

        print(f"Load env from {'editor'if env_filename is None else env_filename}")
        channel = EngineConfigurationChannel()
        self._env = UnityEnvironment(env_filename, worker_id, base_port, no_graphics=no_graphics, side_channels=[channel])
        if not no_graphics:
            channel.set_configuration_parameters(width=width, height=height)
        channel.set_configuration_parameters(time_scale=time_scale)
                
        if not self._env.behavior_specs:
            self._env.step()
            
        self.game_over = False
        self._previous_decision_step = None     
        
        # Currently only one behavior available. no more
        self.name = list(self._env.behavior_specs.keys())[0]
        self.group_spec = self._env.behavior_specs[self.name]
        
        if self._get_n_vec_obs() == 0:
            raise UnityGymException(
                "There are no observations provided by the environment."
            )
            
        self._env.reset()
        decision_steps, _ = self._env.get_steps(self.name)  
        self.num_agents = len(decision_steps)
        
        self._previous_decision_step = decision_steps
        self._flattener = None
        # Set action spaces
        if self.group_spec.action_spec.is_discrete():
            self.action_size = self.group_spec.action_spec.discrete_size
            branches = self.group_spec.action_spec.discrete_branches
            if self.group_spec.action_spec.discrete_size == 1:
                self.action_space = spaces.Discrete(branches[0])
            else:
                if flatten_branched:
                    self._flattener = ActionFlattener(branches)
                    self.action_space = self._flattener.action_space
                else:
                    self.action_space = spaces.MultiDiscrete(branches)

        elif self.group_spec.action_spec.is_continuous():
            if flatten_branched:
                logger.warning(
                    "The environment has a non-discrete action space. It will "
                    "not be flattened."
                )

            self.action_size = self.group_spec.action_spec.continuous_size
            high = np.array([1] * self.group_spec.action_spec.continuous_size)
            self.action_space = spaces.Box(-high, high, dtype=np.float32)
        else:
            raise UnityGymException(
                "The gym wrapper does not provide explicit support for both discrete "
                "and continuous actions."
            )

        if action_space_seed is not None:
            self.action_space.seed(action_space_seed)
            
        
        # Set observations space
        high = np.array([np.inf] * self._get_vec_obs_size())            
        self.observation_space = spaces.Box(-high, high, dtype=np.float32) 
        
        
    def reset(self):
        
        self._env.reset()
        
        decision_steps, _ = self._env.get_steps(self.name)
        obs, _, _, _ = self._postprocess_step(decision_steps)
        
        info = {
            'decision_steps': decision_steps,
            'terminal_steps': [],
            'terminal_state': []
        }
        
        return obs, info
    
    
    def step(self, action : np.array):
        
        if len(action.shape) != 2:
            raise UnityGymException(
                "Action shape must be two dimension"
            )
        
        # Process action
        if self._flattener is not None:
            action = self._flattener.lookup_action(action)
            
        action_tuple = ActionTuple()
        if self.group_spec.action_spec.is_continuous():
            action_tuple.add_continuous(action)
        else:
            action_tuple.add_discrete(action)
        self._env.set_actions(self.name, action_tuple)            
        self._env.step()
        
        # Postprocess steps
        decision_steps, terminal_steps = self._env.get_steps(self.name)
        obs, reward, term, trunc = self._postprocess_step(decision_steps)
        
        terminal_state = self._postprocess_step(terminal_steps)
        info = {
            'decision_steps': decision_steps,
            'terminal_steps': terminal_steps,
            'terminal_state': terminal_state
        }
        
        return obs, reward, term, trunc, info
    
    def _postprocess_step(self, steps : Union[DecisionSteps, TerminalSteps]):
        
        obs : List[np.array] = []
        reward : List[float] = []
        term : List[int] = []
        trunc : List[int] = []       
        
        for i in range(len(steps.reward)):
            obs.append(np.concatenate([o[i, :] for o in steps.obs]))
            reward.append(steps.reward[i])
            
            if isinstance(steps, TerminalSteps):
                interrupred = steps.interrupted[i]
                term.append(not interrupred)
                trunc.append(interrupred)
            else:
                term.append(0)
                trunc.append(0)
            
            
        obs = np.array(obs).astype(np.float32)
        reward = np.array(reward).astype(np.float32)
        term = np.array(term).astype(np.bool8)
        trunc = np.array(trunc).astype(np.bool8)

        return obs, reward, term, trunc
           
    
    def _get_vec_obs_size(self) :
        result = 0
        for obs_spec in self.group_spec.observation_specs:
            if len(obs_spec.shape) == 1:
                result += obs_spec.shape[0]
        return result
    
    def _get_n_vec_obs(self) :
        result = 0
        for obs_spec in self.group_spec.observation_specs:
            if len(obs_spec.shape) == 1:
                result += 1
        return result
                

if __name__ == "__main__":
    
   # from ..replay_buffer import UnityReplayBuffer
    
    env = VertorizedUnityEnv(None)
   # buffer = UnityReplayBuffer(env.agen)
    
    
    state, _ = env.reset()
    
    action_space = env.action_space
    obs_space = env.observation_space
    
    print(f"Action: {action_space}")
    print(f"Obs: {obs_space}")
    while True:
        
        
        action = np.stack([env.action_space.sample() for _ in range(len(state))])        
        state, reward, term, trunc, info = env.step(action)
        
        # add terminal state to buffer
        
        while len(info['decision_steps']) == 0:
            
            action = np.empty((0, *action_space.shape))      
            
            state, reward, term, trunc, info = env.step(action)
            
            # add terminal state to buffer
            if len(info['terminal_steps']) > 0:
                pass
                
            
        
            # print(f"Obs: {state.shape}")
            # print(f"reward: {reward.shape}")
            # print(f"info: {len(info['decision_steps'])} {len(info['terminal_steps'])}")
            # print(f"info: {info['decision_steps'].agent_id} {info['terminal_steps'].agent_id}")
            
            # print(f"Real Obs: {np.round(state[:, :5], 3)}")
        
    
        