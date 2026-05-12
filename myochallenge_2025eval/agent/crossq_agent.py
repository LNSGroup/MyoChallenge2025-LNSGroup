'''
This script defines how to load a JAX-based CrossQ agent for the MyoChallenge 2025.
'''
import os
import pickle
import json
import jax
import jax.numpy as jnp
import numpy as np
import gymnasium as gym
import myosuite
from crossq.crossq import create_crossq_net
from tabletennis_wrapper import tabletennisWrapper

class CrossQAgent:
    '''A wrapper for a JAX-based CrossQ agent.'''
    def __init__(self, config_path, policy_path, obs_dim, act_dim):
        with open(config_path, 'r') as f:
            config = json.load(f)

        hidden_num = config['hidden_num']
        hidden_dim = config['hidden_dim']
        hidden_sizes = [hidden_dim] * hidden_num
        
        key = jax.random.key(0)
        self.agent, _ = create_crossq_net(key, obs_dim=obs_dim, act_dim=act_dim, hidden_sizes=hidden_sizes)

        with open(policy_path, 'rb') as f:
            self.policy_params = pickle.load(f)

        @jax.jit
        def _policy_fn(policy_params, obs):
            policy_param, policy_state = policy_params
            if obs.ndim == 1:
                obs = jnp.expand_dims(obs, axis=0)
            action = self.agent.get_deterministic_action(policy_param, policy_state, obs).clip(-1, 1)
            return action.flatten()
        self.policy_fn = _policy_fn

    def __call__(self, obs):
        action = self.policy_fn(self.policy_params, obs)
        return np.array(action)

def load_tabletennis(checkpoint_path):
    '''Loads the table tennis agent.'''
    config_path = os.path.join(checkpoint_path, 'PingPongv2.json')
    policy_path = os.path.join(checkpoint_path, 'logs', 'pingpong_crossq.pkl')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    dummy_env = gym.make('myoChallengeTableTennisP2-v0', obs_keys=config['single_env_kwargs']['obs_keys'])
    wrapped_dummy_env = tabletennisWrapper(dummy_env)
    act_dim = wrapped_dummy_env.action_space.shape[0]
    obs_dim = wrapped_dummy_env.observation_space.shape[0]
    
    wrapped_dummy_env.close()

    return CrossQAgent(config_path=config_path, policy_path=policy_path, obs_dim=obs_dim, act_dim=act_dim)