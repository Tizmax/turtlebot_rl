import warnings

import gymnasium as gym

from envs.wrappers.tensor import TensorWrapper
from envs.tb2_kobuki import make_env as make_tb2_kobuki_env


warnings.filterwarnings('ignore', category=DeprecationWarning)


def make_env(cfg):
	"""
	Make an environment for TD-MPC2 experiments.
	"""
	gym.logger.min_level = 40
	env = make_tb2_kobuki_env(cfg)
	env = TensorWrapper(env)
	try: # Dict
		cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
	except: # Box
		cfg.obs_shape = {cfg.get('obs', 'state'): env.observation_space.shape}
	cfg.action_dim = env.action_space.shape[0]
	cfg.episode_length = env.max_episode_steps
	cfg.seed_steps = max(1000, 5*cfg.episode_length)
	return env
