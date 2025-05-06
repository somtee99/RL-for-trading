import tensorflow as tf
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env
from agents.utils import get_env_wrapper

# Hyperparameters
ACTOR_LR = 1e-3
CRITIC_LR = 1e-3
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.005

class DDPGAgent:
    def __init__(self, env, log_dir):
        self.env = make_vec_env(lambda: get_env_wrapper(env), n_envs=1)
        self.model = DDPG("MlpPolicy", self.env, verbose=1, buffer_size=BUFFER_SIZE,
                          learning_rate=ACTOR_LR, batch_size=BATCH_SIZE,
                          gamma=GAMMA, tau=TAU, tensorboard_log=log_dir)

    def train(self, timesteps):
        self.model.learn(total_timesteps=timesteps)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = DDPG.load(path, env=self.env)