import os
import sys
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
# Ensure utils path is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
from wrapper import get_env_wrapper

# TD3 Hyperparameters
ACTOR_LR = 1e-3
CRITIC_LR = 1e-3  # Unused in Stable Baselines3 TD3 (uses single lr param)
BUFFER_SIZE = 100000
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.005

class TD3Agent:
    def __init__(self, symbol, log_dir):
        self.env = make_vec_env(get_env_wrapper(symbol), n_envs=1)
        self.model = TD3(
            "MlpPolicy",
            self.env,
            verbose=1,
            buffer_size=BUFFER_SIZE,
            learning_rate=ACTOR_LR,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            tau=TAU,
            tensorboard_log=log_dir
        )

    def train(self, timesteps):
        self.model.learn(total_timesteps=timesteps)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = TD3.load(path, env=self.env)
