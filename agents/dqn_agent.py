from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from agents.utils import get_env_wrapper

# Hyperparameters
LEARNING_RATE = 1e-3
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 32
GAMMA = 0.99
EXPLORATION_FRACTION = 0.1
EXPLORATION_FINAL_EPS = 0.05
TARGET_UPDATE_INTERVAL = 250

class DQNAgent:
    def __init__(self, env, log_dir):
        self.env = make_vec_env(lambda: get_env_wrapper(env), n_envs=1)
        self.model = DQN(
            "MlpPolicy",
            self.env,
            verbose=1,
            learning_rate=LEARNING_RATE,
            buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            exploration_fraction=EXPLORATION_FRACTION,
            exploration_final_eps=EXPLORATION_FINAL_EPS,
            target_update_interval=TARGET_UPDATE_INTERVAL,
            tensorboard_log=log_dir
        )

    def train(self, timesteps):
        self.model.learn(total_timesteps=timesteps)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = DQN.load(path, env=self.env)
