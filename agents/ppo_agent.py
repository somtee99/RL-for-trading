from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import os
import sys
import numpy as np

# Ensure utils path is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
from wrapper import get_env_wrapper

# Hyperparameters
PI_LAYERS = [64, 64] # Policy (Actor) network layers 
VF_LAYERS = [64, 64] # Value (Critic) network layers

LEARNING_RATE = 0.00005   
N_EPOCHS = 10 # Number of epochs to train the model per update
N_ENVS = 4 # Number of parallel environments (for vectorized training)  
N_STEPS = 1024 # Number of steps to run per update (batch size)      
BATCH_SIZE = 128 # Minibatch size for training     
GAMMA = 0.99 # Discount factor for rewards           
GAE_LAMBDA = 0.95 # Factor for trade-off of bias vs variance for Generalized Advantage Estimation       
CLIP_RANGE = 0.2 # Clipping parameter for PPO     
ENT_COEF = 0.05 # Exploration coefficient for entropy regularization          
VF_COEF = 1.0  # Coefficient for the value function loss term in the loss function
# Policy kwargs (separate networks for policy and value)
POLICY_KWARGS = dict(
    net_arch=dict(pi=PI_LAYERS, vf=VF_LAYERS)
)    


class PPOAgent:
    def __init__(self, symbol, log_dir):

        env = make_vec_env(get_env_wrapper(symbol), n_envs=N_ENVS)
        self.env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0)

        self.model = PPO(
            "MlpPolicy",
            self.env,
            verbose=1,
            learning_rate=LEARNING_RATE,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            clip_range=CLIP_RANGE,
            ent_coef=ENT_COEF,
            vf_coef=VF_COEF,
            tensorboard_log=log_dir,
            policy_kwargs=POLICY_KWARGS,
            n_epochs=N_EPOCHS
        )

    def train(self, timesteps):
        self.model.learn(total_timesteps=timesteps)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = PPO.load(path, env=self.env)
