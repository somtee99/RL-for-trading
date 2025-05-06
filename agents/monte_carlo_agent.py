import numpy as np
from collections import defaultdict
import random

# Hyperparameters
DISCOUNT_FACTOR = 0.99
EPSILON = 0.1  # for ε-greedy action selection
LEARNING_RATE = 0.1

class MonteCarloOptimizerAgent:
    def __init__(self, action_space):
        self.action_space = action_space
        self.q_table = defaultdict(lambda: np.zeros(self.action_space.n))
        self.returns_sum = defaultdict(lambda: np.zeros(self.action_space.n))
        self.returns_count = defaultdict(lambda: np.zeros(self.action_space.n))
        self.policy = defaultdict(lambda: np.ones(self.action_space.n) / self.action_space.n)

    def choose_action(self, state):
        """ ε-greedy policy """
        if random.random() < EPSILON:
            return self.action_space.sample()
        else:
            return np.argmax(self.q_table[state])

    def update_policy(self, state):
        """ Improve policy by setting greedy action """
        best_action = np.argmax(self.q_table[state])
        self.policy[state] = np.ones(self.action_space.n) * (EPSILON / self.action_space.n)
        self.policy[state][best_action] += 1.0 - EPSILON

    def update(self, episode):
        """ Run First-Visit Monte Carlo updates for a complete episode """
        G = 0
        visited_state_actions = set()

        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = DISCOUNT_FACTOR * G + reward
            if (state, action) not in visited_state_actions:
                self.returns_sum[state][action] += G
                self.returns_count[state][action] += 1
                self.q_table[state][action] = (
                    self.returns_sum[state][action] / self.returns_count[state][action]
                )
                self.update_policy(state)
                visited_state_actions.add((state, action))
