import random
import numpy as np


class Agent:
    def __init__(self):
        pass

    def move(self, observation, board):
        pass

    def train(self, obs_old, act, rwd, obs_new):
        pass


class Random(Agent):
    """
    Basic Agent. Always returns a random move.
    """
    def __init__(self):
        super().__init__()

    def move(self, observation, board):
        return random.choice(np.flatnonzero(observation["action_mask"]))
