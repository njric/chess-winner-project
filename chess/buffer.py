import random
import torch
from collections import deque

from config import CFG

class ReplayBuffer():

    def __init__(self):
        self.buffer = deque(maxlen=CFG.buffer_size)

    def set(self, obs):
        """
        Store a single new observation in the replay buffer.
        """

        old, act, new, rwd = obs
        old = torch.tensor(old).float()
        new = torch.tensor(new).float()

        self.buffer.append((old, act, new, rwd))

    def get(self):
        """
        Get a minibatch of observations of size CFG.batch_size.
        """
        if len(self.buffer) < CFG.batch_size:
            raise Exception("Not enough data in replay buffer.")

        batch = random.sample(self.buffer, CFG.batch_size)
        old, act, new, rwd = zip(*batch)

        old = torch.stack(old)
        new = torch.stack(new)
        act = torch.tensor(act).unsqueeze(1)
        rwd = torch.tensor(rwd)

        return old, act, new, rwd

BUF = ReplayBuffer()