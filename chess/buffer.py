import random
import torch
from collections import deque

from config import CFG

class ReplayBuffer():

    def __init__(self):
        self.buffer = deque(maxlen=CFG.buffer_size)

    def len(self):
        return len(self.buffer)

    def set(self, obs):
        """
        Store a single new observation in the replay buffer.
        """

        old, act, rwd, new = obs

        # We deal with terminal states by giving them the same value as the previous state.
        # This ensures that CFG.gamma * V(new) - V(old) ~ 0
        if new is None:
            new = old

        old = torch.permute(torch.tensor(old["observation"]).float(), (2, 0, 1))
        new = torch.permute(torch.tensor(new["observation"]).float(), (2, 0, 1))

        self.buffer.append((old, act, rwd, new))

    def get(self):
        """
        Get a minibatch of observations of size CFG.batch_size.
        """
        if len(self.buffer) < CFG.batch_size:
            raise Exception("Not enough data in replay buffer.")

        batch = random.sample(self.buffer, CFG.batch_size)
        old, act, rwd, new = zip(*batch)

        old = torch.stack(old)
        new = torch.stack(new)
        act = torch.tensor(act).unsqueeze(1)
        rwd = torch.tensor(rwd)

        return old, act, rwd, new

BUF = ReplayBuffer()