"""
Agent
"""

import collections
import random
import subprocess

import chess.engine
import numpy as np
import torch

from network import A2CNet
from config import CFG
from buffer import BUF
from utils import move_to_act


class Agent:
    def __init__(self):
        pass

    def move(self, observation, board):
        pass

    def feed(self, old, act, rwd, new):
        pass


class StockFish(Agent):
    """
    Agent that uses the Stockfish chess engine.
    """

    def __init__(self):
        super().__init__()

        SF_dir = (
            subprocess.run(["which", "stockfish"], stdout=subprocess.PIPE)
            .stdout.decode("utf-8")
            .strip("\n")
        )
        ## If subprocess cannot find stockfish, move it to .direnv and switch to method :
        # import os
        # SF_dir = os.path.join(os.path.dirname(__file__), '../.direnv/stockfish')

        self.engine = chess.engine.SimpleEngine.popen_uci(SF_dir)
        self.limit = chess.engine.Limit(time=0.1)
        ## TODO config stockfish options to optimize performance:
        # print(self.engine.options["Hash"], self.engine.options["Threads"])
        # self.engine.configure({"Skill Level": 1,
        #                        "Threads": 8,
        #                        "Hash": 1024})

    def stop_engine(self):
        self.engine.quit()
        print("Stockfish stop \n")

    def move(self, observation, board):
        if board.turn is False:
            board = board.mirror()

        move = self.engine.engine.play(
            board=board,
            limit=chess.engine.Limit(time=None, depth=None),
        )  # TODO Test Different Settings & Depths

        return move_to_act(move.move)


class Random(Agent):
    """
    Always returns a random move from the action_mask.
    """

    def __init__(self):
        super().__init__()

    def move(self, observation, board):
        return random.choice(np.flatnonzero(observation["action_mask"]))


class A2C(Agent):
    """
    Main agent.
    Take observations from a tuple.
    Feeds it to a A2C network.
    """

    def __init__(self):
        super().__init__()

        self.idx = 0
        self.net = A2CNet()
        self.obs = collections.deque(maxlen=CFG.buffer_size)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=CFG.learning_rate)

    def step(self):
        """
        Experience replay before feeding our data to the model.
        """
        # TODO Needs doing
        pass

    def move(self, obs, _):
        """
        Next action selection.
        """
        # TODO Remove when we get real masks
        mask = np.array([random.randint(0, 1) for _ in range(4672)])
        obs = torch.tensor(obs).float().unsqueeze(0)
        _, pol = self.net(obs)
        pol = pol.squeeze(0).detach().numpy() * mask
        pol = pol / sum(pol)
        return np.random.choice(range(len(pol)), p=pol)


    def learn(self):
        """
        Trains the model.
        """
        old, act, rwd, new = BUF.get()
        val, pol = self.net(old)

        entropy = (pol.detach() * torch.log(pol.detach())).sum(axis=1)

        y_pred_pol = torch.log(torch.gather(pol, 1, act).squeeze(1) + 1e-6)
        y_pred_val = val.squeeze(1)
        y_true_val = rwd + CFG.gamma * self.net(new)[0].squeeze(1).detach()
        adv = y_true_val - y_pred_val

        val_loss = 0.5 * torch.square(adv)
        pol_loss = -(adv * y_pred_pol)
        loss = (pol_loss + val_loss).mean() # + 1e-6 * entropy

        self.idx += 1

        #print(y_pred_pol)
        tp = pol[0].detach()
        tps, _ = torch.sort(tp, descending=True)
        print(tp.max(), tp.mean(), tp.min())
        print(tps.numpy()[:5])
        #print(self.idx, pol_loss, loss)

        self.opt.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.net.pol.parameters(), 0.001)
        #torch.nn.utils.clip_grad_norm_(self.net.val.parameters(), 0.001)
        self.opt.step()

    def save(self, path: str):
        """
        Save the agent's model to disk.
        """
        torch.save(self.net.state_dict(), path)

    def load(self, path: str):
        """
        Load the agent's weights from disk.
        """
        dat = torch.load(path, map_location=torch.device("cpu"))
        self.net.load_state_dict(dat)