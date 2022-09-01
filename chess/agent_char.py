"""
Agent
"""

import collections
import random
import subprocess

import chess.engine
import numpy as np
import torch

from network import Network
from parameters import params
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
        if board.turn == False:
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


class ChessAgent(Agent):
    """
    Main agent.
    Take observations from a tuple.
    Feeds it to a A2C network.
    """

    def __init__(self):
        super().__init__()

        #TODO Needs finishing

        self.net = Network()

        #TODO ?? not sure why we're using a deque instead of a list?...
        self.obs = collections.deque(maxlen = params.buffer_size)

        self.opt = torch.optim.Adam(self.net.parameters(), lr = params.learning_rate)



    def feed_er():
        """
        Experience replay before feeding our data to the model.
        """
        #TODO Needs doing
        pass


    def move():
        """
        Next action selection.
        """
        #TODO Needs doing
        pass


    def learn(self):
        """
        Trains the model.
        """

        #Gets a training batch from the feed function
        training_batch = random.sample(self.obs, params.batch_size)

        #Init. the log_probs, values & rewards of the batch to compute the loss
        log_probs = []
        values = []
        rewards = []
        total_entropy = 0

        #Gets state_t, action_t, reward_t+1, state_t+1
        for i, (old, act, rwd, new) in enumerate(zip(*training_batch)):
            with torch.no_grad():
                #Gets the output from the network: one value and a policy list
                value, policy_list = self.net(old)
                values.append(value)

                #TODO Double check if the first action is 0 or 1 (right index?)
                log_prob = torch.log(policy_list[act - 1])
                log_probs.append(log_prob)

                rewards.append(rwd)

                #Computes the entropy
                entropy = -np.sum(act * np.log(act))
                total_entropy += entropy

        #Computes the Qs
        Qs = np.zeros_like(values)
        for t in reversed(range(len(rewards))):
            Q = rewards[t] - params.gamma * Q
            Qs[t] = Q

        #Computes the advantage
        advantages = Qs - values

        #Value loss
        value_loss = 0.5 * (advantages ** 2).mean()

        #Policy loss
        policy_loss = -(advantages * log_probs).mean()

        #Total loss
        total_loss = policy_loss + value_loss + params.entropy_weight * entropy


        #Backward propagation to update the weights
        #TODO must have an issue with the type of total loss...
        self.opt.zero_grad()
        total_loss.backward()
        self.opt.step()
