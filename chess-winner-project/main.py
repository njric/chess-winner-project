import argparse
import math
import os
from tkinter import E

from agent import DQNAgent, Random, StockFish, A2C
from environnement import Environment, load_pgn
from utils import *
from buffer import BUF
from config import CFG

from typing import List


# TODO:
# - Implement train from pkl
# - Implement stats
# - Implement eval
# - Implement CFG

def feed(path: str):
    """
    Train a single agent using pickles game files
    """
    path = os.path.join(os.path.dirname(__file__), f"../data/")
    # game_files = [x for x in ]
    agent = A2C()

    for game_file in utils.list_pickles(path):
        for idx, obs in enumerate(utils.from_disk(game_file)):
            BUF.set(obs)
            if idx % CFG.batch_size == 0 and BUF.len() >= CFG.batch_size:
                agent.learn()

    # agent.save(path)


def play(agent, agent2, evaluate=False):
    """
    Play chess using two agents.
    """
    # raise NotImplementedError
    env = Environment((agent, agent2))

    agent.model = weight_loader(agent.model, 'model_#')

    for n in range(10000):
        epsilon = math.exp(epsilon_decay * n)  # -> implement in agent move
        env.play(render=True)


        if evaluate:
            eval(agent, agent2)

    weight_saver(agent.model, f"model_{eval_idx}")


def eval(agent, agent2, n_eval=5, eval_idx=0, render=False):
    # raise NotImplementedError
    tot_win, tot_draws = 0, 0
    env = Environment((agent, agent2))
    eval_idx += 1
    # agent.eval()   #Set NN model to evaluation mode.
    results = {0: [], 1: []}

    for _ in range(n_eval):
        env.play(render)
        results[0].append(env.results)
        results[1].append(- env.results)

    wins = results[0].count(1)
    draws = results[0].count(0)
    losses = results[0].count(-1)
    outcome = (wins, draws, losses)
    tot_win =  tot_win + wins
    tot_draws =  tot_draws + draws

    print(f"{eval_idx}: White: Wins: {results[0].count(1)}, Draws {results[0].count(0)}, Losses {results[0].count(-1)} \n")
    print(f"{eval_idx}: Black: Wins: {results[1].count(1)}, Draws {results[1].count(0)}, Losses {results[1].count(-1)} \n")
    print(f"Since init: total wins {tot_win} & total draws {tot_draws}")

    #agent.model.train()  # Set NN model back to training mode
    return outcome


def baseline():

    list_pickles = list_pickles()
    for pickle in list_pickles:
        data = from_disk(pickle)
        print(data)
        exit()

    pass

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", action="store",
                        help="Set the preprocessing to val")  # création d'un argument
    return parser.parse_args()  # lancement de argparse


if __name__ == "__main__":

    path = os.path.join(os.path.dirname(__file__), f"../weights")

    agent = DQNAgent()
    agent.load(f'{path}/saved_model.pt')

    agent2 = StockFish(epsilon_greed=0.8)

    eval(agent, agent2, n_eval=10, render=False)
