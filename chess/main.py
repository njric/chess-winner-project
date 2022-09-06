import argparse
import math
import os

from agent import Random, StockFish, A2C
from agent import Agent
from environnement import Environment, load_pgn
from utils import *
from buffer import BUF
from config import CFG

from typing import List
from config import CFG
epsilon_decay = CFG.epsilon_decay
# TODO:
# - Implement train from pkl
# - Implement stats
# - Implement eval
# - Implement CFG
#eval_idx = 0
tot_win, tot_draw, tot_loss = 0, 0, 0

def feed(path: str):
    """
    Train a single agent using pickles game files
    """
    path = os.path.join(os.path.dirname(__file__), f"../data/")
    # game_files = [x for x in ]
    agent = A2C()

    for game_file in list_pickles(path):
        for idx, obs in enumerate(from_disk(game_file)):
            BUF.set(obs)
            if idx % CFG.batch_size == 0 and BUF.len() >= CFG.batch_size:
                agent.learn()

    # agent.save(path)


def play():
    """
    Play chess using two agents.
    """
    # raise NotImplementedError
    env = Environment(('agt0', 'agt1'))

    'agt0'.model = weight_loader('agt0'.model, 'model_#')

    for n in range(10000):
        CFG.epsilon = math.exp(epsilon_decay * n)  # -> implement in agent move
        env.play(render=True)

        if n % 500 == 0 and n != 0:
            eval('agt0')

    weight_saver('agt0.model', f"model_{eval_idx}")


def eval(agent, agent2, n_eval=5, eval_idx=0):
    # raise NotImplementedError
    tot_win = 0
    tot_draws = 0
    env = Environment((agent, agent2))
    eval_idx += 1
    #agent.model.eval()   Set NN model to evaluation mode.
    results = {0: [], 1: []}

    for _ in range(n_eval):
        env.play(render=True)
        results[env.idx].append(env.results)
        results[1 - env.idx].append(- env.results)

    wins = results[0].count(1)
    draws = results[0].count(0)
    losses = results[0].count(-1)
    outcome = (wins, draws, losses)
    tot_win =  tot_win + wins
    tot_draws =  tot_draws + draws

    print(f"{eval_idx}: White wins{results[0].count(1)}, Draws {results[0].count(0)}, Losses {results[0].count(-1)} \n")
    print(f"{eval_idx}: Black wins{results[1].count(1)}, Draws {results[1].count(0)}, Losses {results[1].count(-1)} \n")
    print(f"Since init: total wins {tot_win} & total draws {tot_draw}")

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
    parser.add_argument("-f", "--file", action="store", help="Set the preprocessing to val") #création d'un argument
    return parser.parse_args() #lancement de argparse

if __name__ == "__main__":
    #args = parse_arguments()
    #d = vars(args)['file'].split("/")[1]
    # load_pgn()
    tot_win, tot_draw, tot_loss = 0, 0, 0
    agent = Random()
    agent2 = Random()
    eval(agent, agent2, 100)
