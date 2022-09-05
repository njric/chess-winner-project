import argparse
import math
import os

from agent import Random, StockFish, A2C
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


def play(learning = True):
    """
    Play chess using two agents.
    """
    # raise NotImplementedError
    env = Environment(('agt0', 'agt1'))

    'agt0'.model = weight_loader('agt0'.model, 'model_#')

    for n in range(10000):
        epsilon = math.exp(epsilon_decay * n)  # -> implement in agent move
        env.play(render=True)

        if learning == False:
            if n % 500 == 0 and n != 0:
                eval('agt0')

    weight_saver('agt0.model', f"model_{eval_idx}")


def eval(agent_1, agent_2, n_eval=5):
    # raise NotImplementedError
    env = Environment((agent_1, agent_2))
    eval_idx += 1
    agent_1.model.eval()  # Set NN model to evaluation mode.
    results = {}

    for _ in range(n_eval):
        env.play(render=True)
        results[env.idx].append(env.results)
        results[1 - env.idx].append(- env.results)


    wins = results.count(1)
    draws = results.count(0)
    losses = results.count(-1)
    outcome = (wins, draws, losses)
    tot_win += wins
    tot_draws += draws

    print(f"{eval_idx}: White {agent_1}: Wins {results[0].count(1)}, Draws {results[0].count(0)}, Losses {results[0].count(-1)} \n")
    print(f"{eval_idx}: Black {agent_2}: Wins {results[1].count(1)}, Draws {results[1].count(0)}, Losses {results[1].count(-1)} \n")

    print(f"Since init: White: total wins {tot_win} & total draws {tot_draw}")

    agent_1.model.train()  # Set NN model back to training mode for agent 1
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
    # feed("")
    agent_1 = Random()
    agent_2 = Random()
    eval(agent_1,agent_2)
