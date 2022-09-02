import argparse
import math
import os

from agent import Random, StockFish, A2C, DQNAgent
from environnement import Environment, load_pgn
import utils
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
    pkl_path = os.path.join(path, f"../pickle/")
    wgt_path = os.path.join(path, f"../weights/")

    # game_files = [x for x in ]
    agent = DQNAgent()

    tracker = 1
    for game_file in utils.list_pickles(pkl_path):
        print(f'Starting learning phase #{tracker}')
        for idx, obs in enumerate(utils.from_disk(game_file)):
            BUF.set(obs)
            if idx % CFG.batch_size == 0 and BUF.len() >= CFG.batch_size:
                agent.learn()

        agent.save(wgt_path)
        tracker += 1



def play():
    """
    Play chess using two agents.
    """
    raise NotImplementedError
    env = Environment(('agt0', 'agt1'))

    'agt0'.model = weight_loader('agt0'.model, 'model_#')

    for n in range(10000):
        epsilon = math.exp(epsilon_decay * n)  # -> implement in agent move
        env.play(render=True)

        if n % 500 == 0 and n != 0:
            eval('agt0')

    weight_saver('agt0.model', f"model_{eval_idx}")



def eval(agent, n_eval=5):
    # raise NotImplementedError
    env = Environment((agent, StockFish()))

    agent.model.eval()  # Set NN model to evaluation mode.

    for _ in range(n_eval):
        env.play()

    results = list('eval game outcomes') # -> white player reward
    print(f"{eval_idx}: Wins {results.count(1)}, Draws {results.count(0)}, Losses {results.count(-1)} \n")
    print(f"Since init: total wins {tot_win} & total draws {tot_draw}")

    agent.model.train()  # Set NN model back to training mode




def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", action="store", help="Set the preprocessing to val") #création d'un argument
    return parser.parse_args() #lancement de argparse

if __name__ == "__main__":
    #args = parse_arguments()
    #d = vars(args)['file'].split("/")[1]
    # load_pgn()
    path = os.path.dirname(__file__)
    feed(path)
