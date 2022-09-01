import argparse
import math

from agent import Random, StockFish
from environnement import Environment, load_pgn
from utils import *


# TODO:
# - Implement train from pkl
# - Implement stats
# - Implement eval
# - Implement CFG

eval_idx = 0

def offline():
    # raise NotImplementedError
    agent = 'Agent()'

    for idx, file in enumerate(list_pickles()):
        print(file)

        for obs in from_disk(file):
            agent.train(obs)

        if idx % 10 == 0 and idx != 0:
            eval(agent)

    weight_saver(agent.model, f"model_{eval_idx}")



def play():
    # raise NotImplementedError
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
    results = []
    agent.model.eval()  # Set NN model to evaluation mode.
    eval_idx += 1

    for _ in range(n_eval):
        env.play(eval_state=True)
        results.append(env.play_outcome)

    print(f"Results are:")
    # results = list('eval game outcomes') # -> white player reward
    # print(f"{eval_idx}: Wins {results.count(1)}, Draws {results.count(0)}, Losses {results.count(-1)} \n")
    # print(f"Since init: total wins {tot_win} & total draws {tot_draw}")

    agent.model.train()  # Set NN model back to training mode


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", action="store", help="Set the preprocessing to val") #création d'un argument
    return parser.parse_args() #lancement de argparse

if __name__ == "__main__":
    args = parse_arguments()
    d = vars(args)['file'].split("/")[1]
    load_pgn(d)
