from datetime import datetime
import os
import pickle
import glob
from pettingzoo.classic.chess import chess_utils

# Saving and loading batches of generated game data
def to_disk(obs):

    pdt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir = os.path.join(os.path.dirname(__file__), f"../raw_data/{pdt}_databatch.pkl")

    with open(dir, "wb") as file:
        pickle.dump(obs, file)

    print(f"Save to pickle @ {pdt}")


def list_pickle():
    dir = os.path.join(os.path.dirname(__file__), f"../raw_data")
    return glob.glob(dir + "/*.pkl")

def load_pickels(list_pickle):

    infile = open(list_pickle,'rb')
    data = pickle.load(infile)
    infile.close()

    return data

def move_to_act(move):
    x, y = chess_utils.square_to_coord(move.from_square)
    panel = chess_utils.get_move_plane(move)
    return (x * 8 + y) * 73 + panel


def score(move: str, agent: int):  # -> score
    if move == "1-0":
        return 1
    elif move == "0-1":
        return -1
    else:
        return 0
