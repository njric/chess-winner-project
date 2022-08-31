import glob
import os
import pickle
from datetime import datetime

from pettingzoo.classic.chess import chess_utils


# Saving and loading batches of generated game data
def to_disk(obs):

    pdt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir = os.path.join(os.path.dirname(__file__), f"../pickle/{pdt}_databatch.pkl")

    with open(dir, "wb") as file:
        pickle.dump(obs, file)

    print(f"Save to pickle @ {pdt}")


def list_pickle():
    dir = os.path.join(os.path.dirname(__file__), f"../pickle")
    return glob.glob(dir + "/*.pkl")


def from_disk(file):
    with open(file, "rb") as f:
        return pickle.load(f)


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
