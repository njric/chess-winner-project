import glob
import os
import pickle
from datetime import datetime

import torch
from pettingzoo.classic.chess import chess_utils


def to_disk(obs, suffix = None):
    # Save a batch of generated game data to a file

    if suffix is None:
        suffix = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


    dir = os.path.join(os.path.dirname(__file__),
                    f"../data/{suffix}_databatch.pkl")

    with open(dir, "wb") as file:
        pickle.dump(obs, file)
        print(f"Save to pickle @ {suffix}")


def list_pickles(dir: str):
    """
    List all pickle files in a given directory.
    """
    return glob.glob(os.path.join(dir) + "/*.pkl")


def from_disk(file):
    # Load a pickle from a file
    with open(file, "rb") as f:
        return pickle.load(f)


def move_to_act(move, mirror=False):
    # Return a pettingzoo-style action from a chess move
    move = chess_utils.mirror_move(move) if mirror else move
    x, y = chess_utils.square_to_coord(move.from_square)
    panel = chess_utils.get_move_plane(move)
    return (x * 8 + y) * 73 + panel
