import bz2
import os

from pettingzoo.classic import chess_v5
from pettingzoo.classic.chess import chess_utils

import chess.pgn
from utils import to_disk


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


def load_pgn():

    env = chess_v5.env()
    pgn = open(os.path.join(os.path.dirname(__file__), "../raw_data/2022-01.bare.[19999].pgn"))
    # pgn = bz2.open(os.path.join(os.path.dirname(__file__), "../raw_data/lichess_db_standard_rated_2022-07.pgn.bz2", mode='rt')
    dat = ({"obs": None, "act": None}, {"obs": None, "act": None})

    DB = []

    count = 0
    while (game := chess.pgn.read_game(pgn)) is not None:

        count += 1
        print(count, len(DB))
        if count % 500 == 0:
            # to_disk(DB)
            DB = []

        env.reset()

        result = game.headers["Result"]
        game_list = [[int(i % 2 == 1), move, None] for i, move in enumerate(game.mainline_moves())]
        game_list[-1][-1] = result

        for (agent, move, result) in game_list:

            new, rwd, done, info = env.last()

            if dat[agent]["obs"] is not None:

                old = dat[agent]["obs"]
                act = dat[agent]["act"]

                DB.append((old, act, rwd, new))

            if type(result) is str:
                rwd = score(result, agent)
                done = True

            if done:  # or ("legal_moves" in info):  # TODO -> check legal_moves necessity
                dat = ({"obs": None, "act": None}, {"obs": None, "act": None})
                break

            move = chess_utils.mirror_move(move) if agent == 1 else move
            action = move_to_act(move)
            env.step(action)

            dat[agent]["obs"] = new
            dat[agent]["act"] = action


load_pgn()
