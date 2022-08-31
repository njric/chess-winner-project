import bz2
import os

from pettingzoo.classic import chess_v5
from pettingzoo.classic.chess import chess_utils

import chess.pgn
from utils import *

import argparse


def load_pgn():

    env = chess_v5.env()
    # make dir variable
    dir = os.path.join(os.path.dirname(__file__))
    data_dir = os.path.join(os.path.dirname(dir), "raw_data")
    pgn = open(f"{data_dir}/2019-01.bare.[22119].pgn")
    # pgn = bz2.open(f"{data_dir}/lichess_db_standard_rated_2022-07.pgn.bz2", mode='rt')

    # Create a buffer to store board state / action by players
    dat = ({"obs": None, "act": None}, {"obs": None, "act": None})

    DB = []

    count = 0
    print("-"*10, "Starting", "-"*10)
    while (game := chess.pgn.read_game(pgn)) is not None: #while a game exist, set game variable

        #count games and save pickel at 500
        count += 1
        if count % 500 == 0:
            to_disk(DB)
            DB = []

        #reset env at start
        env.reset()

        result = game.headers["Result"]  #get result from game in case nor readable by pettingzoo

#       game_list format = [agent, move, result]
#       game_list ex = [0, Move.from_uci('g1f3'), None]
        game_list = [[int(i % 2 == 1), move, None] for i, move in enumerate(game.mainline_moves())]

        if len(game_list) == 0:
            continue

        game_list[-1][-1] = result #set last None of a gamelist as result

        for (agent, move, result) in game_list:

            new, rwd, done, info = env.last()

            # check if result form gamelist exist = a player quit
            if type(result) is str:
                rwd = score(result, agent)
                done = True

            if dat[agent]["obs"] is not None:

                old = dat[agent]["obs"]
                act = dat[agent]["act"]

                if agent == 1:
                    rwd = -1*rwd
                DB.append((old, act, rwd, new))

            if done:  # or ("legal_moves" in info):  # TODO -> check legal_moves necessity
                # Set last move for losing agent
                agent = (agent + 1 ) % 2

                old = dat[agent]["obs"]
                act = dat[agent]["act"]
                rwd = -1*rwd

                DB.append((old, act, rwd, None))

                dat = ({"obs": None, "act": None}, {"obs": None, "act": None})

                break

            move = chess_utils.mirror_move(move) if agent == 1 else move #mirror move for black
            action = move_to_act(move)
            env.step(action)

            dat[agent]["obs"] = new
            dat[agent]["act"] = action


load_pgn()