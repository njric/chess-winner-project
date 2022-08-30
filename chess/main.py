import os
import random
from time import sleep

import chess.pgn
import numpy as np
from pettingzoo.classic import chess_v5
from pettingzoo.classic.chess import chess_utils


def move_to_act(move):
    x, y = chess_utils.square_to_coord(move.from_square)
    panel = chess_utils.get_move_plane(move)
    return (x * 8 + y) * 73 + panel

def score(move: str): # -> score
    if move == "1-0":
        return 1
    elif move == "0-1":
        return -1
    else:
        return 0

DB = []

def load_pgn():

    env = chess_v5.env()
    pgn = open(os.path.join(os.path.dirname(__file__), "2022-01.bare.[19999].pgn"))
    dat = ({'obs': None, 'act': None},
           {'obs': None, 'act': None})

    while (game := chess.pgn.read_game(pgn)) is not None:

        env.reset()

        for agent, move in zip(env.agent_iter(), game.mainline_moves()):

            agent = int(agent == "player_1")
            new, rwd, done, info = env.last()

            if type(move) is str:
                reward = score(move)
                done = True
            if done:
                print("Game done", reward, "\n\n")
                env.reset()
                continue

            if dat[agent]['obs'] is not None:

                old = dat[agent]['obs']
                act = dat[agent]['act']

                DB.append((old, act, rwd, new))
                print(old.keys(), act, rwd, new.keys())

            if agent == 1:
                move = chess_utils.mirror_move(move)
            action = move_to_act(move)
            env.step(action)


            dat[agent]['obs'] = new
            dat[agent]['act'] = action

            if done:
                break

        print("Done")

load_pgn()
