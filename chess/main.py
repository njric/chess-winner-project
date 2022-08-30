import os
import random
from time import sleep

import chess.pgn
import numpy as np
from pettingzoo.classic import chess_v5
from pettingzoo.classic.chess import chess_utils

import bz2

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

def move_list():
    dir = os.path.join(os.path.dirname(__file__), "../raw_data")
    pgn = bz2.open(f"{dir}/lichess_db_standard_rated_2022-07.pgn.bz2", mode='rt')

    move_list = []
    for _ in range(10):
        game = chess.pgn.read_game(pgn)
        for move in game.mainline_moves():
            move_list.append(move)  # .uci())
            # print(game.mainline())
        move_list.append(game.headers["Result"])

    print(move_list, "\n")
    return move_list

def load_pgn(move_list):

    env = chess_v5.env()
    dat = ({'obs': None, 'act': None},
           {'obs': None, 'act': None})

    env.reset()

    for agent, move in zip(env.agent_iter(), move_list):

        agent = int(agent == "player_1")
        new, reward, done, info = env.last()

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

            DB.append((old, act, reward, new))
            # print(old.keys(), act, reward, new.keys())

        if agent == 1:
            move = chess_utils.mirror_move(move)
        action = move_to_act(move)
        env.step(action)


        dat[agent]['obs'] = new
        dat[agent]['act'] = action

    print("Done")



moves = move_list()
load_pgn(moves)
print(len(DB))
