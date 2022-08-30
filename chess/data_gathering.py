import os
import random
from time import sleep

import chess.pgn
from chess import Move
import numpy as np
from pettingzoo.classic import chess_v5
from pettingzoo.classic.chess import chess_utils

import bz2


env = chess_v5.env()
chess_env = env.env.env.env.env
env.reset()

def moves_gathering(batch:int):

    data = bz2.open('../raw_data/lichess_db_standard_rated_2022-07.pgn.bz2', mode='rt')

    move_list = []

    for _ in range(batch):
        game = chess.pgn.read_game(data)
        for move in game.mainline_move_list():
            move_list.append(move)

    return move_list



def move_to_act(move):
    x, y = chess_utils.square_to_coord(move.from_square)
    panel = chess_utils.get_move_plane(move)
    return (x * 8 + y) * 73 + panel

def play_pgn(move_list):
    state_t = []
    rewards = []
    actions = []

    for agent, move in zip(env.agent_iter(), move_list):

        observation, reward, done, info = env.last()
        state_t.append(observation['observation'])
        rewards.append(reward)

        if done:
            print("Game done", reward, "\n\n")
            break

        mirr_move = chess_utils.mirror_move(move) if agent == "player_1" else move

        action = move_to_act(mirr_move)
        env.step(action)
        actions.append(action)

        print(f"{agent}  plays  {move}  (action {action}/4672):")
        env.render()
        print("\n")
        #sleep(1)

    env.reset()
    print(len(state_t), len(rewards), len(actions))

    return state_t, rewards, actions

def pgn_to_tuple(states, rewards, actions):
    number_of_moves = len(states)
    list_tuples = []
    for i in range(number_of_moves - 2):
        tuple = (states[i], actions[i], rewards[i], states[i+1])
        list_tuples.append(tuple)
    list_tuples.append((states[number_of_moves - 1], actions[number_of_moves - 1], rewards[number_of_moves - 1], None))
    return list_tuples

#print(pgn_to_tuple(play_pgn(game_test)[0], play_pgn(game_test)[1],play_pgn(game_test)[2]))



def load_pgn():
    dir = os.path.join(os.path.dirname(__file__), "../data")
    pgn = open(f"{dir}/2022-01.bare.[19999] copy.pgn")

    move_list = []
    for _ in range(1):
        game = chess.pgn.read_game(pgn)
        for move in game.mainline_moves():
            move_list.append(move)  # .uci())
            # print(game.end())

    print(move_list, "\n")
    return move_list
