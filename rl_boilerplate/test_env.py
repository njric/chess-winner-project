import os
import random
from time import sleep

import chess.pgn
import numpy as np
from pettingzoo.classic import chess_v5
from pettingzoo.classic.chess import chess_utils


env = chess_v5.env()
chess_env = env.env.env.env.env
env.reset()


def play_random():
    rewards = []
    for agent in env.agent_iter():

        observation, reward, done, info = env.last()

        if done:
            print("Game done, reward:", reward, "\n\n")
            rewards.append(reward)
            print(rewards)
            env.reset()
            continue
            break

        # action = policy(observation, agent)  #  -> π Function
        action = random.choice(np.flatnonzero(observation["action_mask"]))

        env.step(action)

        print(f"{agent}  plays  random action {action}/4672):")
        env.render()
        print("\n")
        sleep(0.5)

    env.reset()


def play_pgn(move_list):

    for agent, move in zip(env.agent_iter(), move_list):

        observation, reward, done, info = env.last()

        if done:
            print("Game done", reward, "\n\n")
            break

        mirr_move = chess_utils.mirror_move(move) if agent == "player_1" else move

        action = move_to_act(mirr_move)
        env.step(action)

        print(f"{agent}  plays  {move}  (action {action}/4672):")
        env.render()
        print("\n")
        sleep(1)

    env.reset()


def move_to_act(move):
    x, y = chess_utils.square_to_coord(move.from_square)
    panel = chess_utils.get_move_plane(move)
    return (x * 8 + y) * 73 + panel


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


if __name__ == "__main__":
    # play_random()
    # quit()
    move_list = load_pgn()
    play_pgn(move_list)
