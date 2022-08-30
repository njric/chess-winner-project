import os
from time import sleep

import chess.pgn
import numpy as np
from pettingzoo.classic import chess_v5
from pettingzoo.classic.chess import chess_utils


env = chess_v5.env()
chess_env = env.env.env.env.env
env.reset()
def score(move: str): # -> score
    if move == "1-0":
        return 1
    elif move == "0-1":
        return -1
    else:
        return 0

def play_pgn(move_list):

    for agent, move in zip(env.agent_iter(), move_list):

        observation, reward, done, info = env.last()

        if type(move) is str:
            print('END')
            reward = score(move)
            done = True

        if done:
            print("Game done", reward, "\n\n")
            env.reset()
            continue

        mirr_move = chess_utils.mirror_move(move) if agent == "player_1" else move

        action = move_to_act(mirr_move)
        env.step(action)

        print(f"{agent}  plays  {move}  (action {action}/4672):")
        env.render()
        print("\n")
        sleep(0.1)

    env.reset()


def move_to_act(move):
    x, y = chess_utils.square_to_coord(move.from_square)
    panel = chess_utils.get_move_plane(move)
    return (x * 8 + y) * 73 + panel


def load_pgn():
    dir = os.path.join(os.path.dirname(__file__), "../raw_data")
    pgn = open(f"{dir}/2022-01.bare.[19999] copy.pgn")

    move_list = []
    for _ in range(1):
        game = chess.pgn.read_game(pgn)
        for move in game.mainline_moves():
            move_list.append(move)  # .uci())
            # print(game.mainline())
        move_list.append(game.headers["Result"])

    print(move_list, "\n")
    return move_list


if __name__ == "__main__":
    # play_random()
    # quit()
    move_list = load_pgn()
    # play_pgn(move_list)
