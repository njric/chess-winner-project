# from ast import main
import chess
import chess.pgn
import pettingzoo.classic.chess.chess_utils as chess_utils
from main import parse_arguments

import utils
import os


def load_baseline(datafile=None):

    pgn = open(os.path.join(os.path.dirname(__file__), f"../raw_data/{datafile}"))

    DB = {}

    count = 0
    print("-"*10, "Starting", "-"*10)
    while (game := chess.pgn.read_game(pgn)) is not None:  # while a game exist, set game variable

        count += 1
        if (count := count + 1) % 100 == 0:
            print(count)

        board = chess.Board()

        for move in game.mainline_moves():

            if (env := " ".join(board.fen().split(" ")[:4])) not in DB:
                DB[env] = {act: 0 for act in chess_utils.legal_moves(board)}

            act = utils.move_to_act(move, mirror=not board.turn)

            DB[env][act] += 1

            board.push(move)

    utils.to_disk(DB)


if __name__ == "__main__":
    args = parse_arguments()
    d = vars(args)['file'].split("/")[1]
    load_baseline(d)
