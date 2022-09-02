import argparse
import math
import os
from typing import List

from google.cloud import storage

import utils
from agent import A2C, DQNAgent, Random, StockFish
from buffer import BUF
from config import CFG
from environnement import Environment, load_pgn
from utils import from_disk

# TODO:
# - Implement train from pkl
# - Implement stats
# - Implement eval
# - Implement CFG

def feed(path: str):
    """
    Train a single agent using pickles game files
    """
    pkl_path = os.path.join(path, f"../pickle/")
    wgt_path = os.path.join(path, f"../weights/")

    # game_files = [x for x in ]
    agent = DQNAgent()

    for game_file in utils.list_pickles(pkl_path):
        print(f'Starting learning phase #{tracker}')
        for idx, obs in enumerate(utils.from_disk(game_file)):
            BUF.set(obs)
            if idx % CFG.batch_size == 0 and BUF.len() >= CFG.batch_size:
                agent.learn()

        agent.save(wgt_path)

def feed_vm():
    """
    Train a single agent using pickles game files
    """

    path = os.path.dirname(__file__)
    pkl_path = os.path.join(path, f"../pickle/")
    wgt_path = os.path.join(path, f"../weights/")

    # game_files = [x for x in ]
    agent = DQNAgent()

    for game_file in utils.list_pickles(pkl_path):
        for idx, obs in enumerate(utils.from_disk(game_file)):
            BUF.set(obs)
            if idx % CFG.batch_size == 0 and BUF.len() >= CFG.batch_size:
                agent.learn()

        agent.save(wgt_path)

def play():
    """
    Play chess using two agents.
    """
    raise NotImplementedError
    env = Environment(('agt0', 'agt1'))

    'agt0'.model = weight_loader('agt0'.model, 'model_#')

    for n in range(10000):
        epsilon = math.exp(epsilon_decay * n)  # -> implement in agent move
        env.play(render=True)

        if n % 500 == 0 and n != 0:
            eval('agt0')

    weight_saver('agt0.model', f"model_{eval_idx}")



def eval(agent, n_eval=5):
    raise NotImplementedError
    env = Environment((agent, StockFish()))

    agent.model.eval()  # Set NN model to evaluation mode.

    for _ in range(n_eval):
        env.play()

    results = list('eval game outcomes') # -> white player reward
    print(f"{eval_idx}: Wins {results.count(1)}, Draws {results.count(0)}, Losses {results.count(-1)} \n")
    print(f"Since init: total wins {tot_win} & total draws {tot_draw}")

    agent.model.train()  # Set NN model back to training mode


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The ID of your GCS object
    # source_blob_name = "storage-object-name"

    # The path to which the file should be downloaded
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    # print(
    #     "Downloaded storage object {} from bucket {} to local file {}.".format(
    #         source_blob_name, bucket_name, destination_file_name
    #     )


def get_pickle_name():

    with open("bucket_pickle_list.txt", mode= "r") as f:
        d_list = f.readlines()

    pickle_names = []
    for pickle_name in d_list:
        pickle_name = pickle_name.strip('\n')
        pickle_name = pickle_name.split('/')[-1]
        pickle_names.append(pickle_name)

    return pickle_names


if __name__ == "__main__":

    PROJECT = os.environ.get("PROJECT")
    BUCKET = os.environ.get("BUCKET")
    pickle_names = get_pickle_name()[1:]
    pkl_done = []
    tracker = 0

    for p in pickle_names:
        print(f'Starting download {tracker}')
        download_blob(BUCKET, p, f'./pickle/my_unique_pickle.pkl')
        pkl_done.append(p)
        print(f'Starting learning phase #{tracker}')
        feed_vm()
        tracker += 1
