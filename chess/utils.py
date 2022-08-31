from datetime import datetime
import os
import pickle
import glob

# Saving and loading batches of generated game data
def to_disk(obs):

    pdt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir = os.path.join(os.path.dirname(__file__), f"../raw_data/{pdt}_databatch.pkl")

    with open(dir, "wb") as file:
        pickle.dump(obs, file)

    print(f"Save to pickle @ {pdt}")


def list_pickle():
    dir = os.path.join(os.path.dirname(__file__), f"../raw_data")
    return glob.glob(dir + "/*.pkl")

def load_pickels(list_pickle):

    infile = open(list_pickle,'rb')
    data = pickle.load(infile)
    infile.close()

    return data

load_pickels()
