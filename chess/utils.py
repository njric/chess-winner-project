from datetime import datetime
import os
import pickle


# Saving and loading batches of generated game data
def to_disk(obs):

    pdt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir = os.path.join(os.path.dirname(__file__), f"../raw_data/{pdt}_databatch.pkl")

    with open(dir, "wb") as file:
        pickle.dump(obs, file)

    print(f"Save to pickle @ {pdt}")
