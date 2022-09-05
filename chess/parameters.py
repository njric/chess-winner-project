'''All of the parameters we need
and might want to change'''

from time to time.
import os

PROJECT = os.environ.get("PROJECT")
BUCKET = os.environ.get("BUCKET")


class Parameters:

    def __init__(self):

        self.batch_size = 1024
        self.buffer_size = 1000
        self.entropy_weight = 0.0001
        self.gamma = 0.98
        self.learning_rate = 0.001

params = Parameters()
