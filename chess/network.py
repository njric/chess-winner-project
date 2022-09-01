"""
Neural Network
"""

import numpy as np
import torch
import torch.nn as nn

class Network(nn.Module):
    """
    A2C Neural Network
    """

    def __init__(self) -> None:
        super().__init__()

        self.input_shape = 111

        self.net = nn.Sequential(
            nn.Conv2d(self.input_shape, 512, kernel_size = 2, stride=1, padding=1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2),

            nn.Conv2d(512, 256, kernel_size = 2, stride=1, padding=1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size = 2, stride=1, padding=1),
            nn.ReLU(inplace = True),

            nn.Flatten(start_dim=0),

            nn.Linear(in_features = 4608, out_features = 1024),
            nn.ReLU(inplace = True),

        )

        #Returns our first output, the value of the state.
        self.val = nn.Sequential(
            nn.Linear(in_features = 1024, out_features = 512),
            nn.ReLU(inplace = True),

            nn.Linear(in_features = 512, out_features = 256),
            nn.ReLU(inplace = True),

            nn.Linear(in_features = 256, out_features = 1)

        )

        #Returns our second output, the policy for that state.
        self.pol = nn.Sequential(
            nn.Linear(in_features = 1024, out_features = 2048),
            nn.ReLU(inplace = True),

            nn.Linear(in_features = 2048, out_features = 4096),
            nn.ReLU(inplace = True),

            nn.Linear(in_features = 4096, out_features = 4672)
        )


    def forward(self, X):
        """
        Equivalent of a .fit for a neural network in pytorch
        """
        y = self.net(X)
        y_val = self.val(y)
        y_pol = self.pol(y)
        return y_val, y_pol



#Running a test on an empty tensor to see if the shapes in the model are right.

shape = (111, 8, 8)
test_zeros = np.zeros(shape)
tensor_zeros = torch.tensor(test_zeros, dtype=torch.float)
model = Network()
y = model(tensor_zeros)
print(y)


#TODO: Might need to add a loss and a way of evaluating our model?
