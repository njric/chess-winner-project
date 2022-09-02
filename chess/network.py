"""
Neural Network
"""

import numpy as np
import torch
import torch.nn as nn

from config import CFG

class A2CNet(nn.Module):
    """
    A2C Neural Network
    """

    def __init__(self) -> None:
        super().__init__()

        net = [
            nn.Conv2d(111, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        ]

        for _ in range(CFG.convolution_layers - 3):
            net += [
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            ]

        net += [
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=16384, out_features=8192),
            nn.ReLU(inplace=True),
        ]

        self.net = nn.Sequential(*net)

        self.val = nn.Sequential(
            nn.Linear(in_features = 8192, out_features = 2048),
            nn.ReLU(inplace = True),
            nn.Linear(in_features = 2048, out_features = 256),
            nn.ReLU(inplace = True),
            nn.Linear(in_features = 256, out_features = 1),
            nn.Tanh()
        )

        self.pol = nn.Sequential(
            nn.Linear(8192, 8192), nn.ReLU(inplace = True),
            nn.Linear(8192, 4096), nn.ReLU(inplace = True),
            nn.Linear(4096, 4672),
            nn.LogSoftmax(dim=-1),
        )


    def forward(self, X):
        """
        Equivalent of a .fit for a neural network in pytorch
        """
        y = self.net(X)
        y_val = self.val(y)
        y_pol = self.pol(y).double() # We need double precision for small probabilities.
        # torch.clamp(y_pol, min=10e-39, max=1)
        return y_val, y_pol.exp()


class DQN(nn.Module):
    """
    DQN Model
    """

    def __init__(self) -> None:
        super().__init__()

        net = [
            nn.Conv2d(111, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        ]

        for _ in range(CFG.convolution_layers - 3):
            net += [
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            ]

        net += [
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=16384, out_features=8192),
            nn.ReLU(inplace=True),
        ]

        self.net = nn.Sequential(*net)


        self.linear = nn.Sequential(
            nn.Linear(8192, 8192), nn.ReLU(inplace = True),
            nn.Linear(8192, 4096), nn.ReLU(inplace = True),
            nn.Linear(4096, 4672),
            #nn.LogSoftmax(dim=-1),
        )

    def forward(self, X):
        y = self.net(X)
        y = self.linear(y)
        return y
