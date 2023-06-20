"""
   kuzu.py
   COMP9444, CSE, UNSW
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        # 28 * 28 gray scale image
        # up to 10 classes
        self.in_to_h1 = nn.Linear(28 * 28, 10)

    def forward(self, input):
        # flatten image, [10, H, W, C] => [B, H * W * C]
        input = input.view(input.size(0), -1)
        # linear function
        hidden = self.in_to_h1(input)
        # log softmax
        # why do we need to add dim=1
        output = F.log_softmax(hidden, dim=1)
        return output


class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()

    def forward(self, input):
        pass


class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        # INSERT CODE HERE

    def forward(self, x):
        return 0  # CHANGE CODE HERE
