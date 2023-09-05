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
        out_sum = self.in_to_h1(input)
        # log softmax
        # why do we need to add dim=1
        output = F.log_softmax(out_sum, dim=1)
        return output


class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    #  one hidden layer, plus the output layer
    def __init__(self):
        super(NetFull, self).__init__()
        # L1(H * W * C, hidden_size)
        # L2(hidden_size, 10)   try from 10 to 100
        # 10 classes(Given)
        # independent parameters: (28 * 28 + 1) *120 + (120 + 1 ) * 10
        self.in_to_h1 = nn.Linear(28 * 28, 120)
        self.in_to_h2 = nn.Linear(120, 10)

    def forward(self, input_value):
        # flatten image, [10, H, W, C] => [B, H * W * C]
        input_value = input_value.view(input_value.size(0), -1)
        # linear function
        hid_sum = self.in_to_h1(input_value)
        hidden = torch.tanh(hid_sum)
        # log softmax
        # why do we need to add dim=1
        out_sum = self.in_to_h2(hidden)
        output = F.log_softmax(out_sum, dim=1)
        return output


class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        # in_channel in this case, it is 1 because it is a gray scale image
        # out_channels corresponding the number of filter, such as 8, 16, 32
        # kernel_size means: num * num kernel, such as 4, 5, 6
        # padding, arbitrary, such as 1, 2
        # stride, arbitrary, but usually 1
        # 28 * 28 gray scale image
        self.conv1 = nn.Conv2d(in_channels= 1, out_channels=8,  kernel_size= 5, padding= 2, stride= 1)
        self.conv2 = nn.Conv2d(in_channels= 8, out_channels=18, kernel_size= 5, padding= 2, stride= 1)
        # max pooling layer
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        # fully connected layer
        self.fc1 = nn.Linear(3528, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))

        out = self.MaxPool(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)
        return out
