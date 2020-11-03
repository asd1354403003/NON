import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as F


class DNN(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, real):
        super(DNN, self).__init__()

        self.loss = 0
        self.hidden1 = nn.Linear(n_input, n_hidden, True)

        self.hidden2 = nn.Linear(n_hidden, n_output, True)

        self.o = nn.Linear(n_output, real, True)

        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.o(x)
        x = self.sig(x)
        return x.squeeze(1)


