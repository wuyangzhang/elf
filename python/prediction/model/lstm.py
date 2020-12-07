import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, window, num_layers=4):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False).float()
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.hidden_size, window)

        self.sigmoid = torch.nn.Sigmoid()
        self.metrics = {}
        self.mse_loss = nn.MSELoss()

    def forward(self, x, target=None):
        """
        :param x: [batch, time step, features]
        :param target:
        :return: [batch, features]
        """
        x, h_state = self.rnn(x)

        x = x[:, -1, :]
        x = self.relu(x)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x
