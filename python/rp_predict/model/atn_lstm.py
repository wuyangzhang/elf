import torch
import torch.nn.functional as tf
from torch import nn

from config import Config


def init_hidden(config: Config, x, hidden_size: int):
    """
    Train the initial value of the hidden state:
    https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
    """
    if config.use_gpu:
        return torch.cuda.FloatTensor(1, x.size(0), hidden_size).fill_(0)
    return torch.zeros(1, x.size(0), hidden_size)


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(Encoder, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        return self.rnn(x)


class Decoder(nn.Module):
    def __init__(self, config: Config, encoder_hidden_size: int, decoder_hidden_size: int, window: int, out_feats=4):
        super(Decoder, self).__init__()

        self.config = Config()
        self.window = window
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.attn_layer = nn.Sequential(nn.Linear(2 * decoder_hidden_size + encoder_hidden_size,
                                                  encoder_hidden_size),
                                        nn.Tanh(),
                                        nn.Linear(encoder_hidden_size, 1))
        self.lstm_layer = nn.LSTM(input_size=out_feats, hidden_size=decoder_hidden_size)
        self.fc = nn.Linear(encoder_hidden_size + out_feats, out_feats)
        self.fc_final = nn.Linear(decoder_hidden_size + encoder_hidden_size, out_feats)
        self.fc.weight.data.normal_()

        self.hidden = torch.zeros(1, 512, self.decoder_hidden_size)
        self.cell = torch.zeros(1, 512, self.decoder_hidden_size)
        self.context = torch.zeros(1, 512, self.encoder_hidden_size)

        if self.config.use_gpu:
            self.hidden = self.hidden.cuda()
            self.cell = self.cell.cuda()
            self.context = self.context.cuda()

    def forward(self, input_encoded, y_history):
        # input_encoded: (batch_size, T - 1, encoder_hidden_size)
        # y_history: (batch_size, (T-1))
        # Initialize hidden and cell, (1, batch_size, decoder_hidden_size)
        hidden = init_hidden(self.config, input_encoded, self.decoder_hidden_size)
        cell = init_hidden(self.config, input_encoded, self.decoder_hidden_size)
        context = torch.zeros(input_encoded.size(0), self.encoder_hidden_size)  # .cuda()
        # context = torch.cuda.FloatTensor(input_encoded.size(0), self.encoder_hidden_size).fill_(0)
        # size = input_encoded.size(0)
        # self.hidden[:, :size, :] = 0
        # self.cell[:, :size, :] = 0
        # self.context[:size, :] = 0
        # hidden = self.hidden[:, :size, :]
        # cell = self.cell[:, :size, :]
        # context = self.context[:size, :]

        for t in range(self.window):
            # (batch_size, T, (2 * decoder_hidden_size + encoder_hidden_size))
            x = torch.cat((hidden.repeat(self.window, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.window, 1, 1).permute(1, 0, 2),
                           input_encoded), dim=2)
            # Eqn. 12 & 13: softmax on the computed attention weights
            x = tf.softmax(
                self.attn_layer(
                    x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size)
                ).view(-1, self.window),
                dim=1)  # (batch_size, T - 1)

            # Eqn. 14: compute context vector
            context = torch.bmm(x.unsqueeze(1), input_encoded)[:, 0, :]  # (batch_size, encoder_hidden_size)
            # Eqn. 15
            y_tilde = self.fc(torch.cat((context, y_history[:, t]), dim=1))  # (batch_size, out_size)
            # Eqn. 16: LSTM
            self.lstm_layer.flatten_parameters()
            _, lstm_output = self.lstm_layer(y_tilde.unsqueeze(0), (hidden, cell))
            hidden = lstm_output[0]  # 1 * batch_size * decoder_hidden_size
            cell = lstm_output[1]  # 1 * batch_size * decoder_hidden_size

        # Eqn. 22: final output
        return self.fc_final(torch.cat((hidden[0], context), dim=1))


class AttnLSTM(nn.Module):
    def __init__(self, config: Config, input_size, hidden_size, window, num_layers=1):
        super(AttnLSTM, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, 1)
        self.decoder = Decoder(config, encoder_hidden_size=hidden_size, decoder_hidden_size=hidden_size, window=window)

    def forward(self, x):
        """
        :param x: [batch, time step, features]
        :return: [batch, features]
        """
        output, _ = self.encoder(x)
        return self.decoder(output, x)
