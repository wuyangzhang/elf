import os

import torch
from torch import nn

from config import Config
from prediction.model.atn_lstm import AttnLSTM
from prediction.model.lstm import LSTM


def create_model(config: Config, load_state: bool = False):
    net: nn.Module = None
    if config.prediction_model == "lstm":
        net = LSTM(input_size=4, hidden_size=16, window=config.window_size - 1, num_layers=2)
    elif config.prediction_model == "attn":
        net = AttnLSTM(config, input_size=4, hidden_size=8, window=config.window_size - 1, num_layers=1)

    assert net is not None, "Fail to find the model {}".format(config.prediction_model)

    if load_state:
        assert os.path.exists(config.prediction_model_path), "cannot load model from {}".format(config.prediction_model_path)
        net.load_state_dict(torch.load(config.prediction_model_path))

    if config.use_cuda:
        return net.cuda()
    else:
        return net


def save_model(config: Config, net, epoch):
    if config.prediction_model == "lstm":
        torch.save(net.state_dict(), "model/outputs/lstm_single_checkpoint{}.pth".format(epoch))
        print("save model/outputs/lstm_single_checkpoint{}.pth".format(epoch))
    elif config.prediction_model == 'attn':
        torch.save(net.state_dict(), "model/outputs/attn_lstm_checkpoint{}.pth".format(epoch))
        print("save model/outputs/tmp/attn_lstm_checkpoint{}.pth".format(epoch))


def convert_jit_trace(config: Config):
    model: nn.Module = create_model(config, False)
    example: torch.Tensor = torch.rand(8, 2, 4)
    jit_model = torch.jit.trace(model, example)
    return jit_model
