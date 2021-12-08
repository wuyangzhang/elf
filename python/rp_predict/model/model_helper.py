import os

import torch
from torch import nn

from config import Config
from rp_predict.model.atn_lstm import AttnLSTM
from rp_predict.model.lstm import LSTM


def create_model(
        config: Config,
        load_state: bool = False
):
    net: nn.Module
    if config.rp_prediction_model == "lstm":
        net = LSTM(
            input_size=4,
            hidden_size=16,
            window=4,
            num_layers=2
        )

    elif config.rp_prediction_model == "attn_lstm":
        net = AttnLSTM(
            config,
            input_size=4,
            hidden_size=8,
            window=config.rp_predict_window_size - 1,
            num_layers=1
        )

    if load_state:
        if not os.path.exists(config.rp_prediction_model_path):
            raise ValueError(f"Cannot find the path {config.rp_prediction_model_path}")

        net.load_state_dict(
            torch.load(
                config.rp_prediction_model_path
            )
        )

    if config.use_gpu:
        return net.cuda()

    return net


def save_model(
        config: Config,
        net,
        epoch
):
    if config.rp_prediction_model == "lstm":
        torch.save(
            net.state_dict(),
            "rp_predict/model/outputs/lstm_single_checkpoint{}.pth".format(epoch)
        )
        print(f"Saved model/outputs/lstm_single_checkpoint{epoch}.pth")

    elif config.rp_prediction_model == 'attn_lstm':
        torch.save(
            net.state_dict(),
            "rp_predict/model/outputs/attn_lstm_checkpoint{}.pth".format(epoch)
        )
        print(f"Saved model/outputs/tmp/attn_lstm_checkpoint{epoch}.pth")


def convert_jit_trace(config: Config):
    model: nn.Module = create_model(config, False)
    example: torch.Tensor = torch.rand(8, 2, 4)
    jit_model = torch.jit.trace(model, example)
    return jit_model
