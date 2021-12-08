import random

import torch.optim

from config import Config
from dataset.box_loader import BoxLoader
from rp_predict.model.loss import cal_loss
from rp_predict.model.model_helper import create_model, save_model
from rp_predict.rp_index import *

random.seed(6)
np.random.seed(6)
config: Config = Config()

LEARNING_RATE: float = 1e-3

print_freq: int = 5
save_epoch_freq: int = 5
total_epoch: int = 1000


def train_model():
    # Loading dataset.
    data_loader = BoxLoader().get_data_loader()

    # Model selection.
    net = create_model(
        config,
        load_state=False
    )
    net.train()

    # Optimizer & learning rate.
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=LEARNING_RATE
    )

    for epoch in range(total_epoch):
        for batch_id, data in enumerate(data_loader):

            train_x, train_y, _, _ = data
            train_x = train_x[:, :, :, :4]
            x = index_rps(train_x)

            y = remove_zero_bbox(x[:, -1, ...])[:, :4]
            x = remove_zero_bbox(x[:, :-1, ...])[:, :, :4]

            x = torch.from_numpy(x).float()
            y = torch.from_numpy(y).float()
            if config.use_gpu:
                x = x.cuda()
                y = y.cuda()

            optimizer.zero_grad()

            output = net(x)

            loss, metrics = cal_loss(output, y)

            loss.backward()

            optimizer.step()

            if batch_id % print_freq == 0:
                print(f"Epoch: {epoch}, batch {batch_id}, loss:{loss.item()}")

        if epoch % save_epoch_freq == 0:
            save_model(config, net, epoch)


if __name__ == '__main__':
    train_model()
