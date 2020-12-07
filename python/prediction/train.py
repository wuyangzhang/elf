"""
Predict coordinate distribution of region proposals in the next frame.
"""
import random

import torch.optim

from prediction.helper import *
from prediction.model.loss import cal_loss
from prediction.model.model_helper import create_model, save_model

random.seed(6)
np.random.seed(6)


def train_model():
    from dataset.kitti.bbox_loader import BBoxLoader
    from config import Config

    config: Config = Config()
    # Loading dataset
    config.video_dir_root = config.video_dir_roots[1]
    data_loader = BBoxLoader().get_data_loader()

    # Model selection
    net = create_model(config, load_state=False)
    net.train()

    # Optimizer & learning rate
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    print_freq: int = 10
    save_epoch_freq: int = 5
    total_epoch: int = 1000

    for epoch in range(total_epoch):
        for batch_id, data in enumerate(data_loader):

            train_x, train_y, _, _ = data
            train_x = train_x[:, :, :, :4]
            x = rp_index_np(train_x)
            y = remove_zero_bbox(x[:, -1, ...])[:, :4]
            x = remove_zero_bbox(x[:, :-1, ...])[:, :, :4]

            x = torch.from_numpy(x).float()
            y = torch.from_numpy(y).float()
            if config.use_cuda:
                x = x.cuda()
                y = y.cuda()

            optimizer.zero_grad()

            output = net(x)

            loss, metrics = cal_loss(output, y)

            loss.backward()

            optimizer.step()

            if batch_id % print_freq == 0:
                print('Epoch: {}, batch {}, '
                      'loss:{:.5f}, '
                    .format(
                    epoch + 1, batch_id + 1,
                    loss.item()))

        if epoch % save_epoch_freq == 0:
            save_model(config, net, epoch)


if __name__ == '__main__':
    train_model()
