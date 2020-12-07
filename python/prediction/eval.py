from dataset.box_loader import BBoxLoader

from prediction.helper import *
from prediction.model.loss import cal_loss
from prediction.model.model_helper import create_model

vis = False
config = Config()
rp_index = True


@torch.no_grad()
def eval():
    model = create_model(config, load_state=True)
    config.batch_size = 1
    config.video_dir_root = config.video_dir_roots[2]
    dataset = BBoxLoader().get_data_loader(shuffle=False)

    all_loss = []
    for data in dataset:
        train_x, train_y, path, _ = data
        path = path[-1]

        train_x = train_x[:, :, :, :4]
        if rp_index:
            x = rp_index_np(train_x)
        else:
            x = train_x.numpy()

        y = remove_zero_bbox(x[:, -1, ...])
        x = remove_zero_bbox(x[:, :-1, ...])

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        if config.cuda:
            x = x.cuda()
            y = y.cuda()

        output = model(x)

        loss, metrics = cal_loss(output, y)
        all_loss.append(loss.item())

        if vis:
            output = remove_zero_bbox(output)
            img = get_image(path)
            output = rescale(output, img.shape)

            target = remove_zero_bbox(y)
            target = rescale(target, img.shape)

            cv2.imshow('pred', render_bbox(output, img))
            cv2.imshow('gt', render_bbox(target, img))
            cv2.waitKey(100000)

    avg = sum(all_loss) / len(all_loss)
    print('avg loss', avg)


def write(path, loss):
    with open(path, 'w') as f:
        for l in loss:
            f.write('%s\n' % l)


if __name__ == '__main__':
    eval()
