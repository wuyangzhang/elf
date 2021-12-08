from cv2 import imshow, waitKey

from dataset.box_loader import BoxLoader

from rp_predict.rp_index import *
from rp_predict.util import *
from rp_predict.model.loss import cal_loss
from rp_predict.model.model_helper import create_model
from config import Config

visualization_mode = True
config = Config()
rp_index = True


@torch.no_grad()
def eval_model():
    model = create_model(
        config,
        load_state=True
    )

    dataset = BoxLoader(
        video_dir_root=config.rp_prediction_eval_dataset_path,
        batch_size=1,
    ).get_data_loader(
        shuffle=False
    )

    all_loss = []
    for data in dataset:
        eval_x, eval_y, path, _ = data
        path = path[-1]

        eval_x = eval_x[:, :, :, :4]
        if rp_index:
            x = index_rps(eval_x)
        else:
            x = eval_x.numpy()

        y = remove_zero_bbox(x[:, -1, ...])
        x = remove_zero_bbox(x[:, :-1, ...])

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        if config.use_gpu:
            x = x.cuda()
            y = y.cuda()

        output = model(x)

        loss, metrics = cal_loss(output, y)
        all_loss.append(loss.item())

        if visualization_mode:
            output = remove_zero_bbox(output)
            image = get_image(path[0])

            output = rescale(output, image.shape)

            target = remove_zero_bbox(y)
            target = rescale(target, image.shape)

            imshow("Predicted RP", render_bbox(output, image))
            imshow("Ground truth RP", render_bbox(target, image))
            waitKey(100000)

    avg = sum(all_loss) / len(all_loss)
    print(f"Average loss is {avg}.")


if __name__ == '__main__':
    eval_model()
