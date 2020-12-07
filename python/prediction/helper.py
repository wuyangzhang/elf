import os

import cv2
import numpy as np
import torch


np.set_printoptions(precision=4)

# hyper parameter. filter unmatched RPs.
threshold = 0.02


def rp_index_np(data: np.ndarray) -> np.ndarray:
    """
    Refers to Section 4.2 RP indexing in the paper.
    This function will best match the historical objects with each object in the last frame.

    inputs shape: (batch, temporal seq, rp, features)
    :param data:
    :return:
    """
    if data.ndim == 3:
        data = np.expand_dims(data, axis=0)
    areas = (data[:, :, :, 2] - data[:, :, :, 0]) * (data[:, :, :, 3] - data[:, :, :, 1])
    output = np.zeros(data.shape)
    batch_size = data.shape[0]

    for batch in range(batch_size):
        last_frame = data[batch, -1, :, :]

        # evaluate the change of centroid point
        x_offset = np.abs(np.subtract.outer(
            last_frame[:, 0] + last_frame[:, 2],
            data[batch, :, :, 0] + data[batch, :, :, 2]
        )) / 2

        y_offset = np.abs(np.subtract.outer(
            last_frame[:, 1] + last_frame[:, 3],
            data[batch, :, :, 1] + data[batch, :, :, 3]
        )) / 2

        area_diff = np.abs(np.subtract.outer(
            np.sqrt(areas[batch, -1, :]),
            np.sqrt(areas[batch, :, :])
        ))

        metrics = x_offset + y_offset + area_diff  # + shape_diff

        # Find the smallest metric values
        index = np.argmin(metrics, axis=2).swapaxes(0, 1)

        output[batch, :, :, :] = np.take_along_axis(data[batch], np.expand_dims(index, axis=-1), axis=1)

        # Set a hard threshold for the metric.
        metrics_min = np.min(metrics, axis=2)

        mask = (metrics_min > threshold).swapaxes(0, 1)

        # Reset the RPs running out of the threshold to zero
        output[batch, mask] = np.zeros((1, 4))

        # Post-processing:
        # Corner case 1: if finding any empty RPs in the last frame, we set the RP in prev frames to empty also.
        zero_mask = last_frame[:, 0:4].sum(axis=1) == 0

        output[batch, :, zero_mask] = np.zeros((1, 4))

        # Corner case 2: if locating an non-empty RP in the last frame, but an empty position in prev frames,
        # Set prev frames to the same RP.
        non_zero_mask = (~zero_mask) & (output[batch, :, :, 0:4].sum(axis=2) == 0)
        t, index = np.where(non_zero_mask)
        output[batch, t, index] = last_frame[index]

    return output


def remove_zero_bbox(data, rp_index=True):
    '''
    given a prediction result with the shape of [32, 4], remove zero rps
    :param data: prediction output
    :return:
    '''
    if isinstance(data, np.ndarray):
        if data.ndim == 4:
            time_len = data.shape[1]
            data = np.swapaxes(data, 1, 2).reshape(-1, data.shape[1], data.shape[-1])
            mask = data[:, :, :4].sum(axis=2) != 0.
            # if not rp_index:
            #     if data[mask].shape[0] % time_len != 0:
            #         pad_len = data[mask].shape[0] - data[mask].shape[0] % time_len
            #         data[mask] = np.concatenate((data[mask], np.zeros([pad_len, data.shape[-1]])))
            #         return data[mask].reshape(-1, time_len, data.shape[-1])
            if data[mask].shape[0] % time_len != 0:
                pad_len = time_len * (1 + data[mask].shape[0] // time_len) - data[mask].shape[0]
                return np.concatenate((data[mask], np.zeros([pad_len, data.shape[-1]]))).reshape(-1, time_len,
                                                                                                 data.shape[-1])

            return data[mask].reshape(-1, time_len, data.shape[-1])
        if data.ndim == 3:
            mask = data[:, :, :4].sum(axis=2) != 0.
            return data[mask]
    if isinstance(data, torch.Tensor):
        if data.dim() == 3:
            data = data[0]
    mask = data.sum(axis=1) != 0.
    return data[mask]


def check_label_matching(data):
    '''
    match the rp label along the temporal axis and check the matching accuracy
    :param data: data is in the format of [batch, time, rp, feature]
    :return: acc (%) of correct label matching
    '''

    def _check(test, gt):
        non_zero_mask = gt[:, :, 0:4].sum(axis=2) != 0.0
        match = gt[non_zero_mask][:, -3] == test[non_zero_mask][:, -3]
        return float(sum(match)) / len(match) if len(match) > 0 else 1

    last = data[:, -1, :, :]
    accs = []
    for i in range(1, data.shape[1]):
        test = data[:, i, :, :]
        acc = _check(test, last)
        accs.append(acc)
    return sum(accs) / len(accs)


def get_image(path):
    '''
    fetch an image based on the bbox path
    :param path: bbox path
    :return: image
    '''
    path = path[0]
    path = path.replace('.txt', '.png')
    dir, file = path.split('/')[-2:]
    for im in os.listdir(config.video_dir_roots[0] + dir):
        if im.strip('0') == file:
            image_path = config.video_dir_roots[0] + '/'.join([dir, im])
            return cv2.imread(image_path)


def render_bbox(bbox, image, color=(255, 0, 0), thickness=2) -> np.ndarray:
    '''
    render output bbox on an image
    :param bbox:
    :param image:
    :param color:
    :param thickness:
    :return: a rendered result
    '''
    output = image.copy()
    for box in bbox:
        top_left, bottom_right = (0, 0), (0, 0)
        if isinstance(box, torch.Tensor):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        elif isinstance(box, np.ndarray):
            top_left, bottom_right = box[:2], box[2:]

        output = cv2.rectangle(
            output, tuple(top_left), tuple(bottom_right), color=tuple(color), thickness=thickness,
        )
    return output


def rescale(bbox, shape):
    '''
    given a RP prediction result which in the scale between 0 and 1,
    rescale it based on the image shape
    '''
    bbox[:, 0] *= shape[1]
    bbox[:, 1] *= shape[0]
    bbox[:, 2] *= shape[1]
    bbox[:, 3] *= shape[0]
    return bbox


def show(img, path=None):
    from matplotlib import pyplot as plt

    plt.imshow(img, cmap='gray', interpolation='bicubic')
    if path:
        plt.title(path)
    plt.show()


if __name__ == '__main__':
    from dataset.kitti.bbox_loader import BBoxLoader

    accs = []
    after_accs = []
    for pred in BBoxLoader().get_data_loader():
        acc = check_label_matching(pred)
        accs.append(acc)
        new_acc = check_label_matching(rp_index_np(pred))
        after_accs.append(new_acc)
    print(sum(accs) / len(accs))
    print(sum(after_accs) / len(after_accs))
