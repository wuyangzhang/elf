import os

from typing import List

import cv2
import numpy as np
import torch

from config import Config

config = Config()


def load_tensors(path: str) -> List[torch.tensor]:
    """load ground truth RPs of video frames"""
    root_path = path
    files = os.listdir(root_path)
    t = {}
    for f in files:
        index = int(f.split('.')[0].split('_')[1])
        t[index] = f
    ans = []
    for index in sorted(t):
        path = root_path + t[index]
        ans.append(load_tensor(path))
    return ans


def load_tensor(file_path: str) -> torch.tensor:
    return torch.load(file_path)


def scale_tensor(tensors: List[torch.tensor], scale_ratio_x: float, scale_ratio_y: float):
    for i in range(len(tensors)):
        tensors[i][:, 0] = tensors[i][:, 0] * scale_ratio_x
        tensors[i][:, 1] = tensors[i][:, 1] * scale_ratio_y
        tensors[i][:, 2] = tensors[i][:, 2] * scale_ratio_x
        tensors[i][:, 3] = tensors[i][:, 3] * scale_ratio_y


def box_area(boxes: np.ndarray):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Arguments: boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format

    Returns: area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def calculate_rps_iou(rps_0: np.ndarray, rps_1: np.ndarray) -> np.ndarray:
    """
    :param rps_0: (n, 4)
    :param rps_1 (m, 4)
    :returns: (n, m)
    Copy from https://www.codeleading.com/article/54902955171/
    """
    lt = np.maximum(rps_0[:, None, :2], rps_1[:, :2])  # left_top (x, y)
    rb = np.minimum(rps_0[:, None, 2:], rps_1[:, 2:])  # right_bottom (x, y)
    wh = np.maximum(rb - lt + 1, 0)  # inter_area (w, h)
    inter_areas = wh[:, :, 0] * wh[:, :, 1]  # shape: (n, m)
    box_areas = (rps_0[:, 2] - rps_0[:, 0] + 1) * (rps_0[:, 3] - rps_0[:, 1] + 1)
    gt_areas = (rps_1[:, 2] - rps_1[:, 0] + 1) * (rps_1[:, 3] - rps_1[:, 1] + 1)
    iou = inter_areas / (box_areas[:, None] + gt_areas - inter_areas)
    return np.round(iou, 3)


def remove_zero_bbox(data: np.ndarray):
    """
    Given a rp_predict result with the shape of [32, 4], remove zero rps
    :param data: rp_predict output
    :return:
    """

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

    mask = data.sum(axis=1) != 0.
    return data[mask]


def check_label_matching(data):
    """
    Match the rp label along the temporal axis and check the matching accuracy
    :param data: data is in the format of [batch, time, rp, feature]
    :return: acc (%) of correct label matching
    """
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
    """
    Fetch an image based on the bbox path.
    :param path: bbox path
    :return: image
    """
    path = path.replace('.txt', '.png')
    dir, file = path.split('/')[-2:]
    for image in os.listdir(config.video_dataset_dir + dir):
        if image.strip('0') == file:
            image_path = config.video_dataset_dir + '/'.join([dir, image])
            return cv2.imread(image_path)


def rescale(bbox, shape):
    """
    Given a RP rp_predict result which in the scale between 0 and 1,
    this function rescales it based on the original image shape.
    """
    bbox[:, 0] *= shape[1]
    bbox[:, 1] *= shape[0]
    bbox[:, 2] *= shape[1]
    bbox[:, 3] *= shape[0]

    return bbox


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


def show(img, path=None):
    from matplotlib import pyplot as plt

    plt.imshow(img, cmap='gray', interpolation='bicubic')
    if path:
        plt.title(path)
    plt.show()
