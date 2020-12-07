import os
from typing import List

import torch


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


def box_area(boxes: torch.tensor):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Arguments: boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format

    Returns: area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1: torch.tensor, boxes2: torch.tensor):
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou
