from typing import Optional, Union, List, Tuple

import cv2
import numpy as np
import torch


def crop_frame(
    frame: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int
) -> np.ndarray:
    """
    It creates a np array view instead of a copy to save the expensive copy cost.
    (x0, y0) the coordinate of the left upper point.
    (x1, y1) the coordinate of the right lower point.
    """
    # if x0 < 0 or y0 < 0 or x0 > x1 or y0 > y1 or x1 > get_frame_width(frame) or y1 > get_frame_height(frame):
    #     raise ValueError(f"Invalid crop parameters! Got x0 {x0}, y0 {y0}, x1 {x1}, y1 {y1}!")
    x0 = max(0, x0)

    return frame[
       y0: y1,
       x0: x1,
       :
    ]


def get_frame_height_width(frame: np.ndarray) -> Tuple[int, int]:
    return frame.shape[0], frame.shape[1]


def get_frame_width(frame: np.ndarray) -> int:
    return frame.shape[1]


def get_frame_height(frame: np.ndarray) -> int:
    return frame.shape[0]


def add_offset(
    rp: np.ndarray,
    dx: int,
    dy: int,
) -> np.ndarray:
    """Add offsets to an rp."""
    rp[:, 0] += dx
    rp[:, 1] += dy
    rp[:, 2] += dx
    rp[:, 3] += dy
    return rp


def rescale_frame(
    frame: np.ndarray,
    ratio_x: float,
    ratio_y: Optional[float] = None
) -> np.ndarray:
    if ratio_y is None:
        ratio_y = ratio_x

    width = int(frame.shape[1] * ratio_x)
    height = int(frame.shape[0] * ratio_y)
    if width % 2 != 0:
        width += 1

    if height % 2 != 0:
        height += 1

    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_CUBIC)


def calculate_frame_intersection(
    frame_a: np.ndarray,
    frame_b: np.ndarray
) -> Union[float, int]:
    """Calculate the area intersection of two frames."""
    dx = min(frame_a[2], frame_b[2]) - max(frame_a[0], frame_b[0])
    dy = min(frame_a[3], frame_b[3]) - max(frame_a[1], frame_b[1])
    if (dx >= 0) and (dy >= 0):
        return dx * dy
    return 0


def calculate_partition_area_ratio(
    frame: np.ndarray,
    frame_pars: List[np.ndarray]
) -> List[float]:
    """
    find the size ratio between each sub-frame and its parent frame
    :param frame:
    :param frame_pars:
    :return:
    """
    return [s.size / frame.size for s in frame_pars]


def rp_area(rp: np.ndarray) -> int:
    return (rp[2] - rp[0]) * (rp[3] - rp[1])


def display_imgs(
    imgs: Union[np.ndarray, List[np.ndarray]],
    window_name="img",
):
    if type(imgs) == list:
        for i, img in enumerate(imgs):
            cv2.imshow(f"window_name{i}", img)
    else:
        cv2.imshow(window_name, imgs)


def render_bbox(bbox, image, color=(255, 0, 0), thickness=2) -> np.ndarray:
    """
    render output bbox on an image
    :param bbox:
    :param image:
    :param color:
    :param thickness:
    :return: a rendered result
    """
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
