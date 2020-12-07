from typing import Optional, Union, List

import cv2
import numpy as np


def scale_in(img: np.ndarray, ratio_x: float, ratio_y: Optional[float] = None) -> np.ndarray:
    """
    Scale in the image
    :param img:
    :param ratio_x: scale ratio
    :param ratio_y: y scale ratio
    :return: scaled image
    """
    if ratio_y is None:
        ratio_y = ratio_x
    width = int(img.shape[1] * ratio_x)
    height = int(img.shape[0] * ratio_y)
    if width % 2 != 0:
        width += 1
    if height % 2 != 0:
        height += 1
    dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)


def display_imgs(imgs: Union[np.ndarray, List[np.ndarray]]):
    if type(imgs) == list:
        for i, img in enumerate(imgs):
            cv2.imshow(f"img{i}", img)
    else:
        cv2.imshow("img", imgs)
    cv2.waitKey(10 * 1000)
