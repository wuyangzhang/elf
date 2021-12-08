from typing import List

import numpy as np


def schedule_offloading(
    rp_boxes: List[np.ndarray]
) -> List[np.ndarray]:
    """Match each rp box with an offloading server.
    Reorder RP boxes in the server order.
    """
    return prioritize_larger_rp(rp_boxes)


def prioritize_larger_rp(
    rp_boxes: List[np.ndarray]
) -> List[np.ndarray]:
    return sorted(
        rp_boxes,
        key=lambda x: x.size,
        reverse=True
    )
