import numpy as np

from util.helper import rescale_frame


def create_lrc_frame(
    frame: np.ndarray,
    lrc_downsample_ratio: float,
) -> np.ndarray:
    return rescale_frame(
        frame,
        lrc_downsample_ratio,
    )


def rescale_lrc_results(
    rps: np.ndarray,
    lrc_downsample_ratio: float,
) -> np.ndarray:
    """LRC down-samples an original frame for model inference.
    To get the matched RP results on the original frame, we need to rescale it back.
    """
    return rps / lrc_downsample_ratio
