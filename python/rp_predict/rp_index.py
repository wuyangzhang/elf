from typing import Union, Iterable, Tuple

import numpy as np
import torch

np.set_printoptions(precision=4)

RP_OFFSET_THRESHOLD = 0.02
"""Hyper parameter to filter unmatched RPs."""


def index_rps(
        rps: np.ndarray
) -> np.ndarray:
    """
    Refers to Section 4.2 RP indexing in the paper.
    This function will best match the historical objects with each object in the last frame.

    inputs shape: (batch, temporal seq, rp, features)
    :param rps:
    :return:
    """

    if rps.ndim == 3:
        rps = np.expand_dims(rps, axis=0)

    areas = (rps[:, :, :, 2] - rps[:, :, :, 0]) * (rps[:, :, :, 3] - rps[:, :, :, 1])
    output = np.zeros(rps.shape)
    batch_size = rps.shape[0]

    for batch in range(batch_size):
        last_frame = rps[batch, -1, :, :]

        # Evaluate the change of centroid point.
        x_offset = np.abs(np.subtract.outer(
            last_frame[:, 0] + last_frame[:, 2],
            rps[batch, :, :, 0] + rps[batch, :, :, 2]
        )) / 2

        y_offset = np.abs(np.subtract.outer(
            last_frame[:, 1] + last_frame[:, 3],
            rps[batch, :, :, 1] + rps[batch, :, :, 3]
        )) / 2

        # Evaluate the change of size.
        area_diff = np.abs(np.subtract.outer(
            np.sqrt(areas[batch, -1, :]),
            np.sqrt(areas[batch, :, :])
        ))

        # Should square area_diff
        metrics = x_offset + y_offset + area_diff  # + shape_diff

        # find the smallest metric values
        index = np.argmin(metrics, axis=2).swapaxes(0, 1)

        # assign the updated result. significant bug.
        output[batch, :, :, :] = np.take_along_axis(rps[batch], np.expand_dims(index, axis=-1), axis=1)

        # set a hard threshold for the metric.
        metrics_min = np.min(metrics, axis=2)

        mask = (metrics_min > RP_OFFSET_THRESHOLD).swapaxes(0, 1)
        # reset the RPs running out of the threshold to zero
        output[batch, mask] = np.zeros((1, 4))

        # post-processing:
        # corner case 1: if finding any empty RPs in the last frame, we set the RP in prev frames to empty also.
        zero_mask = last_frame[:, 0:4].sum(axis=1) == 0

        # 7 is the total feature number, may accordingly change based on the input.
        output[batch, :, zero_mask] = np.zeros((1, 4))

        # corner case 2: if locating an non-empty RP in the last frame, but an empty position in prev frames,
        # set prev frames to the same RP.
        non_zero_mask = (~zero_mask) & (output[batch, :, :, 0:4].sum(axis=2) == 0)
        t, index = np.where(non_zero_mask)
        output[batch, t, index] = last_frame[index]

    return output


def index_rps_local(
        rps: np.ndarray
) -> np.ndarray:
    """
    This function best matches the historical objects with each object in the last frame.
    Please Refers to Section 4.2 RP indexing in the paper.
    :param rps: historical RPs. Shape: (batch, temporal sequence, rp, rp coordinates)
    :return: reordered RP in the last frame. Shape: (batch, RP, RP coordinates)
    """
    rps = check_and_expand_rp_dimension(rps)

    output_rps = rps

    last_frame = rps[:, -1, :, :]

    previous_frame = rps[:, -2, :, :]

    x_offset = calculate_x_offset(
        previous_frame,
        last_frame,
    )

    y_offset = calculate_y_offset(
        previous_frame,
        last_frame,
    )

    area_offset = calculate_area_offset(
        previous_frame,
        last_frame,
    )

    total_offsets = sum_offsets(
        x_offset=x_offset,
        y_offset=y_offset,
        area_offset=area_offset,
    )

    minimal_offset_index, minimal_offsets = find_minimal_offset(
        total_offsets
    )

    updated_last_frame = reorder_rps_in_last_frames(
        last_frame,
        minimal_offset_index,
    )

    mask = minimal_offsets > RP_OFFSET_THRESHOLD

    updated_last_frame = updated_last_frame

    # Reset the RPs whose offsets are above the threshold to zero.
    updated_last_frame[mask,] = 0

    output_rps = post_rp_index_process(
        updated_last_frame,
        output_rps,
    )

    return output_rps


def remove_zero_bbox(rps, rp_index=True):
    """
    Given a prediction result with the fixed shape, it removes zero rps.
    :param rps:
    :return: rps with removed zeros
    """
    if isinstance(rps, np.ndarray):
        if rps.ndim == 4:
            time_len = rps.shape[1]
            rps = np.swapaxes(rps, 1, 2).reshape(-1, rps.shape[1], rps.shape[-1])
            mask = rps[:, :, :4].sum(axis=2) != 0.

            if rps[mask].shape[0] % time_len != 0:
                pad_len = time_len * (1 + rps[mask].shape[0] // time_len) - rps[mask].shape[0]
                return np.concatenate((rps[mask], np.zeros([pad_len, rps.shape[-1]]))).reshape(-1, time_len,
                                                                                               rps.shape[-1])

            return rps[mask].reshape(-1, time_len, rps.shape[-1])

        if rps.ndim == 3:
            mask = rps[:, :, :4].sum(axis=2) != 0.
            return rps[mask]

    elif isinstance(rps, torch.Tensor):
        if rps.dim() == 3:
            rps = rps[0]

    mask = rps.sum(axis=1) != 0.
    return rps[mask]


def check_and_expand_rp_dimension(rps: np.ndarray) -> np.ndarray:
    """
    Expand the dimension of RP to four to fit RP indexing.
    """
    if rps.ndim == 3:
        return np.expand_dims(
            rps,
            axis=0,
        )

    return rps


def calculate_x_offset(
        rps_in_previous_frame: np.ndarray,
        rps_in_last_frame: np.ndarray,
) -> np.ndarray:
    """
    Calculate x offset.
    :param rps_in_previous_frame: Shape (batch, #RP, RP coordinates)
    :param rps_in_last_frame: Shape (batch, #RP, RP coordinate)
    :return: Shape (batch, #RP_previous_frame, 1, #RP_last_frame)
    """
    x_offset = np.abs(
        np.subtract.outer(
            rps_in_last_frame[:, :, 0] + rps_in_last_frame[:, :, 2],
            rps_in_previous_frame[:, :, 0] + rps_in_previous_frame[:, :, 2],
        )
    ) / 2
    return x_offset


def calculate_y_offset(
        rps_in_previous_frame: np.ndarray,
        rps_in_last_frame: np.ndarray,
) -> np.ndarray:
    """
    Calculate y offset.
    :param rps_in_previous_frame: Shape (batch, #RP, RP coordinates)
    :param rps_in_last_frame: Shape (batch, #RP, RP coordinate)
    :return: Shape (batch, #RP_previous_frame, 1, #RP_last_frame)
    """
    y_offset = np.abs(
        np.subtract.outer(
            rps_in_previous_frame[:, :, 1] + rps_in_previous_frame[:, :, 3],
            rps_in_last_frame[:, :, 1] + rps_in_last_frame[:, :, 3],
        )
    ) / 2
    return y_offset


def calculate_area_offset(
        rps_in_previous_frame: np.ndarray,
        rps_in_last_frame: np.ndarray,
) -> np.ndarray:
    """
    Calculate area offset.
    :param rps_in_previous_frame: Shape (batch, #RP, RP coordinates)
    :param rps_in_last_frame: Shape (batch, #RP, RP coordinate)
    :return: Shape (batch, #RP_last_frame, 1, #RP_previous_frame)
    """
    area_offset = np.abs(
        np.subtract.outer(
            np.sqrt(
                calculate_rp_areas(
                    rps_in_previous_frame,
                )
            ),
            np.sqrt(
                calculate_rp_areas(
                    rps_in_last_frame,
                )
            ),
        )
    )
    return area_offset


def calculate_rp_areas(rps: np.ndarray) -> np.ndarray:
    """
    Calculate the areas of RPs.
    :param rps: shape (batch, #RP, RP coordinates)
    :return: shape (batch, #RP, area)
    """
    areas = (
                    rps[:, :, 2] - rps[:, :, 0]
            ) * (
                    rps[:, :, 3] - rps[:, :, 1]
            )
    return areas


def sum_offsets(
        x_offset: np.ndarray,
        y_offset: np.ndarray,
        area_offset: np.ndarray,
) -> np.ndarray:
    """
    Add all offsets together.
    :param x_offset:
    :param y_offset:
    :param area_offset:
    :return: Shape (batch, #RP_last_frame, 1, #RP_previous_frame)
    """
    return x_offset + y_offset + area_offset


def find_minimal_offset(offsets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given the offsets between each last frame and each previous frame,
    it finds the RP in the previous frame with the minimal offset.
    :param offsets: Offsets: shape (batch, #RP_last_frame, 1, #RP_previous_frame)
    :return: index of minimal offsets: shape (batch, #RP_last_frame, 1) and
    minimal offsets: shape (batch, #RP_last_frame, 1)
    """
    index = np.argmin(offsets, axis=3)
    min_offsets = np.min(offsets, axis=3)
    return index, min_offsets


def reorder_rps_in_last_frames(rps: np.ndarray, index: np.ndarray) -> np.ndarray:
    """
    Reorder RPs in the last frame that have the minimal offsets in the previous one.
    :return: shape (batch, #RP, RP coordinates)
    """
    return rps[
           :,
           index,
           :,
           ].squeeze(
        axis=1
    ).squeeze(
        axis=2
    )


def mask_large_offsets(
        offsets: np.ndarray,
        threshold: Union[
            float,
            Iterable[float]
        ]
):
    """
    Given offsets and a threshold, mask the offsets that are higher than the threshold.
    :param offsets: shape (batch, #RP, RP coordinates)
    :param threshold
    :return: masked offsets
    """
    metrics_min = np.min(offsets, axis=2)

    return (metrics_min > threshold).swapaxes(0, 1)


def post_rp_index_process(
        last_frame: np.ndarray,
        output_rps: np.ndarray,
) -> np.ndarray:
    """Post-process RP indexing.
    :param last_frame: shape (batch, #RP, RP coordinates)
    :param output_rps: shape (batch, temporal, #RP, RP coordinates)
    """
    output_rps[:, -1, :, :] = last_frame

    # Corner case 1: if finding any zero RPs in the last frame,
    # we set the RP in prev frames with the same index to zero.
    # This means an RP may disappear in the last frame.
    zero_mask = last_frame[:, ].sum(axis=2) == 0

    output_rps[batch, :, zero_mask] = np.zeros((1, 4))

    # Corner case 2: if locating an non-empty RP in the last frame, but an empty position in prev frames,
    # we set prev frames to the same RP.
    non_zero_mask = (~zero_mask) & (output_rps[batch, :, :, 0:4].sum(axis=2) == 0)

    t, index = np.where(non_zero_mask)

    output[batch, t, index] = last_frame[index]

    return updated_last_frame
