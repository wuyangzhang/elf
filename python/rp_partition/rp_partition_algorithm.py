from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from config import Config
from rp_partition.rp_partition import RPPartition, Offset
from util.helper import get_frame_height, get_frame_width, get_frame_height_width, crop_frame


class PartitionAlgorithmBase(ABC):
    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def partition_frame(
        self,
        frame: np.ndarray,
        **kwargs,
    ) -> List[RPPartition]:
        raise NotImplementedError


class EqualPartition(PartitionAlgorithmBase):
    """Partition a frame from the vertical way equally."""

    def partition_frame(
        self,
        frame: np.ndarray,
        **kwargs,
    ) -> List[RPPartition]:
        frame_height, frame_width = get_frame_height_width(frame)

        total_partition_num = self.config.total_partition_num
        if total_partition_num < 1:
            raise ValueError(f"Total partition number should be larger than 0! Got {total_partition_num}.")

        partition_height, partition_width = EqualPartition._get_partition_height_width(
            frame_height,
            frame_width,
            total_partition_num=total_partition_num,
        )

        return [
            EqualPartition._generate_equal_partition(
                frame=frame,
                index=index,
                partition_height=partition_height,
                partition_width=partition_width,
            ) for index in range(total_partition_num)
        ]

    @staticmethod
    def _generate_equal_partition(
        frame: np.ndarray,
        index: int,
        partition_height: int,
        partition_width: int
    ) -> RPPartition:
        return RPPartition(
            partition=crop_frame(
                frame=frame,
                x0=0,
                y0=partition_height * index,
                x1=partition_width,
                y1=partition_height * (index + 1)
            ),
            offset=[
                0,
                partition_height * index,
            ],
        )

    @staticmethod
    def _get_partition_height_width(
        frame_height: int,
        frame_width: int,
        total_partition_num: int
    ) -> Tuple[int, int]:
        if total_partition_num < 1:
            raise ValueError(f"Total partition number cannot be smaller than 1. Got {total_partition_num}.")

        if frame_height < total_partition_num:
            raise ValueError(
                f"Total partition number {total_partition_num} cannot be larger than frame height {frame_height}!"
            )

        return frame_height // total_partition_num, frame_width


class RPAwarePartition(PartitionAlgorithmBase):
    def partition_frame(
        self,
        frame: np.ndarray,
        **kwargs,
    ) -> List[RPPartition]:
        """An RP-aware frame partition scheme.

        :param frame: the target frame to be partition.
        :param rps: all the region proposals along with their coordinates
        :return N partitions
        """
        rps = kwargs.pop("rps")
        if rps is None:
            raise ValueError("RPs is required!")

        rp_boxes = self.create_rp_boxes(
            self.config.total_partition_num,
            self.find_rps_periphery(
                rps
            )
        )

        index_max_overlaps = self.find_max_overlaps(
            rps,
            rp_boxes
        )

        contained_rps = self.find_contained_rps(
            rps,
            rp_boxes,
            index_max_overlaps,
        )

        rp_boxes = self.adjust_rp_boxes(
            rp_boxes,
            contained_rps
        )

        rp_boxes = self.expand_rp_boxes(
            rp_boxes,
            self.config.rp_extend_ratio,
            self.config.rp_extend_ratio,
            get_frame_height(frame),
            get_frame_width(frame),
        )

        return [
            RPPartition(
                partition=crop_frame(
                    frame=frame,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1
                ),
                offset=[
                    x0,
                    y0
                ],
            ) for x0, y0, x1, y1 in rp_boxes
        ]

    @staticmethod
    def find_rps_periphery(rps: np.ndarray) -> Tuple[int, int, int, int]:
        """Find the external periphery covering all the rps."""
        min_rp_x0 = np.min(rps[:, 0])
        min_rp_y0 = np.min(rps[:, 1])
        max_rp_x1 = np.max(rps[:, 2])
        max_rp_y1 = np.max(rps[:, 3])

        return min_rp_x0, min_rp_y0, max_rp_x1, max_rp_y1

    @staticmethod
    def create_rp_boxes(
        total_rp_box_num: int,
        rp_periphery: Tuple[int, int, int, int],
    ) -> np.array:
        """
        Create rp boxes based on rp periphery:
        It partitions the rp_periphery area horizontally to #total_rp_box_num sub-areas.
        """
        if total_rp_box_num < 1:
            raise ValueError(f"Total RP box number should be larger than 1! Got {total_rp_box_num}.")

        min_rp_x0, min_rp_y0, max_rp_x1, max_rp_y1 = rp_periphery

        height_unit, width_unit = max_rp_y1 - min_rp_y0, (max_rp_x1 - min_rp_x0) // total_rp_box_num

        rp_boxes = np.zeros(
            (
                total_rp_box_num,
                4,
            )
        )

        for i in range(total_rp_box_num):
            rp_boxes[i] = np.array(
                [
                    i * width_unit + min_rp_x0,
                    min_rp_y0,
                    (i + 1) * width_unit + min_rp_x0,
                    height_unit + min_rp_y0,
                ]
            )

        return rp_boxes

    @staticmethod
    def find_contained_rps(
        rps: np.ndarray,
        rp_boxes: np.ndarray,
        index_max_overlaps: np.ndarray,
    ) -> List[np.ndarray]:
        contained_rps = []

        for i in range(len(rp_boxes)):
            contained_rps.append(
                rps[
                    index_max_overlaps == i
                    ]
            )
        return contained_rps

    @staticmethod
    def find_max_overlaps(
        rps: np.ndarray,
        rp_boxes: np.ndarray
    ) -> np.ndarray:
        """
        For each RP, it finds which rp boxes sharing the maximal overlap.
        :returns 1d np array. Each value is the index of the RP box sharing the maximal overlap.
        """
        a = np.maximum(rps[:, None, 0], rp_boxes[:, 0])
        c = np.minimum(rps[:, None, 2], rp_boxes[:, 2])
        max_par_index = np.argmax(c - a, axis=1)

        return max_par_index

    @staticmethod
    def adjust_rp_boxes(
        rp_boxes: np.ndarray,
        contained_rps: List[np.ndarray]
    ) -> np.ndarray:
        """Rescale each rp_partition box in order to fully cover its contained RPs."""
        for i in range(len(contained_rps)):
            if len(contained_rps[i]) == 0:
                """If there is no RP contained in a box, just rescale this box to an ingorable block."""
                rp_boxes[i] = np.array([0, 0, 3, 3])
                continue
            rp_boxes[i, :2] = np.min(contained_rps[i][:, :2], axis=0)
            rp_boxes[i, 2:] = np.max(contained_rps[i][:, 2:], axis=0)

        return rp_boxes

    def expand_rp_boxes(
        self,
        rp_boxes: np.ndarray,
        width_rescale_ratio: float,
        height_rescale_ratio: float,
        max_height: int,
        max_width: int,
    ) -> np.ndarray:
        rp_width = (rp_boxes[:, 2] - rp_boxes[:, 0]) * width_rescale_ratio
        rp_width = np.maximum(rp_width, self.config.min_rp_rescale_ratio * max_width)
        rp_height = (rp_boxes[:, 3] - rp_boxes[:, 1]) * height_rescale_ratio
        rp_height = np.maximum(rp_height, self.config.min_rp_rescale_ratio * max_height)

        rp_boxes[:, 0] = np.maximum(0, rp_boxes[:, 0] - rp_width)
        rp_boxes[:, 1] = np.maximum(0, rp_boxes[:, 1] - rp_height)
        rp_boxes[:, 2] = np.minimum(max_width, rp_boxes[:, 2] + rp_width)
        rp_boxes[:, 3] = np.minimum(max_height, rp_boxes[:, 3] + rp_height)

        return rp_boxes.astype(int)

    @staticmethod
    def prepare_nvjpeg_encode_input(frame: np.ndarray) -> np.ndarray:
        """
        This function is specifically designed for the usage of nvJPEG encoder.
        :param frame:
        :return:
        """
        return frame[
               :frame.shape[0] - int(frame.shape[0] % 2 != 0),
               :frame.shape[1] - int(frame.shape[1] % 2 != 0),
               :
               ].copy()
