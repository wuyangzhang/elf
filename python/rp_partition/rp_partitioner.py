from typing import List, Any

import numpy as np

from config import Config
from rp_partition.rp_partition_algorithm import PartitionAlgorithmBase, RPAwarePartition, RPPartition


class RPPartitioner:
    def __init__(
        self,
        config: Config = Config()
    ) -> None:
        self.config: Config = config

        self.partition_algorithm: PartitionAlgorithmBase = RPAwarePartition(config)

        self._frame_partitions: List[np.ndarray] = list()
        """Frame partitions after splitting a frame."""

    def partition_frame(
        self,
        frame: np.ndarray,
        rps: np.ndarray
    ) -> List[RPPartition]:
        """
        Partition a frame based on existing RPs.
        """
        self._frame_partitions = self.partition_algorithm.partition_frame(
            frame,
            rps=rps,
        )

        return self._frame_partitions

    @property
    def frame_partitions(self) -> List[np.ndarray]:
        return self._frame_partitions

    @frame_partitions.setter
    def frame_partitions(
        self,
        frame_partitions: List[np.ndarray]
    ):
        self._frame_partitions = frame_partitions

    @property
    def partition_assistant_data(self) -> List[Any]:
        """
        Prepare assisting data that helps to perform frame partition.
        For example, server resource availability, or historical server processing latency.
        """
        return [self.config]
