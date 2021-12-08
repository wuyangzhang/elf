from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class Offset:
    """Partition offset between left upper corner to (0, 0)."""
    x: Optional[int] = None
    y: Optional[int] = None


@dataclass
class RPPartition:
    """Meta data of a frame partition."""
    partition: np.ndarray
    """Frame partition data."""

    offset: List[int]
    """Partition offset between left upper corner to (0, 0)."""

    frame_index: Optional[int] = None
    """Index of the frame that contains this RP partition."""

    server_id: Optional[str] = None
    """Id of server that will process this RP partition."""

    @property
    def partition_size(self) -> int:
        return self.partition.size

    @property
    def coordinates(self) -> np.ndarray:
        """
        RP partition coordinates (x0, y0, x1, y1).
        :return:
        """
        return np.array(
            [
                self.offset[0],
                self.offset[1],
                self.partition.shape[1] + self.offset[0],
                self.partition.shape[0] + self.offset[1],
            ]
        )


