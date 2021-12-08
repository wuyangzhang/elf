import unittest

import cv2
import numpy as np
from numpy.testing import assert_array_equal

from config import Config
from rp_partition.rp_partitioner import RPPartitioner


class TestRPPartitioner(unittest.TestCase):
    def test_partition_frame(self) -> None:
        np.random.seed(6)
        frame = np.random.rand(2560, 1080, 3)
        rp_partitioner = RPPartitioner(
            config=Config(
                total_partition_num=3,
            )
        )
        rps = np.array(
            [
                [50, 50, 300, 300],
                [323, 55, 383, 980],
                [350, 55, 380, 900],
                [50, 60, 900, 980],
            ]
        )

        frame_pars = rp_partitioner.partition_frame(
            frame,
            rps=rps,
        )

        self.assertEqual(
            len(frame_pars),
            3
        )

        self.assertEqual(
            frame_pars[0].shape,
            (1035, 951, 3),
        )

        self.assertEqual(
            frame_pars[1].shape,
            (1035, 67, 3),
        )

        self.assertEqual(
            frame_pars[2].shape,
            (3, 3, 3),
        )