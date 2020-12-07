import unittest

import numpy as np
from numpy.testing import assert_array_equal

from partitioning import PartitionAlgorithm
from config import Config


class TestPartitionAlgorithm(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config()
        self.config.rescale_ratio = 0.1

        self.config.frame_height = 1080
        self.config.frame_width = 2560
        self.config.total_remote_servers = 3

    def test_find_rps_boundary(self) -> None:
        rp = np.array([[1, 100, 200, 300], [101, 2, 300, 400], [102, 201, 888, 401], [105, 203, 303, 999]])
        self.assertEqual(PartitionAlgorithm.find_rps_boundary(rp), [1, 2, 888, 999])

    def test_init_rp_boxes(self) -> None:
        assert_array_equal(
            PartitionAlgorithm.init_rp_boxes(4, [0, 0, 100, 100]),
            np.array(
                [
                    [0, 0, 25, 100],
                    [25, 0, 50, 100],
                    [50, 0, 75, 100],
                    [75, 0, 100, 100],
                ]
            )
        )

        assert_array_equal(
            PartitionAlgorithm.init_rp_boxes(3, [20, 20, 1080, 1080]),
            np.array(
                [
                    [20, 20, 373, 1080],
                    [373, 20, 726, 1080],
                    [726, 20, 1079, 1080],
                ]
            )
        )

    def test_find_max_overlaps(self) -> None:
        rps_0 = np.array(
            [
                [50, 50, 300, 800],
                [420, 55, 600, 980],
                [800, 60, 900, 980]
            ]
        )

        rp_boxes_0 = np.array(
            [
                [20, 20, 373, 1080],
                [373, 20, 726, 1080],
                [726, 20, 1079, 1080],
            ]
        )

        assert_array_equal(
            PartitionAlgorithm.find_max_overlaps(rps_0, rp_boxes_0),
            np.array([0, 1, 2])
        )

        rps_1 = np.array(
            [
                [50, 50, 300, 800],
                [323, 55, 383, 980],
                [50, 60, 900, 980]
            ]
        )

        assert_array_equal(
            PartitionAlgorithm.find_max_overlaps(rps_1, rp_boxes_0),
            np.array([0, 0, 1])
        )

    def test_adjust_rp_boxes(self) -> None:
        rps_0 = np.array(
            [
                [50, 50, 300, 800],
                [323, 55, 383, 980],
                [50, 60, 900, 980]
            ]
        )

        rp_boxes_0 = np.array(
            [
                [20, 20, 373, 1080],
                [373, 20, 726, 1080],
                [726, 20, 1079, 1080],
            ]
        )

        index_max_rp_box_overlap = PartitionAlgorithm.find_max_overlaps(rps_0, rp_boxes_0)
        rp_assoc = [rps_0[index_max_rp_box_overlap == i] for i in range(len(rp_boxes_0))]

        assert_array_equal(
            PartitionAlgorithm.adjust_rp_boxes(rps_0, rp_assoc),
            np.array(
                [
                    [50, 50, 383, 980],
                    [50, 60, 900, 980],
                    [0, 0, 5, 5],
                ]
            )
        )

    def test_rescale_rp_boxes(self) -> None:
        rp_boxes = np.array(
            [
                [20, 20, 500, 1000],
                [40, 30, 600, 1000],
                [60, 40, 800, 1200],
            ]
        )

        assert_array_equal(
            PartitionAlgorithm.rescale_rp_boxes(rp_boxes, self.config),
            np.array(
                [
                    [0, 0, 548, 1080],
                    [0, 0, 656, 1080],
                    [0, 0, 874, 1080],
                ]
            )
        )

    def test_reformat(self) -> None:
        pass

    def test_frame_partition(self) -> None:
        np.random.seed(6)
        frame = np.random.rand(2560, 1080, 3)
        par_algo = PartitionAlgorithm()
        rps = np.array(
            [
                [50, 50, 300, 300],
                [323, 55, 383, 980],
                [350, 55, 380, 900],
                [50, 60, 900, 980],
            ]
        )

        frame_pars = par_algo.frame_partition(frame, rps, [self.config])
        self.assertEqual(len(frame_pars), self.config.total_remote_servers)
