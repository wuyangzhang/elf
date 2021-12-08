import unittest

import cv2
import numpy as np
from numpy.testing import assert_array_equal

from config import Config
from rp_partition.rp_partition import RPPartition, Offset
from rp_partition.rp_partition_algorithm import EqualPartition, RPAwarePartition
from util.helper import get_frame_height, get_frame_width

TEST_IMAGE_PATH = '/Users/wuyang/python/elf/python/tests/test_images/test_image_0.jpeg'


class TestEqualPartition(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = cv2.imread(TEST_IMAGE_PATH)

    def test_get_partition_height_width(self) -> None:
        partition_shape = EqualPartition._get_partition_height_width(
            frame_height=get_frame_height(self.frame),
            frame_width=get_frame_width(self.frame),
            total_partition_num=2,
        )

        self.assertEqual(
            partition_shape,
            (
                get_frame_height(self.frame) // 2,
                get_frame_width(self.frame),
            )
        )

        partition_shape = EqualPartition._get_partition_height_width(
            frame_height=get_frame_height(self.frame),
            frame_width=get_frame_width(self.frame),
            total_partition_num=4,
        )

        self.assertEqual(
            partition_shape,
            (
                get_frame_height(self.frame) // 4,
                get_frame_width(self.frame),
            )
        )

        partition_shape = EqualPartition._get_partition_height_width(
            frame_height=get_frame_height(self.frame),
            frame_width=get_frame_width(self.frame),
            total_partition_num=8,
        )

        self.assertEqual(
            partition_shape,
            (
                get_frame_height(self.frame) // 8,
                get_frame_width(self.frame),
            )
        )

    def test_get_partition_height_width_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            EqualPartition._get_partition_height_width(
                frame_height=get_frame_height(self.frame),
                frame_width=get_frame_width(self.frame),
                total_partition_num=0,
            )

        with self.assertRaises(ValueError):
            EqualPartition._get_partition_height_width(
                frame_height=get_frame_height(self.frame),
                frame_width=get_frame_width(self.frame),
                total_partition_num=-1,
            )

        with self.assertRaises(ValueError):
            EqualPartition._get_partition_height_width(
                frame_height=get_frame_height(self.frame),
                frame_width=get_frame_width(self.frame),
                total_partition_num=get_frame_height(self.frame) + 1,
            )

    def test_generate_equal_partition(self) -> None:
        rp_partition: RPPartition = EqualPartition._generate_equal_partition(
            frame=self.frame,
            index=1,
            partition_height=get_frame_height(self.frame) // 2,
            partition_width=get_frame_width(self.frame),
        )

        self.assertEqual(
            rp_partition.offset,
            Offset(0, 225),
        )

        self.assertEqual(
            rp_partition.partition.shape,
            (225, 880, 3),
        )

        self.assertEqual(
            rp_partition.partition_size,
            225 * get_frame_width(self.frame) * 3,
        )

    def test_partition_frame(self) -> None:
        config = Config(
            total_partition_num=2
        )

        partitions = EqualPartition(config).partition_frame(
            self.frame
        )

        self.assertEqual(
            partitions[0].partition.shape,
            (225, 880, 3)
        )

        self.assertEqual(
            partitions[0].offset,
            Offset(0, 0),
        )

        self.assertEqual(
            partitions[1].partition.shape,
            (225, 880, 3)
        )

        self.assertEqual(
            partitions[1].offset,
            Offset(0, 225),
        )

    def test_partition_frame_invalid_partition_number(self) -> None:
        with self.assertRaises(ValueError):
            EqualPartition(
                Config(
                    total_partition_num=-1
                )
            ).partition_frame(
                self.frame
            )


class TestRPAwarePartition(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = cv2.imread(TEST_IMAGE_PATH)

    def test_find_rps_periphery(self) -> None:
        rps = np.array(
            [
                [10, 10, 100, 100],
                [30, 30, 150, 150],
                [50, 50, 250, 450],
                [10, 10, 300, 200],
            ]
        )
        self.assertEqual(
            RPAwarePartition.find_rps_periphery(
                rps
            ),
            (
                10,
                10,
                300,
                450,
            )
        )

        rp = np.array(
            [
                [1, 100, 200, 300],
                [101, 2, 300, 400],
                [102, 201, 888, 401],
                [105, 203, 303, 999]
            ]
        )
        self.assertEqual(
            RPAwarePartition.find_rps_periphery(
                rp
            ),
            (
                1,
                2,
                888,
                999
            )
        )

    def test_create_rp_boxes(self) -> None:
        assert_array_equal(
            RPAwarePartition.create_rp_boxes(
                total_rp_box_num=1,
                rp_periphery=(0, 0, 100, 100)
            ),
            np.array(
                [
                    [0, 0, 100, 100],
                ]
            )
        )

        assert_array_equal(
            RPAwarePartition.create_rp_boxes(
                total_rp_box_num=2,
                rp_periphery=(0, 0, 100, 100)
            ),
            np.array(
                [
                    [0, 0, 50, 100],
                    [50, 0, 100, 100],
                ]
            )
        )

        assert_array_equal(
            RPAwarePartition.create_rp_boxes(
                total_rp_box_num=4,
                rp_periphery=(0, 0, 100, 100)
            ),
            np.array(
                [
                    [0, 0, 25, 100],
                    [25, 0, 50, 100],
                    [50, 0, 75, 100],
                    [75, 0, 100, 100],
                ]
            )
        )

    def test_create_rp_boxes_invalid_input(self) -> None:
        with self.assertRaises(ValueError):
            RPAwarePartition.create_rp_boxes(
                total_rp_box_num=0,
                rp_periphery=(10, 10, 20, 20),
            )

        with self.assertRaises(ValueError):
            RPAwarePartition.create_rp_boxes(
                total_rp_box_num=-1,
                rp_periphery=(10, 10, 20, 20),
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
            RPAwarePartition.find_max_overlaps(rps_0, rp_boxes_0),
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
            RPAwarePartition.find_max_overlaps(rps_1, rp_boxes_0),
            np.array([0, 0, 1])
        )

    def test_adjust_rp_boxes(self) -> None:
        rp_boxes_0 = np.array(
            [
                [20, 20, 373, 1080],
                [373, 20, 726, 1080],
                [726, 20, 1079, 1080],
            ]
        )

        rps_0 = np.array(
            [
                [50, 50, 300, 800],
                [323, 55, 383, 980],
                [50, 60, 900, 980]
            ]
        )

        index_max_rp_box_overlap = RPAwarePartition.find_max_overlaps(
            rps_0,
            rp_boxes_0
        )

        contained_rps = [
            rps_0[
                index_max_rp_box_overlap == i
            ] for i in range(
                len(rp_boxes_0)
            )
        ]

        assert_array_equal(
            RPAwarePartition.adjust_rp_boxes(rps_0, contained_rps),
            np.array(
                [
                    [50, 50, 383, 980],
                    [50, 60, 900, 980],
                    [0, 0, 3, 3],
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
            RPAwarePartition.expand_rp_boxes(
                rp_boxes,
                0.1,
                0.1,
                1080,
                750,
            ),
            np.array(
                [
                    [0, 0, 548, 1080],
                    [0, 0, 656, 1080],
                    [0, 0, 750, 1080],
                ]
            )
        )

    def test_frame_partition(self) -> None:
        np.random.seed(6)
        frame = np.random.rand(2560, 1080, 3)
        rp_aware_partition = RPAwarePartition(
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

        frame_pars = rp_aware_partition.partition_frame(
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