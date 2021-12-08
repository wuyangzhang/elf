import unittest

from flow.offload_schedule import *


class TestOffloadSchedule(unittest.TestCase):
    def test_prioritize_larger_rp(self) -> None:
        ordered_rp_boxes = prioritize_larger_rp(
            [
                np.random.rand(100, 100, 3),
                np.random.rand(50, 50, 3),
                np.random.rand(200, 200, 3),
            ]
        )

        self.assertEqual(
            ordered_rp_boxes[2].size,
            50 * 50 * 3,
        )

        self.assertEqual(
            ordered_rp_boxes[1].size,
            100 * 100 * 3,
        )

        self.assertEqual(
            ordered_rp_boxes[0].size,
            200 * 200 * 3,
        )

    def test_match_rp_boxes_servers(self) -> None:
        ordered_rp_boxes = schedule_offloading(
            [
                np.random.rand(100, 100, 3),
                np.random.rand(50, 50, 3),
                np.random.rand(200, 200, 3),
            ]
        )

        self.assertEqual(
            ordered_rp_boxes[2].size,
            50 * 50 * 3,
        )

        self.assertEqual(
            ordered_rp_boxes[1].size,
            100 * 100 * 3,
        )

        self.assertEqual(
            ordered_rp_boxes[0].size,
            200 * 200 * 3,
        )