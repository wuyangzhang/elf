import unittest

import cv2

from util.helper import (
    crop_frame,
    get_frame_height_width,
    get_frame_width,
    get_frame_height,
    rescale_frame,
)

TEST_IMAGE_PATH = '/Users/wuyang/python/elf/python/tests/test_images/test_image_0.jpeg'


class TestHelper(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = cv2.imread(TEST_IMAGE_PATH)
        """Shape: height: 450, width: 880."""

    def test_crop_image(self) -> None:
        output = crop_frame(
            self.frame,
            0,
            0,
            0,
            0
        )
        self.assertEqual(
            output.shape,
            (0, 0, 3),
        )

        output = crop_frame(
            self.frame,
            0,
            0,
            400,
            400,
        )
        self.assertEqual(
            output.shape,
            (400, 400, 3),
        )

        output = crop_frame(
            self.frame,
            10,
            10,
            870,
            440,
        )

        self.assertEqual(
            output.shape,
            (self.frame.shape[0] - 10 - 10, self.frame.shape[1] - 10 - 10, 3),
        )

    def test_get_frame_height(self) -> None:
        self.assertEqual(
            get_frame_height(self.frame),
            self.frame.shape[0]
        )

    def test_get_frame_width(self) -> None:
        self.assertEqual(
            get_frame_width(self.frame),
            self.frame.shape[1]
        )

    def test_get_frame_height_width(self) -> None:
        self.assertEqual(
            get_frame_height_width(self.frame),
            (self.frame.shape[0], self.frame.shape[1]),
        )