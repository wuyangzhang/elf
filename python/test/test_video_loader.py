import unittest

from dataset import VideoLoader
from config import Config


class TestVideoLoader(unittest.TestCase):

    def test_run(self) -> None:
        dataset = VideoLoader(Config())
        for img, path in dataset:
            print(img.shape)
