import unittest

import numpy as np

from client_model import ClientModelDetectron2, ClientModelMMDetection
from config import Config


class TestApplication(unittest.TestCase):

    def test_application_detectron2(self) -> None:
        config = Config()
        detectron2 = ClientModelDetectron2(config)
        img = np.random.rand(2560, 1280, 3)
        detectron2.run(img)