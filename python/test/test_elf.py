import unittest

import numpy as np

from config import Config
from control import ElfControlManager
from dataset import VideoLoader


class TestElfControlManager(unittest.TestCase):

    @staticmethod
    def return_example_inference_result(soc, ans: list):
        ans[0] = 0

    @staticmethod
    def example_send(soc, img: np.ndarray):
        return

    def test_run(self) -> None:
        elf_control_mgr = ElfControlManager(Config())
        elf_control_mgr._receive_inference_results = self.return_example_inference_result
        elf_control_mgr._send_frame = self.example_send

        img = np.random.rand(2560, 1280, 3)
        for _ in range(10):
            elf_control_mgr.run(img)

        img = np.random.rand(375, 1280, 3)
        for _ in range(10):
            elf_control_mgr.run(img)

    def test_run_video_dataset(self) -> None:
        elf_control_mgr = ElfControlManager(Config())
        elf_control_mgr._receive_inference_results = self.return_example_inference_result
        elf_control_mgr._send_frame = self.example_send

        video_dataset = VideoLoader(config=Config())
        index = 0
        for img, _ in video_dataset:
            elf_control_mgr.run(img)
            index += 1
            if index == 30:
                break
