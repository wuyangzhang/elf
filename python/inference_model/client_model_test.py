from typing import Any

import numpy as np
import torch

from inference_model import InferenceModelInterface


class InferenceModelTest(InferenceModelInterface):
    def create_model(self):
        pass

    def run(self, img: np.ndarray) -> Any:
        pass

    def extract_rps(self, inference_result: Any) -> torch.tensor:
        rps = torch.rand(32, 4)
        rps[:, 0] *= self.config.frame_height
        rps[:, 1] *= self.config.frame_width
        rps[:, 2] *= self.config.frame_height
        rps[:, 3] *= self.config.frame_width
        return rps

    def render(self, img: np.ndarray, inference_result: Any) -> np.ndarray:
        pass
