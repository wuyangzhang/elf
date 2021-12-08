from typing import Any, List

import numpy as np
import torch

from inference_model import InferenceModelInterface


class InferenceModelTest(InferenceModelInterface):
    def create_model(self) -> None:
        return None

    def run(self, img: np.ndarray) -> Any:
        return np.random.rand(4, 50, 50) * 500

    @staticmethod
    def extract_rps(inference_result: Any) -> torch.tensor:
        return torch.tensor(
            np.random.rand(4, 50, 50) * 500
        )

    @staticmethod
    def extract_masks(inference_result: Any) -> torch.tensor:
        pass

    def render(self, img: np.ndarray, inference_result: Any) -> np.ndarray:
        pass

    def merge(self, inference_results: List[Any], offsets: List[int], **kwargs) -> Any:
        pass