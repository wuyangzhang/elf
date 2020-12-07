"""
This class is an example integration with Detectron2
https://github.com/facebookresearch/detectron2
"""
import sys
from pathlib import Path

from client_model import ClientModelInterface

sys.path.append(str(Path.home()) + '/detectron2')
from demo import maskrcnn_interface
from demo import render

from typing import Any

import numpy as np
import torch


class ClientModelDetectron2(ClientModelInterface):
    def create_model(self):
        self.app = maskrcnn_interface()

    def run(self, img: np.ndarray) -> Any:
        self.predictions, _ = self.app.run_on_image(img)
        return self.predictions

    @staticmethod
    def extract_rps(inference_result: Any) -> torch.tensor:
        return inference_result['instances'].get_fields()['pred_boxes'].tensor.cpu()

    @staticmethod
    def extract_masks(inference_result: Any) -> torch.tensor:
        return inference_result['instances'].get_fields()['pred_masks']

    def render(self, img: np.ndarray, inference_result: Any) -> np.ndarray:
        return render(self.app, img, inference_result).get_image()
