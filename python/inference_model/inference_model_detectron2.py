"""
This class is an example integration with Detectron2:
https://github.com/facebookresearch/detectron2.
"""
import sys
from pathlib import Path
from typing import List, Any

import numpy as np
import torch

from inference_model import InferenceModelInterface

sys.path.append(str(Path.home()) + '/detectron2')
"""This is the path to install the source code of detectron2."""

from demo import maskrcnn_interface, render
"""Please follow the instruction in REAMDE to add the API maskrcnn_interface under detectron/demo/demo.py."""


class InferenceModelDetectron2(InferenceModelInterface):
    def create_model(self):
        self.app = maskrcnn_interface()

    def run(self, img: np.ndarray) -> Any:
        self.predictions, _ = self.app.run_on_image(img)
        return self.predictions

    @staticmethod
    def extract_rps(inference_result: Any) -> np.ndarray:
        return inference_result['instances'].get_fields()['pred_boxes'].tensor.numpy()

    @staticmethod
    def extract_masks(inference_result: Any) -> torch.tensor:
        return inference_result['instances'].get_fields()['pred_masks']

    def render(self, img: np.ndarray, inference_result: Any) -> np.ndarray:
        return render(img, inference_result).get_image()

    def merge(
        self,
        inference_results: List[Any],
        offsets: List[List[int]],
        **kwargs
    ) -> Any:

        frame_height = kwargs.get("frame_height")
        frame_width = kwargs.get('frame_width')
        merge_mask = kwargs.get('merge_mask')

        for index, inference_result in enumerate(inference_results):
            if len(inference_result["instances"]) == 0:
                continue

            w, h = offsets[index]

            # Offset boxes.
            inference_result["instances"].get_fields()["pred_boxes"].tensor[:, 0] += w
            inference_result["instances"].get_fields()["pred_boxes"].tensor[:, 1] += h
            inference_result["instances"].get_fields()["pred_boxes"].tensor[:, 2] += w
            inference_result["instances"].get_fields()["pred_boxes"].tensor[:, 3] += h

            # Offset mask.
            if merge_mask:
                shape = inference_result["instances"].image_size
                pad = torch.nn.ConstantPad2d((w, frame_width - w - shape[1],
                                              h, frame_height - h - shape[0]),
                                             0)
                inference_result["instances"].get_fields()["pred_masks"] = pad(
                    inference_result["instances"].get_fields()["pred_masks"][:, :, ])

        index = 0
        while index < len(inference_results) and len(inference_results[index]["instances"]) == 0:
            index += 1

        # Return if the current frame contains zero object of interest.
        if index == len(inference_results):
            return inference_results[0]

        ans = inference_results[index]

        ans["instances"]._image_size = (frame_height, frame_width)
        for i in range(index+1, len(inference_results)):
            if len(inference_results[i]["instances"]) == 0:
                continue

            ans["instances"].get_fields()["pred_boxes"].tensor = torch.cat(
                [ans["instances"].get_fields()["pred_boxes"].tensor,
                 inference_results[i]["instances"].get_fields()["pred_boxes"].tensor], dim=0)

            ans["instances"].get_fields()["scores"] = torch.cat(
                [ans["instances"].get_fields()["scores"],
                 inference_results[i]["instances"].get_fields()["scores"]], dim=0)

            ans["instances"].get_fields()["pred_classes"] = torch.cat(
                [ans["instances"].get_fields()["pred_classes"],
                 inference_results[i]["instances"].get_fields()["pred_classes"]], dim=0)

            if merge_mask:
                ans["instances"].get_fields()["pred_masks"] = torch.cat(
                    [
                        ans["instances"].get_fields()["pred_masks"],
                        inference_results[i]["instances"].get_fields()["pred_masks"],
                    ],
                    dim=0
                )

        return ans