"""
This class is an example integration with MMDetection:
https://github.com/open-mmlab/mmdetection.
"""
import sys
from pathlib import Path

from inference_model import InferenceModelInterface

sys.path.append(str(Path.home()) + '/mmdetection')
"""This is the path to install the source code of mmdetection."""

from tools.test import mmdetection_interface, run, detector_postprocess
"""Please follow the instruction in REAMDE to add the API 
mmdetection_interface,run,detector_postprocess under mmdetection/tools/test.py.
"""


class InferenceModelMMDetection(InferenceModelInterface):
    def __init__(self):
        super().__init__()

        self.predictions = {}
        self.model = None
        self.create_model()

    def create_model(self):
        self.model = mmdetection_interface()

    def run(self, img):
        predictions = run(self.model, img)
        return predictions

    def extract_rps(self, predictions):
        pass

    def render(self, img, predictions):
        pass

    @staticmethod
    def post_processing(predictions, output_height, output_width):
        return detector_postprocess(predictions, output_height, output_width)
