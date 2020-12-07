"""
This class is an example integration with MMDetection
https://github.com/open-mmlab/mmdetection
"""
import sys
from pathlib import Path

from client_model import ClientModelInterface
from config import Config

sys.path.append(str(Path.home()) + '/mmdetection')
from tools import mmdetection_interface, run, detector_postprocess


class ClientModelMMDetection(ClientModelInterface):
    def __init__(self):
        super().__init__(Config())

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
