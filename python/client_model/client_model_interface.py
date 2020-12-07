import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch

from config import Config


class ClientModelInterface(ABC):
    """
    This class wrap high level applications to interact with Elf.
    For each high level application to run with Elf, it needs to provide the below general interfaces:
    """

    def __init__(self, config: Config):
        self.config = config
        self.create_model()

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def run(self, img: np.ndarray) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def extract_rps(inference_result: Any) -> torch.tensor:
        pass

    @abstractmethod
    def render(self, img: np.ndarray, inference_result: Any) -> np.ndarray:
        pass









