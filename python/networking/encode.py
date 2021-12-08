from abc import ABC, abstractmethod

import numpy as np
from pickle import dumps, loads


class EncoderBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def encode(self, frame: np.ndarray):
        pass

    @abstractmethod
    def decode(self, byte_arr) -> np.ndarray:
        pass


class EncoderNVJPEG(EncoderBase):
    def __init__(self):
        super().__init__()
        import py_nvjpeg
        self.encoder = py_nvjpeg.NvJPEG(1920, 2560)
        self.encoder.init()

    def encode(self, frame: np.ndarray):
        return self.encoder.encode(frame)

    def decode(self, byte_arr) -> np.ndarray:
        return self.encoder.decode(byte_arr)


class EncoderPickle(EncoderBase):
    def encode(self, frame: np.ndarray):
        return dumps(frame)

    def decode(self, byte_arr) -> np.ndarray:
        return loads(byte_arr)
