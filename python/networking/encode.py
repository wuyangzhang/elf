from abc import ABC, abstractmethod

import numpy as np


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
    from pickle import dumps, loads

    def __init__(self):
        super().__init__()

    def encode(self, frame: np.ndarray):
        return self.dumps(frame)

    def decode(self, byte_arr) -> np.ndarray:
        return self.loads(byte_arr)


if __name__ == '__main__':
    x = np.random.rand(1024, 3, 3)
    a = 1
