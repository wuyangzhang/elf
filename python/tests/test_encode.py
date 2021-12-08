import unittest

import numpy as np
from numpy.testing import assert_array_equal

from networking import EncoderPickle, EncoderNVJPEG


class TestEncode(unittest.TestCase):
    def test_pickle_encoder(self):
        encoder = EncoderPickle()
        img = np.random.rand(1024, 768, 3)
        encoded_img = encoder.encode(img)
        decoded_img = encoder.decode(encoded_img)

        assert_array_equal(img, decoded_img)

    @unittest.skip("NVJPEG package not available")
    def test_nvjpeg_encoder(self):
        encoder = EncoderNVJPEG()
        img = np.random.rand(1024, 768, 3)
        encoded_img = encoder.encode(img)
        decoded_img = encoder.decode(encoded_img)

        assert_array_equal(img, decoded_img)

