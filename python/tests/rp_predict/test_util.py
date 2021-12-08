import unittest

import numpy as np
from numpy.testing import assert_array_equal

from rp_predict.util import calculate_rps_iou
from tests.data_provider import data_provider


test_iou_data = lambda: (
        (
            np.array(
                [
                    [
                        0, 0, 100, 100,
                    ],
                ]
            ),
            np.array(
                [
                    [
                        0, 0, 100, 100,
                    ],
                ]
            ),
            np.array(
                [
                    [
                        1.
                    ]
                ]
            ),
        ),
        (
            np.array(
                [
                    [
                        0, 0, 100, 100,
                    ],
                ]
            ),
            np.array(
                [
                    [
                    150, 150, 200, 200,
                    ],
                ]
            ),
            np.array(
                [
                    [
                        0.
                    ]
                ]
            ),
        ),
        (
            np.array(
                [
                    [
                        0, 0, 100, 100,
                    ],
                ]
            ),
            np.array(
                [
                    [
                        50, 50, 150, 150,
                    ],
                ]
            ),
            np.array(
                [
                    [
                        0.146,
                    ]
                ]
            ),
        ),
        (
            np.array(
                [
                    [
                        0, 0, 100, 100,
                    ]
                ]
            ),
            np.array(
                [
                    [
                        0, 0, 100, 100,
                    ],
                    [
                        50, 50, 150, 150,
                    ],
                    [
                        150, 150, 200, 200,
                    ]
                ]
            ),
            np.array(
                [
                    [
                        1.,
                        0.146,
                        0,
                    ]
                ]
            )
        ),
        (
            np.array(
                [
                    [
                        0, 0, 100, 100,
                    ],
                    [
                        50, 50, 100, 100,
                    ]
                ]
            ),
            np.array(
                [
                    [
                        0, 0, 100, 100,
                    ],
                    [
                        50, 50, 150, 150,
                    ],
                    [
                        150, 150, 200, 200,
                    ]
                ]
            ),
            np.array(
                [
                    [
                        1.,
                        0.146,
                        0,
                    ],
                    [
                        0.255,
                        0.255,
                        0,
                    ]
                ]
            )
        )
    )


class TestUtil(unittest.TestCase):
    @data_provider(test_iou_data)
    def test_rps_iou(self, rps_0: np.ndarray, rps_1: np.ndarray, iou: np.ndarray) -> None:
        assert_array_equal(
            calculate_rps_iou(
                rps_0,
                rps_1,
            ),
            iou,
        )