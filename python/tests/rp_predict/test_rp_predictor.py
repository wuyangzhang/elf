import unittest

import numpy as np
from numpy.testing import assert_array_equal

from rp_predict.rp_predictor import RPPredictor
from tests.data_provider import data_provider


test_data_find_duplicated_rps=lambda: (
        (
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
            ),
            np.array(
                [
                    False,
                    True,
                ]
            )
        ),
        (
            np.array(
                [
                    [
                        1.,
                        0.146,
                        0,
                    ],
                ]
            ),
            np.array(
                [
                    False,
                ]
            ),
        )
    )


class TestRPPredictor(unittest.TestCase):
    def test_len(self) -> None:
        rp_predictor = RPPredictor()

        self.assertEqual(
            len(rp_predictor),
            0,
        )

        rp_predictor._historical_rps = [
            np.array(
                [0.]
            )
        ]

        self.assertEqual(
            len(rp_predictor),
            1,
        )

    def test_add_new_rps(self) -> None:
        rp_predictor = RPPredictor()
        rp_predictor._max_queue_size = 1

        rp_predictor.add_new_rps(
            np.array(
                [

                ]
            )
        )

        self.assertEqual(
            len(rp_predictor),
            0,
        )

        rp_predictor.add_new_rps(
            np.array(
                [
                    1.0,
                ]
            )
        )

        self.assertListEqual(
            rp_predictor._historical_rps,
            [
                np.array(
                    [
                        1.0,
                    ]
                )
            ]
        )

        rp_predictor.add_new_rps(
            np.array(
                [
                    2.0
                ]
            )
        )

        self.assertListEqual(
            rp_predictor._historical_rps,
            [
                np.array(
                    [
                        2.0,
                    ]
                )
            ]
        )

    def test_is_active(self) -> None:
        rp_predictor = RPPredictor()
        rp_predictor._max_queue_size = 1

        self.assertFalse(
            rp_predictor.is_active()
        )

        rp_predictor._historical_rps = [
            np.array([0])
        ]

        self.assertTrue(
            rp_predictor.is_active()
        )

    @data_provider(test_data_find_duplicated_rps)
    def test_find_duplicated_rps(self, iou, mask) -> None:
        assert_array_equal(
            RPPredictor._find_duplicated_rps(
                iou,
            ),
            mask,
        )

    def test_concat_new_lrc_rps(self) -> None:
        rp_predictor = RPPredictor()
        rp_predictor._historical_rps = [
            np.array(
                [
                    [
                        0, 0, 50, 50,
                    ],
                    [
                        0, 0, 100, 100,
                    ]
                ]
            )
        ]

        rp_predictor._concat_new_lrc_rps(
            rps=np.array(
                [
                    [
                        0, 0, 100, 100,
                    ],
                    [
                        0, 0, 150, 150,
                    ]
                ]
            ),
            mask=np.array(
                [
                    True,
                    False,
                ]
            )
        )

        assert_array_equal(
            rp_predictor._historical_rps[-1],
            np.array(
                [
                    [
                        0, 0, 50, 50,
                    ],
                    [
                        0, 0, 100, 100,
                    ],
                    [
                        0, 0, 150, 150,
                    ]
                ]
            )
        )



