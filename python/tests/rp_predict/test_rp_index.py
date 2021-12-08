import unittest

import numpy as np
from numpy.testing import assert_array_equal

from rp_predict.rp_index import (
    calculate_x_offset,
    calculate_y_offset,
    calculate_area_offset,
    find_minimal_offset,
    reorder_rps_in_last_frames,
    mask_large_offsets,
)


class TestRPIndex(unittest.TestCase):
    def setUp(self) -> None:
        self.rps = np.array(
            [
                [
                    [
                        [
                            100, 100, 150, 150,
                        ],
                        [
                            150, 150, 200, 200,
                        ],
                        [
                            40, 40, 90, 90,
                        ]
                    ],
                    [
                        [
                            160, 160, 220, 220,
                        ],
                        [
                            100, 100, 170, 170,
                        ],
                        [
                            500, 500, 600, 600,
                        ]
                    ]
                ]
            ]
        )

        self.last_frame = self.rps[:, -1, :, :]

        self.previous_frame = self.rps[:, -2, :, :]

    def test_calculate_x_offset(self) -> None:
        x_offset = calculate_x_offset(
            self.previous_frame,
            self.last_frame,
        )

        assert_array_equal(
            x_offset,
            np.array(
                [
                    [
                        [
                            [65, 15, 125.],
                        ],
                        [
                            [10, 40., 70.],
                        ],
                        [
                            [425., 375., 485.],
                        ],
                    ]
                ]
            )
        )

    def test_calculate_y_offset(self) -> None:
        y_offset = calculate_y_offset(
            self.previous_frame,
            self.last_frame,
        )

        assert_array_equal(
            y_offset,
            np.array(
                [
                    [
                        [
                            [65, 10., 425.],
                        ],
                        [
                            [15, 40., 375.],
                        ],
                        [
                            [125, 70., 485.],
                        ],
                    ]
                ]
            )
        )

    def test_calculate_area_offset(self) -> None:
        area_offset = calculate_area_offset(
            self.previous_frame,
            self.last_frame,
        )
        assert_array_equal(
            area_offset,
            np.array(
                [
                    [
                        [
                            [10, 20, 50],
                        ],
                        [
                            [10, 20, 50],
                        ],
                        [
                            [10, 20, 50],
                        ],
                    ]
                ]
            )
        )

    def test_add_offsets(self) -> None:
        offsets = calculate_x_offset(
            self.previous_frame,
            self.last_frame,
        ) + calculate_y_offset(
            self.previous_frame,
            self.last_frame,
        ) + calculate_area_offset(
            self.previous_frame,
            self.last_frame,
        )

        assert_array_equal(
            offsets,
            np.array(
                [
                    [
                        [
                            [140, 45, 600],
                        ],
                        [
                            [35, 100, 495],
                        ],
                        [
                            [560, 465, 1020],
                        ],
                    ]
                ]
            )
        )

    def test_find_minimal_offset(self) -> None:
        offsets = np.array(
            [
                [
                    [
                        [0.14, 0.04, 0.26],
                    ],
                    [
                        [0.04, 0.10, 0.16],
                    ],
                    [
                        [0.90, 0.80, 0.70],
                    ],
                ]
            ]
        )

        minimal_offset_index, minimal_offsets = find_minimal_offset(
            offsets
        )

        assert_array_equal(
            minimal_offset_index,
            np.array(
                [
                    [
                        [1],
                        [0],
                        [2],
                    ]
                ]
            )
        )

        assert_array_equal(
            minimal_offsets,
            np.array(
                [
                    [
                        [0.04],
                        [0.04],
                        [0.70],
                    ]
                ]
            )
        )

    def test_reorder_rps_in_last_frames(self) -> None:
        last_frame = np.array(
            [
                [
                    [160, 160, 220, 2201],
                    [100, 100, 170, 1701],
                    [500, 500, 600, 6001]
                ]
            ]
        )

        assert_array_equal(
            reorder_rps_in_last_frames(
                last_frame,
                np.array(
                    [
                        [
                            [1],
                            [0],
                            [2],
                        ]
                    ]
                )
            ),
            np.array(
                [
                    [
                        [100, 100, 170, 1701],
                        [160, 160, 220, 2201],
                        [500, 500, 600, 6001]
                    ]
                ]
            )
        )

        assert_array_equal(
            reorder_rps_in_last_frames(
                last_frame,
                np.array(
                    [
                        [
                            [2],
                            [0],
                            [1],
                        ]
                    ]
                )
            ),
            np.array(
                [
                    [
                        [500, 500, 600, 6001],
                        [160, 160, 220, 2201],
                        [100, 100, 170, 1701],
                    ]
                ]
            )
        )

    def test_mask_large_offsets(self) -> None:
        offsets = np.array(
            [
                [
                    [0.5, 0.4, 0.4, 0.3],
                    [0.2, 0.1, 0.7, 0.0],
                ]
            ]
        )

        assert_array_equal(
            mask_large_offsets(
                offsets,
                0.05,
            ),
            np.array(
                [
                    [
                        True, False, False, False
                    ]
                ]
            )
        )

    def test_filter_rp_with_large_offsets(self) -> None:
        last_frame = np.array(
            [
                [
                    [500, 500, 600, 6001],
                    [160, 160, 220, 2201],
                    [100, 100, 170, 1701],
                ]
            ]
        )

        mask = np.array(
            [
                [
                    True, False, False,
                ]
            ]
        )

        last_frame[mask, ] = 0

        assert_array_equal(
            last_frame,
            np.array(
                [
                    [
                        [0, 0, 0, 0],
                        [160, 160, 220, 2201],
                        [100, 100, 170, 1701],
                    ]
                ]
            )
        )

    def test_post_rp_index_process(self) -> None:
        last_frame = np.array(
            [
                [
                    [0, 0, 0, 0],
                    [160, 160, 220, 2201],
                    [100, 100, 170, 1701],
                ]
            ]
        )

        output_rps = self.rps.copy()

        output_rps1 = 0
