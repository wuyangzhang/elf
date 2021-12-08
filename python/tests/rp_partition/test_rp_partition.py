import unittest

import numpy as np

from rp_partition.rp_partition import RPPartition, Offset


class TestRPPartition(unittest.TestCase):
    def test_rp_partition_create(self) -> None:
        rp_partition = RPPartition(
            partition=np.random.rand(100, 100, 3),
            offset=Offset(0, 1)
        )

        self.assertTrue(
            isinstance(rp_partition, RPPartition)
        )

    def test_rp_partition_size(self) -> None:
        rp_partition = RPPartition(
            partition=np.random.rand(100, 100, 3),
            offset=Offset(0, 1)
        )

        self.assertEqual(
            rp_partition.partition_size,
            100 * 100 * 3,
        )
