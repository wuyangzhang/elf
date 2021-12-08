from typing import Optional, List

import numpy as np

from config import Config
from rp_predict import (
    RPPredictionAlgorithmBase,
    RPPredictionAlgorithmFastTracker,
    RPPredictionAlgorithmAttentionLSTM,
)
from rp_predict.util import calculate_rps_iou


class RPPredictor:
    def __init__(
        self,
        config: Config = Config(),
        predict_algorithm: RPPredictionAlgorithmBase = RPPredictionAlgorithmAttentionLSTM(
            config=Config()
        )
    ):
        self._config: Config = config

        self._predict_algorithm: RPPredictionAlgorithmBase = predict_algorithm

        # The queue to store all the historical RPs returned by the model inference.
        # Each element contains all RPs from a previous frame.
        self._historical_rps: List[np.ndarray] = list()

        # The historical RP queue will keep at most #max_queue_size elements.
        self._max_queue_size: int = config.rp_predict_window_size - 1

        # The RP prediction result for the current frame before model inference.
        self._latest_predicted_rps: Optional[np.ndarray]

    def __len__(self) -> int:
        """ Get the length of the historical rp queue"""
        return len(self._historical_rps)

    def predict_rps(self):
        """
        To predict region proposals in the current frame by taking RPs in historical frames.
        """
        return self._predict_algorithm.predict(
            self._historical_rps
        )

    def add_new_rps(self, rps: np.ndarray) -> None:
        """Add new detected rps to the historical rp queue."""
        if len(rps) == 0:
            return None

        # Only keep the last max_queue_size results.
        if len(self._historical_rps) == self._max_queue_size:
            self._historical_rps.pop(0)

        self._historical_rps.append(rps)

    def update_new_rps_from_lrc(
        self,
        rps: np.ndarray,
    ) -> None:
        """
        LRC intends to find new RPs that have not been detected before.
        However, we need to remove the duplicated ones that are already existed.
        :param rps:
        :param is_new_rp_detected: if there is any new RP has been detected through non-LRC flow from the current frame.
        :return:
        """
        # If those new detected bbox is already in the queue.. then just ignore them.
        iou = calculate_rps_iou(
            rps,
            self._historical_rps[-1]
        )

        # If cannot find any iou among <1, M> >= 0.5, mark it as True (a new one not appeared in the queue).
        mask = self._find_duplicated_rps(iou)

        self._concat_new_lrc_rps(
            rps,
            mask,
        )

    @staticmethod
    def _find_duplicated_rps(iou: np.ndarray) -> np.ndarray:
        """
        Iou is in the shape (n, m) between LRC RPs and latest RPs from a normal Elf flow.
        :return: a boolean 1d array in the shape (n)
        """
        return np.sum(iou >= 0.5, axis=1) == 0

    def _concat_new_lrc_rps(
        self,
        rps: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        self._historical_rps[-1] = np.concatenate(
            (
                self._historical_rps[-1],
                rps[~mask]
            ),
            axis=0
        )

    def is_active(self) -> bool:
        """Decide whether the rp_predict is ready to proceed."""
        return len(self._historical_rps) >= self._max_queue_size
