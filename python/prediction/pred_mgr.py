import torch

from config import Config
from prediction import PredictionAlgorithmBase, PredictionAlgorithmAttentionLSTM
from prediction.util import box_iou


class PredictionManager:
    def __init__(self, config: Config):
        self._config: Config = config
        self._predict_algo: PredictionAlgorithmBase = PredictionAlgorithmAttentionLSTM(config)
        self._historical_rps = list()
        self._max_queue_size: int = config.window_size - 1
        self._next_predict = None

    def predict_rps(self):
        """
        To predict region proposals in the current frame.
        :return: predicted RPs
        """
        return self._predict_algo.predict(self._historical_rps)

    def add_rps(self, rps: torch.Tensor) -> None:
        """ Record new detected rps to the historical rp queue"""
        # Skip if no object has been found.
        if rps.size()[0] == 0:
            return
        # Only keep the last max_queue_size results.
        if len(self._historical_rps) == self._max_queue_size:
            self._historical_rps.pop(0)
        self._historical_rps.append(rps)

    def update_rps(self, rps):
        # If those new detected bbox is already in the queue.. then just ignore them.
        ious = box_iou(rps, self._historical_rps[-1])
        # if cannot find any iou among <1, M> >= 0.5, mark it as True (a new one not appeared in the queue)..
        mask = torch.sum(ious >= 0.5, dim=1) == 0
        self._historical_rps[-1] = torch.cat([rps[mask], self._historical_rps[-1]])

    def get_rps_queue_len(self) -> int:
        """ Get the length of the historical rp queue"""
        return len(self._historical_rps)

    def is_active(self):
        """ Decide whether the prediction is ready to proceed"""
        return len(self._historical_rps) >= self._max_queue_size
