from abc import ABC, abstractmethod
from typing import Union, List

from prediction.helper import *
from prediction.model.model_helper import create_model
from prediction.util import load_tensors, scale_tensor


class PredictionAlgorithmBase(ABC):
    def __init__(self, config: Config):
        self.config = config

    def predict(self, predict_input: Union[List[np.ndarray], np.ndarray]) -> np.ndarray:
        """Predict RPs in the current frame based on historical RPs"""
        predict_input = self.prepare_input(predict_input)
        predict_output = self.run(predict_input)
        post_output = self.post_proc(predict_output)
        return self.tensor_to_numpy(post_output)

    @abstractmethod
    def prepare_input(self, predict_input: Union[List[np.ndarray], np.ndarray]) -> np.ndarray:
        """ Padding and normalize input"""
        pass

    @abstractmethod
    def post_proc(self, predict_output: np.ndarray) -> np.ndarray:
        """Convert the normalized prediction results to the true image size."""
        pass

    @abstractmethod
    def run(self, predict_input: np.ndarray) -> np.ndarray:
        pass

    def extend_rp(self, output: np.ndarray, extend_ratio: float) -> np.ndarray:
        """Further region proposal extension..
        :param output:
        :param extend_ratio:
        :return:
        """
        rp_width = (output[:, 2] - output[:, 0]) * extend_ratio
        rp_height = (output[:, 3] - output[:, 1]) * extend_ratio
        output[:, 0] = np.maximum(0, output[:, 0] - rp_width)
        output[:, 1] = np.maximum(0, output[:, 1] - rp_height)
        output[:, 2] = np.minimum(self.config.frame_width, output[:, 2] + rp_width)
        output[:, 3] = np.minimum(self.config.frame_height, output[:, 3] + rp_height)
        return output

    @staticmethod
    def tensor_to_numpy(predict_output: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert the normalized prediction results to the true image size."""
        if type(predict_output) == torch.Tensor:
            return predict_output.numpy()
        return predict_output


class PredictionAlgorithmAttentionLSTM(PredictionAlgorithmBase):
    def __init__(self, config: Config):
        super().__init__(config)
        self.model = create_model(config, load_state=True)

    def prepare_input(self, predict_input: Union[List[np.ndarray], np.ndarray]) -> np.ndarray:
        max_len = max(len(x) for x in predict_input)
        predict_input = np.stack(
            [np.concatenate([d.numpy(), np.zeros([max_len - d.shape[0], 4])]) for d in predict_input])

        predict_input[:, :, 0] = predict_input[:, :, 0] / self.config.frame_width
        predict_input[:, :, 1] = predict_input[:, :, 1] / self.config.frame_height
        predict_input[:, :, 2] = predict_input[:, :, 2] / self.config.frame_width
        predict_input[:, :, 3] = predict_input[:, :, 3] / self.config.frame_height

        predict_input = rp_index_np(predict_input)
        predict_input = remove_zero_bbox(predict_input)

        return torch.from_numpy(predict_input).float()

    def post_proc(self, output: np.ndarray) -> np.ndarray:
        """Convert the normalized prediction results to the true image size."""
        output[:, 0] = output[:, 0] * self.config.frame_width
        output[:, 1] = output[:, 1] * self.config.frame_height
        output[:, 2] = output[:, 2] * self.config.frame_width
        output[:, 3] = output[:, 3] * self.config.frame_height
        output = self.extend_rp(output, self.config.rescale_ratio)
        return output

    @torch.no_grad()
    def run(self, predict_input: np.ndarray):
        return self.model(predict_input).cpu()


class PredictionAlgorithmFastTracker(PredictionAlgorithmBase):
    """Zoom in the RPs of the last frame"""

    def __init__(self, config: Config, extend_ratio=0.5):
        super().__init__(config)
        self.extend_ratio = extend_ratio

    def prepare_input(self, predict_input: Union[List[np.ndarray], np.ndarray]) -> np.ndarray:
        return predict_input

    def run(self, predict_input: np.ndarray) -> np.ndarray:
        return self.extend_rp(predict_input[-1], self.extend_ratio)

    def post_proc(self, predict_output: np.ndarray) -> np.ndarray:
        return predict_output


class PredictionAlgorithmUseGT(PredictionAlgorithmBase):
    """Use the RP ground truth as the prediction results"""

    def __init__(self, config: Config, extend_ratio=0.5):
        super().__init__(config)
        self.gt_tensors = load_tensors("")
        scale_tensor(self.gt_tensors, config.scale_ratio_x, config.scale_ratio_y)

    def prepare_input(self, predict_input: Union[List[np.ndarray], np.ndarray]) -> np.ndarray:
        return predict_input

    def run(self, predict_input: np.ndarray) -> np.ndarray:
        if len(self.gt_tensors) == 0:
            raise ValueError("Empty tensors")
        return self.gt_tensors.pop(0)

    def post_proc(self, predict_output: np.ndarray) -> np.ndarray:
        return predict_output
