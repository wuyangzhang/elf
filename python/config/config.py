from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass(frozen=False)
class Config:
    servers: List[Tuple[str, int]] = (
            ('localhost', 5053),
            ('localhost', 5053),
            ('localhost', 5053),
        )
    """Distributed servers to run model inference."""

    server_index = 0

    lrc_server: Tuple[str, int] = ('localhost', 5053)
    """Server to run LRC service."""

    use_gpu: bool = True
    """GPU is used for model inference if the flag is True. Otherwise, it uses CPU."""

    frame_height: int = 370
    frame_width: int = 1224
    """Video Frame resolution configuration."""

    video_dataset_dir: str = str(Path.home()) + '/datasets/kitti/mots/training/image_02/'
    """Directory of the video/image datasets."""

    total_partition_num: int = len(servers)
    """Total number that a frame will be partitioned."""

    rp_extend_ratio: float = 0.06
    """
    RP will be extended after model inference to error compensation.
    See more details in Section 4.3.
    """

    min_rp_rescale_ratio: float = 0.1
    """
    In case the RP is rescaled to an extremely small size and an inference engine cannot process it, this value controls
    a minimal rescale ratio.
    """

    lrc_downsample_ratio: float = 0.4
    """
    LRC (Low resolution compensation) down-samples a raw video frame.
    A dedicated server will run the same model inference on the frame 
    to get the rough estimation of new objects. 
    See more details in Section 4.4.
    """

    lrc_window: int = 3
    """Receive every lrc_window frames, it run a LRC service."""

    RP_PREDICTION_MODELS: Tuple[str] = ('lstm', 'attn_lstm')
    """All available models for RP rp_predict."""

    rp_prediction_model: str = 'attn_lstm'
    """Selected model for RP rp_predict."""

    rp_prediction_model_path: str = './rp_predict/model/outputs/attn_lstm_checkpoint25.pth'
    """Path to load RP rp_predict model."""

    rp_prediction_training_dataset_path: str = './rp_predict/data/train/'
    """Path to load the training dataset of RP prediction."""

    rp_prediction_eval_dataset_path: str = './rp_predict/data/eval/'
    """Path to load the eval dataset of RP prediction."""

    rp_predict_window_size: int = 3
    """The number of historical video frames used for RP rp_predict."""

    frame_scale_ratio: int = 2
    """Dynamic scaling ratios of video frames"""

    visualization_mode: bool = True

    """If the flag is set True, it shows the received frame and frame partitions."""

    padding_len: int = 128
    """The max value of detected objects within a single video frame."""

    batch_size: int = 16

    merge_mask: bool = True
    """Disable mask merge to lower Elf overheads."""
