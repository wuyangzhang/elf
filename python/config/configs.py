import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

sys.path.append(str(Path.home()) + '/elf/')


@dataclass
class Config:
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(Config, cls).__new__(cls)
        return cls.instance

    """ Model selection"""
    client_models: str = ('maskrcnn')
    client_model_index: int = 1

    """ Offloading modules"""
    servers: Tuple[Tuple[str, int]] = (
        # Local test by setting three servers with the same IP
        (('localhost', 5053), ('localhost', 5053), ('localhost', 5053)),

        # Server deployment by setting any number of servers
        (('node21-20', 5051), ('node21-21', 5051), ('node21-22', 5051)),
    )

    server: str = servers[0]

    # Whether use CUDA for model inference
    use_cuda: bool = False

    """ Video Frame details"""
    frame_height: int = 375
    frame_width: int = 1242

    """ Dataset specification"""
    home_addr: str = str(Path.home())

    datasets: Tuple[str] = ('kitti', 'davis')
    eval_dataset: str = 'davis'
    # Kitti, h: 375, w: 1242
    kitti_video_path: str = home_addr + '/kitti/testing/seq_list.txt'
    davis_video_path: str = home_addr + '/datasets/davis/DAVIS/JPEGImages/480p/'

    video_dir_roots: Tuple[str] = (
        home_addr + '/datasets/kitti/mots/training/image_02/',
        home_addr + '/pose/posetrack_data/images/bonn/',
        home_addr + '/datasets/kitti/mots/wz/train/',
    )

    video_dir_root: str = video_dir_roots[0]

    # Dynamic scale ratios of video frames
    scale_ratio: int = 2
    scale_ratio_x: float = 2560 / 1242
    scale_ratio_y: float = 1920 / 375

    """ Prediction modules"""
    # Prediction model
    prediction_models: Tuple[str] = ('lstm', 'attn')
    model_index: int = 1
    prediction_model: str = prediction_models[model_index]

    # Model loading path
    prediction_model_paths: Tuple[str] = (
        './prediction/model/outputs/lstm_single_checkpoint40.pth',
        './prediction/model/outputs/attn_lstm_checkpoint30.pth',
    )

    prediction_model_path: str = prediction_model_paths[model_index]

    # The number of historical video frames for the prediction
    window_size: int = 3

    # The max value of detected objects within a single video frame
    padding_len: int = 160
    batch_size: int = 16

    # LRC rescale ratio
    lrc_ratio: float = 0.4

    # RP expansion ratio after making predictions
    rp_rescale_ratio: float = 0.06

    """ Partition modules"""
    # RP box rescale ratio
    rescale_ratio: float = 0.05  # 0.05

    # Use the last sever in the server list for running LRC
    total_remote_servers: int = len(server) - 1

    # Disable mask merge to lower Elf overheads
    merge_mask: bool = True
