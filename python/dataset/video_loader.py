import os
import random
from typing import List

import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from config import Config


class VideoLoader(Dataset):
    """This class loads videos from a given root directory"""

    def __init__(self, config: Config):
        random.seed(6)
        np.random.seed(6)
        self.config: Config = config
        self.file_type: str = ".png"
        #self.file_type = ".jpg"

        self.video_dir_root = config.video_dir_root
        self._window_size: int = config.window_size
        self.white_list = self.gen_white_list()
        self.video_dirs = list()
        self.videos = self.find_video_frames()
        self.prefix = [0] * len(self.videos)
        self.cal_prefix()
        self.batch_size = 1

    def __len__(self):
        return self.prefix[-1]

    def __getitem__(self, index: int):
        # Locates its associated video
        video_index = self.find_video_index(index)

        # Locates the frame index at its associated video
        start_frame_index = self.find_start_frame_index(index, video_index)

        frame_path = self.videos[video_index][start_frame_index]

        return self.load_tensor(frame_path), frame_path

    def find_video_frames(self) -> List[List[str]]:
        """
        Given a root of video dir, the function iterate the directory,
        and find out all the frames under each video dir
        :return: the paths of video frames from each videos
        """
        video_dirs = os.listdir(self.video_dir_root)
        self.gen_white_list()
        videos = []
        for dir in sorted(video_dirs):
            if len(self.white_list) > 0 and dir not in self.white_list:
                continue
            dir = self.video_dir_root + dir
            if not os.path.isdir(dir):
                continue
            self.video_dirs.append(dir)
            frames = os.listdir(dir)
            tmp = []
            for v in sorted(frames, key=lambda x: int(x.split('.')[0])):
                tmp.append(dir + '/' + v)
            videos.append(tmp)
        return videos

    def gen_white_list(self) -> List[str]:
        """
        Build a white list to select video dirs
        :return: a video white list
        """
        path: str = self.video_dir_root + "whitelist.txt"
        if not os.path.isfile(path):
            return []
        white_list: List[str] = []
        with open(path, 'r') as f:
            for line in f.readlines():
                if '#' in line:
                    continue
                white_list.append(line.strip())
        return white_list

    def find_start_frame_index(self, index, video_index):
        """
        Given a random index, this function finds the frame index at its associated video
        :param index: global index
        :param video_index: associated video index
        :return: the frame index at its associated video
        """
        start_frame_index = index if video_index < 1 else index - self.prefix[video_index - 1]

        if (video_index == 0 and start_frame_index == self.prefix[video_index] - 1) or (
                video_index > 0 and start_frame_index == index - self.prefix[video_index - 1]):
            start_frame_index -= 1
        # handle corner case:
        # if cannot find sufficient preceding video frames, start from the first one
        if start_frame_index < self._window_size:
            start_frame_index = self._window_size
        return start_frame_index

    def find_video_index(self, index):
        """
        Given a random index, this function locates its associated video
        :param index:
        :return: video index
        """
        l, r = 0, len(self.videos)
        while l < r:
            m = l + (r - l) // 2
            if self.prefix[m] <= index:
                l = m + 1
            else:
                r = m
        return l

    def cal_prefix(self):
        self.prefix[0] = len(self.videos[0])

        for i in range(1, len(self.videos)):
            self.prefix[i] = len(self.videos[i]) + self.prefix[i - 1]

    def get_video_dirs(self):
        return self.video_dirs

    def get_frame_path(self, dir):
        index = self.video_dirs.index(dir)
        return self.videos[index]

    def get_data_loader(self, shuffle=True):
        return DataLoader(self, batch_size=self.batch_size, shuffle=shuffle)

    @staticmethod
    def load_tensor(filepath):
        return cv2.imread(filepath)

    def get_train_test_data_loader(self, test=False):
        dataset = self.get_data_loader()
        validation_split = .2
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        shuffle_dataset = True
        random_seed = 42
        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        if not test:
            return DataLoader(self, batch_size=self.config.batch_size,
                              sampler=train_sampler)
        return DataLoader(self, batch_size=1,
                          sampler=valid_sampler)
