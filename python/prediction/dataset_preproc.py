"""
We prepare the dataset by converting
video frames into the bbox formats: [x1, y1, x2, y2, complexity].
Each frame contains a set of the bbox coordinates and computing
complexity.

We run MaskRCNN over video datasets to extract those information.
We consider the following video datasets:
UCF-101 Datasets
Human3.6M Datasets
CityScape Datasets
KTH Datasets
Robotic Pushing Datasets
"""

import random

import numpy as np
import torch
import torch.nn
import torch.nn.utils.rnn
from torch.utils.data import Dataset, DataLoader


class RPPNDataset(Dataset):

    def __init__(self, video_files, dataset, window=5, batch_size=64):
        """Constructor
        :param video_files: the root folder of video files
        :param window: the window size of loading videos
        :param random: use randomly generated data instead of loading from a video dataset for a testing purpose
        :param batch_size: batch size
        :param max_padding_len: the max number of region proposal at an image
        """
        random.seed(6)
        np.random.seed(6)
        self.window = window
        self.batch_size = batch_size
        self.dataset = dataset
        self.cnt = 0
        self.max_padding_len = 32
        self.shape = (375, 1242)  # video frame's shape in the format of (height, width)

        # get the frame path and total frame number
        self.files = video_files
        with open(self.files, 'r') as f:
            # video files's format: the path to a family of frames, the total number of frames
            self.video_size = []
            self.video_dir = []
            for line in f.readlines():
                video_dir, size = line.split(',')
                self.video_dir.append(video_dir)
                self.video_size.append(int(size))

        # calculate the prefix sum
        self.cal_prefix()

    def __len__(self):
        return self.prefix[-1]

    def __getitem__(self, index):
        """
        # format 1: input shape = batch_size, seq_length (total bbox number), 5 features
        # for path in input_path:
        #     input_tensors += self.load_tensor(path, self.max_padding_len)
        # input_tensors = torch.as_tensor(input_tensors).reshape(-1, 5)

        # format 2: input shape = batch_size, seq_length (total frame number), 5 features * 30 bbox/frame.
        # Designed for LSTM input. e.g., [16, 5, 160]
        :param index: a random index to access a frame
        :return: input videos in the shape of (window size, total objects, object features)
        """
        # find the associated video index with respect to this referred index
        l = self.find_video_index(index)

        start_frame_index = self.find_start_frame_index(index, l)

        # find N preceding video frames where N equals to the window size
        frame_indexes = [start_frame_index - i for i in range(self.window)]

        selected_video_path = self.video_dir[l]

        input_path = [selected_video_path + '/' + '0' * (6 - len(str(i))) + str(i) + '.txt' for i in frame_indexes][
                     ::-1]
        # load each input tensor and stack them
        input_tensors = np.stack([self.load_tensor(path, self.max_padding_len) for path in input_path])

        # load target tensor. this is only for the training purpose
        target_path = selected_video_path + '/' + '0' * (6 - len(str(start_frame_index + 1))) + str(
            start_frame_index + 1) + '.txt'
        target_tensor = self.load_tensor(target_path, self.max_padding_len, padding=True)

        # rescale the features for data augmentation
        # x_scale, y_scale = random.uniform(0.98, 1.02), random.uniform(0.98, 1.02)
        # x_scale = y_scale = 1

        return input_tensors, target_tensor, input_path + [target_path]

    def get_data_loader(self, batch_size, window_size, shuffle=False):
        self.window = window_size
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    def find_start_frame_index(self, index, video_index):
        start_frame_index = index if video_index < 1 else index - self.prefix[video_index - 1]
        # handle corner case:
        # if cannot find sufficient preceding video frames, randomly generate a starting index.
        if start_frame_index >= self.video_size[video_index] - 1 or start_frame_index < self.window:
            start_frame_index = random.randint(self.window, self.video_size[video_index] - 2)
        return start_frame_index

    def find_video_index(self, index):
        l, r = 0, len(self.video_size)
        while l < r:
            m = l + (r - l) // 2
            if self.prefix[m] < index:
                l = m + 1
            else:
                r = m
        return l

    def cal_prefix(self):
        self.prefix = [0] * len(self.video_size)
        self.prefix[0] = self.video_size[0]

        for i in range(1, len(self.video_size)):
            self.prefix[i] = self.video_size[i] + self.prefix[i - 1]

    @staticmethod
    def load_tensor(filepath, max_padding_len, padding=True):
        """
        load bbox's coordinates and computing complexity in a sing frame.
        When the number of bbox is smaller than the max length, we pad
        at the end.
        :param filepath: image path
        :param max_padding_len: max object number at a single frame
        :return an array in the shape of (max_length, feature number)
        """
        res = np.genfromtxt(filepath, delimiter=" ", invalid_raise=False)
        if len(res) == 0:
            return np.zeros([max_padding_len, 5])
        if res.ndim == 1:
            res = res.reshape(1, -1)
        if padding:
            return np.concatenate((res, np.zeros([max_padding_len - len(res), 5])), axis=0)
        return res

    @staticmethod
    def generate_random_input(batch=100):
        res = []
        for _ in range(batch):
            # randomly generate the number of region proposals in a single frame
            rp_num = np.random.randint(0, 10)
            # randomly generate the normalized coordinates and the computing complexity
            single = np.random.rand(rp_num, 5)
            single = torch.tensor(single)
            res.append(single)
        # important: padding zero region proposals to the frame.
        return torch.nn.utils.rnn.pad_sequence(res, batch_first=True)


def make_dataset(X, Y):
    X_train, Y_train, X_test, Y_test = train_test_split(X, Y, test_size=0.3, random_state=40)
    return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)


def cal_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def slide_window(x, look_back, stride=1):
    X, Y = [], []
    for i in range(0, len(x) + 1 - look_back, stride):
        X.append(np.concatenate(x[i: i + look_back]))
        Y.append(x[i + 1])
    return X, Y