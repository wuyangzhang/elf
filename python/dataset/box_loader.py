import numpy as np

from config import Config
from dataset.video_loader import VideoLoader


class BBoxLoader(VideoLoader):
    def __init__(self, config=Config()):
        config.video_dir_root = config.video_dir_roots[2]
        self.file_type = '.txt'
        super().__init__(config)
        self.batch_size = config.batch_size

    def __getitem__(self, index):
        # Locates its associated video
        l = self.find_video_index(index)

        # Locates the frame index at its associated video
        start_frame_index = self.find_start_frame_index(index, l)

        # Find a group of frame index when with a window
        frame_indexes = [start_frame_index - i for i in range(self._window_size)][::-1]

        input_path = [self.videos[l][x] for x in frame_indexes]
        target_path = self.videos[l][start_frame_index + 1]
        input_tensor = np.stack(self.load_tensor(x) for x in input_path)
        target_tensor = self.load_tensor(target_path)
        return input_tensor, target_tensor, input_path, target_path

    def load_tensor(self, filepath, padding=True):
        """
        Bbox stored in the format of [x y w h]
        :param filepath:
        :param padding: boolean. need to padding zero bbox at the end?
        :return: all bbox in a given frame specified by the filepath
        """
        res = np.genfromtxt(filepath, delimiter=" ", invalid_raise=False)
        if len(res) == 0:
            return np.zeros([self.config.padding_len, 7])
        if res.ndim == 1:
            res = res.reshape(1, -1)

        # Convert x, y, w, h => x, y, x, y
        res[:, 2] += res[:, 0]
        res[:, 3] += res[:, 1]

        # Normalize
        res[:, 0] /= res[:, -2]
        res[:, 1] /= res[:, -1]
        res[:, 2] /= res[:, -2]
        res[:, 3] /= res[:, -1]

        if padding:
            return np.concatenate((res, np.zeros([self.config.padding_len - len(res), 7])), axis=0)
        return res


if __name__ == '__main__':
    from config import Config

    config = Config()
    dataset = BBoxLoader().get_data_loader()
    for pred, gt in dataset:
        print(pred, gt)
