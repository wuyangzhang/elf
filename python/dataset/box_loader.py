import numpy as np

from config import Config
from dataset.video_loader import VideoLoader

config = Config()


class BoxLoader(VideoLoader):
    def __init__(
            self,
            video_dir_root: str = config.rp_prediction_training_dataset_path,
            file_type: str = ".txt",
            window_size: int = config.rp_predict_window_size,
            batch_size: int = config.batch_size,
    ):
        super().__init__(
            video_dir_root=video_dir_root,
            file_type=file_type,
            batch_size=batch_size,
            window_size=window_size,
        )

    def __getitem__(self, index):
        # Locates its associated video.
        l = self.find_video_index(index)

        # Locates the frame index at its associated video.
        start_frame_index = self.find_start_frame_index(index, l)

        # Find a group of frame index in a sliding window.
        frame_indexes = [start_frame_index - i for i in range(self.window_size)][::-1]

        input_path = [self.videos[l][x] for x in frame_indexes]
        target_path = self.videos[l][start_frame_index + 1]
        input_tensor = np.stack(
            self.load_tensor(x) for x in input_path
        )
        target_tensor = self.load_tensor(target_path)
        return input_tensor, target_tensor, input_path, target_path

    def load_tensor(self, filepath, padding=True):
        """
        Bbox stored in the format of [x y w h].
        :param filepath:
        :param padding: boolean. need to padding zero bbox at the end?
        :return: all bbox in a given frame specified by the filepath.
        """
        res = np.genfromtxt(filepath, delimiter=" ", invalid_raise=False)
        if len(res) == 0:
            return np.zeros([config.padding_len, 7])

        if res.ndim == 1:
            res = res.reshape(1, -1)

        # Convert x, y, w, h => x, y, x, y.
        res[:, 2] += res[:, 0]
        res[:, 3] += res[:, 1]

        # Normalize.
        res[:, 0] /= res[:, -2]
        res[:, 1] /= res[:, -1]
        res[:, 2] /= res[:, -2]
        res[:, 3] /= res[:, -1]

        if not padding:
            return res

        return np.concatenate(
            (
                res,
                np.zeros(
                    [config.padding_len - len(res), 7])
            ),
            axis=0
        )


if __name__ == '__main__':
    dataset = BoxLoader().get_data_loader(shuffle=True)
    for input_tensor, target_tensor, input_path, target_path in dataset:
        print(input_path, target_path)
