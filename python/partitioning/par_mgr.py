import random
from typing import List, Any, Union, Optional

import numpy as np
import torch
import torch.nn

from config import Config
from partitioning.par_algo import PartitionAlgorithm


class PartitionManager:
    def __init__(self, config: Config):
        self.config: Config = config
        self.avg_par_size: float = 0

        self._frame_pars: List[np.ndarray] = list()
        self.partitioner = PartitionAlgorithm()
        self._merge_result: Optional[Any] = None

    @property
    def frame_pars(self):
        return self._frame_pars

    @frame_pars.setter
    def frame_pars(self, val: List[np.ndarray]):
        self._frame_pars = val

    def frame_partition(self, img: np.ndarray, rps: np.ndarray) -> List[np.ndarray]:
        self._frame_pars = self.partitioner.frame_partition(img, rps, self.get_partition_meta_data())
        return self._frame_pars

    def merge_partition(self, res: Any) -> Any:
        self._merge_result = self.partitioner.merge_partition(res)
        return self._merge_result

    def get_partition_meta_data(self) -> List[Any]:
        """Prepare any meta data assisting frame partitioning.
        for example, server resource availability
        """
        return [self.config]

    @staticmethod
    def get_par_size(pars: Union[np.ndarray, List[np.ndarray]]):
        return [p.size for p in pars]

    @staticmethod
    def find_intersection(frame_a: np.ndarray, frame_b: np.ndarray) -> Union[float, int]:
        dx = min(frame_a[2], frame_b[2]) - max(frame_a[0], frame_b[0])
        dy = min(frame_a[3], frame_b[3]) - max(frame_a[1], frame_b[1])
        if (dx >= 0) and (dy >= 0):
            return dx * dy
        return 0

    @staticmethod
    def find_par_ratio(frame: np.ndarray, frame_pars: List[np.ndarray]) -> List[float]:
        """
        find the size ratio between each sub-frame and its parent frame
        :param frame:
        :param frame_pars:
        :return:
        """
        return [s.size / frame.size for s in frame_pars]

    @staticmethod
    def add_offset(box: np.ndarray, dx: Union[int, float], dy: Union[int, float]) -> np.ndarray:
        box[:, 0] = box[:, 0] + dx
        box[:, 1] = box[:, 1] + dy
        box[:, 2] = box[:, 2] + dx
        box[:, 3] = box[:, 3] + dy
        return box

    @staticmethod
    def bbox_area(bbox: np.ndarray) -> np.ndarray:
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    def associate_rpbox_server(self, node_num: int, shuffle: bool = False) -> None:
        """
        mapping a partitioning box to a specific node
        partitioning box i belongs the node mapping[i]
        """
        if self.server_par_map is None or shuffle:
            order = [_ for _ in range(node_num)]
            random.shuffle(order)
            self.server_par_map = {order[i]: i for i in range(node_num)}
            self.par_server_map = {i: order[i] for i in range(node_num)}

    def eval_server_rsrc(self, proc_capability):
        """
        Evaluate the processing capability based on the e2e latency

        normalize the processing capability divided by the max latency.

        finally, we find out the available resource of each partitioning box.
        """
        max_latency = max(proc_capability)
        server_rsrc = [(1e-3 + max_latency) / latency for latency in proc_capability]
        return [server_rsrc[self.par_server_map[i]] for i in self.par_server_map]

    @staticmethod
    def show_bbox(image, bbox):
        import matplotlib.pyplot as plt
        import cv2

        if bbox.shape == 3:
            for box in bbox:
                top_left = box[:2]
                bottom_right = box[2:]
                image = cv2.rectangle(
                    image, tuple(top_left), tuple(bottom_right), (255, 255, 255), 5
                )
        else:
            top_left = bbox[:2]
            bottom_right = bbox[2:]
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), (255, 255, 255), 5
            )
        b, g, r = cv2.split(image)  # get b,g,r
        image = cv2.merge([r, g, b])
        plt.imshow(image)
        plt.show()

    @staticmethod
    def show(image):
        import matplotlib.pyplot as plt
        import cv2
        b, g, r = cv2.split(image)  # get b,g,r
        image = cv2.merge([r, g, b])
        plt.imshow(image)
        plt.show()

    @staticmethod
    def check(rps, rp_boxes):
        def box_area(boxes):
            """
            Computes the area of a set of bounding boxes, which are specified by its
            (x1, y1, x2, y2) coordinates.

            Arguments:
                boxes (Tensor[N, 4]): boxes for which the area will be computed. They
                    are expected to be in (x1, y1, x2, y2) format

            Returns:
                area (Tensor[N]): area for each box
            """
            return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        def box_iou(boxes1, boxes2):
            """
            Return intersection-over-union (Jaccard index) of boxes.

            Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

            Arguments:
                boxes1 (Tensor[N, 4])
                boxes2 (Tensor[M, 4])

            Returns:
                iou (Tensor[N, M]): the NxM matrix containing the pairwise
                    IoU values for every element in boxes1 and boxes2
            """

            lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
            rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

            wh = (rb - lt).clamp(min=0)  # [N,M,2]
            inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

            return inter

        iou = box_iou(rps, rp_boxes)
        return iou
