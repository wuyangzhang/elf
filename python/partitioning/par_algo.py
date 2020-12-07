import collections
from abc import ABC, abstractmethod
from typing import List, Any, Optional

import numpy as np
import torch

from bbox.bounding_box import BoxList
from config import Config


class PartitionAlgorithmBase(ABC):
    def __init__(self):
        self.config = Config()
        self.rp_boxes_offset: Optional[np.ndarray] = None

    @abstractmethod
    def frame_partition(self, frame, rps, meta_data: Optional[List[Any]] = None):
        pass

    @abstractmethod
    def merge_partition(self, frames):
        pass

    @staticmethod
    def frame_crop(frame, bbox):
        return frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    @staticmethod
    @abstractmethod
    def total_par_num(config: Config):
        pass


class EqualPartitionAlgorithm(PartitionAlgorithmBase):
    def __init__(self):
        super().__init__()

    def frame_partition(self, frame, rps, meta_data: Optional[List[Any]] = None):
        self.server_par_map = {i: i for i in range(self.config.par_num)}
        self.par_server_map = {i: i for i in range(self.config.par_num)}

        h, w = frame.shape[:2]
        frames = []
        dh = h // self.config.par_num
        self.rp_boxes_offset = [0] * self.config.par_num
        for i in range(self.config.par_num):
            f = frame[dh * i:dh * (i + 1), 0:w, :]
            frames.append(f)
            self.rp_boxes_offset[i] = (0, dh * i)
        return frames

    @staticmethod
    def total_par_num(config: Config):
        return config.total_remote_servers

    def merge_partition(self, frames):
        pass


class PartitionAlgorithm(PartitionAlgorithmBase):
    def __init__(self):
        super().__init__()

    @staticmethod
    def find_rps_boundary(rps: np.ndarray) -> List[int]:
        """Find the external boundary of rps"""
        min_rp_x1 = np.min(rps[:, 0])
        min_rp_y1 = np.min(rps[:, 1])
        max_rp_x2 = np.max(rps[:, 2])
        max_rp_y2 = np.max(rps[:, 3])
        return [min_rp_x1, min_rp_y1, max_rp_x2, max_rp_y2]

    @staticmethod
    def total_par_num(config: Config):
        return config.total_remote_servers

    @staticmethod
    def init_rp_boxes(total_boxes_num: int, rp_boundary: List[int]) -> np.array:
        """Create rp boxes based on rp boundary"""
        min_rp_x1, min_rp_y1, max_rp_x2, max_rp_y2 = rp_boundary

        height_unit, width_unit = max_rp_y2 - min_rp_y1, (max_rp_x2 - min_rp_x1) // total_boxes_num

        rp_boxes = np.zeros([total_boxes_num, 4])

        for i in range(total_boxes_num):
            rp_boxes[i] = np.array(
                [
                    i * width_unit + min_rp_x1,
                    min_rp_y1,
                    (i + 1) * width_unit + min_rp_x1,
                    height_unit + min_rp_y1,
                ]
            )
        return rp_boxes

    @staticmethod
    def find_max_overlaps(rps: np.ndarray, rp_boxes: np.ndarray) -> np.ndarray:
        """Find which rp boxes has the maximal overlap with an rp"""
        a = np.maximum(rps[:, None, 0], rp_boxes[:, 0])
        c = np.minimum(rps[:, None, 2], rp_boxes[:, 2])
        max_par_index = np.argmax(c - a, axis=1)

        return max_par_index

    @staticmethod
    def adjust_rp_boxes(rp_boxes: np.ndarray, rp_assoc: List[np.ndarray]) -> np.ndarray:
        """Rescale each partitioning box in order to fully cover its associated RPs."""
        for i in range(len(rp_assoc)):
            if len(rp_assoc[i]) == 0:
                rp_boxes[i] = np.array([0, 0, 5, 5])
                continue
            rp_boxes[i, :2] = np.min(rp_assoc[i][:, :2], axis=0)
            rp_boxes[i, 2:] = np.max(rp_assoc[i][:, 2:], axis=0)

        return rp_boxes

    @staticmethod
    def rescale_rp_boxes(rp_boxes: np.ndarray, config: Config) -> np.ndarray:
        rp_width = (rp_boxes[:, 2] - rp_boxes[:, 0]) * config.rescale_ratio
        rp_height = (rp_boxes[:, 3] - rp_boxes[:, 1]) * config.rescale_ratio
        rp_boxes[:, 0] = np.maximum(0, rp_boxes[:, 0] - rp_width)
        rp_boxes[:, 1] = np.maximum(0, rp_boxes[:, 1] - rp_height)
        rp_boxes[:, 2] = np.minimum(config.frame_width, rp_boxes[:, 2] + rp_width)
        rp_boxes[:, 3] = np.minimum(config.frame_height, rp_boxes[:, 3] + rp_height)

        return rp_boxes

    @staticmethod
    def reformat(imgs: List[np.ndarray]):
        """ This function is specifically designed for the usage of nvJPEG
        :param imgs:
        :return:
        """
        for i in range(len(imgs)):
            xflag = yflag = 0
            if imgs[i].shape[0] % 2 != 0:
                xflag = 1
            if imgs[i].shape[1] % 2 != 0:
                yflag = 1
            imgs[i] = imgs[i][:imgs[i].shape[0] - xflag, :imgs[i].shape[1] - yflag, :].copy()

        return imgs

    def frame_partition(self, frame, rps, meta_data: Optional[List[Any]] = None) -> List[np.ndarray]:
        """A frame partition scheme.
        Multi-Capacity Bin Packing problem.
        This frame partition scheme performs based on the position of bounding boxes(bbox),
        the weights of bbox that indicate the potential computing costs, and
        the available computing resources that are represented by the historical
        computing time.
        step 1. Equal partition.
        step 2. Computation complexity aware placement.
        for each bbox, check whether it is overlapped with multiple partitions.
        if not, add it to that partition and change the partition weight.
        if yes, select one of partitions based on its current weight. Each partition should
        have equal probability to be selected.
        :param frame: the target frame to be partition.
        :param rps: all the region proposals along with their coordinates!
        :param meta_data
        :return N partitions
        """
        config: Config = meta_data[0]
        # initialize par boxes by equally partitioning the frame.
        # the number of partitions equals to the number of involved servers.
        total_par_num = self.total_par_num(config)

        rp_boxes = self.init_rp_boxes(total_par_num, self.find_rps_boundary(rps))

        index_max_rp_box_overlap = self.find_max_overlaps(rps, rp_boxes)

        rp_assoc = [rps[index_max_rp_box_overlap == i] for i in range(len(rp_boxes))]

        rp_boxes = self.adjust_rp_boxes(rp_boxes, rp_assoc)

        rp_boxes = self.rescale_rp_boxes(rp_boxes, config)

        rp_boxes = rp_boxes.astype(int)

        rp_boxes = sorted(rp_boxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))

        self.rp_boxes_offset = [(par[0], par[1]) for par in rp_boxes]

        pars = [self.frame_crop(frame, rp_boxes[i]) for i in
                range(total_par_num)]

        return self.reformat(pars)

    def merge_partition(self, res):
        """
        Merge results from distribution
            # RPs will contains all Rps results from the distributed results
            # extras wil stores all extra fields from the distributed results.
            Returns RPs with the offset compensation & merged mask
            assume the distributed_res stores the results in the server order
                                           frame width
            ---------------------------------------------------------
            |
            |  offset_width       RP width
            |<-------------><------------------->
            |               |                   |
            |               |      RP         |
            |               |                   |
            |               |                   |
            |               |                   |
                            ---------------------
            padding zeros around masks in 4 directions: left, right, top, bottom.
            left: offset width
            right: total width - offset_width - RP_width
            top: offset height
            bottom: total height - offset_height - RP_height
        """

        for index, pred in enumerate(res):
            if len(pred["instances"]) == 0:
                continue
            w, h = self.rp_boxes_offset[index]

            # offset boxes
            pred["instances"].get_fields()["pred_boxes"].tensor[:, 0] += w
            pred["instances"].get_fields()["pred_boxes"].tensor[:, 1] += h
            pred["instances"].get_fields()["pred_boxes"].tensor[:, 2] += w
            pred["instances"].get_fields()["pred_boxes"].tensor[:, 3] += h

            # offset mask
            if self.config.merge_mask:
                shape = pred["instances"].image_size
                pad = torch.nn.ConstantPad2d((w, self.config.frame_width - w - shape[1],
                                              h, self.config.frame_height - h - shape[0]),
                                             0)
                pred["instances"].get_fields()["pred_masks"] = pad(
                    pred["instances"].get_fields()["pred_masks"][:, :, ])

        index = 0
        while index < len(res) and len(res[index]["instances"]) == 0:
            index += 1

        # Early exit if the current frame contains zero object of interest
        if index == len(res):
            return res[0]

        ans = res[index]

        # Modify size
        ans["instances"]._image_size = (self.config.frame_height, self.config.frame_width)
        for i in range(index, len(res)):
            if len(res[i]["instances"]) == 0:
                continue
            ans["instances"].get_fields()["pred_boxes"].tensor = torch.cat(
                [ans["instances"].get_fields()["pred_boxes"].tensor,
                 res[i]["instances"].get_fields()["pred_boxes"].tensor], dim=0)

            if self.config.merge_mask:
                for k in ans["instances"].get_fields().keys():
                    if k == "pred_boxes":
                        continue
                    ans["instances"].get_fields()[k] = torch.cat([ans["instances"].get_fields()[k],
                                                                  res[i]["instances"].get_fields()[k]], dim=0)

        return ans

    @staticmethod
    def merge_bbox(bboxes):
        bbox_res = []
        extras = collections.defaultdict(list)

        for bbox in bboxes:

            if len(bbox.bbox) == 0:
                continue

            bbox_res.append(bbox.bbox)

            # add constant extra fields, labels. scores & adjusted mask. NOT changed
            for key in bbox.extra_fields.keys():
                extras[key].append(bbox.extra_fields[key])

        # handle corner case if no bbox has been detected.
        if len(bbox_res) == 0:
            return BoxList.get_empty()

        # merge bbox from different partitions
        bbox_res = torch.cat(bbox_res, dim=0).float()

        # merge extra keys from different pars
        for key in extras.keys():
            extras[key] = torch.cat(extras[key], dim=0)

        bbox = BoxList(bbox_res, extras['mask'].shape[2:][::-1])
        for key in extras.keys():
            bbox.extra_fields[key] = extras[key]
        return bbox
