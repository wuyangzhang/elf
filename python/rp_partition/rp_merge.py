from typing import List, Any
import collections

import torch

from rp_partition.rp_partition import RPPartition


class RPPartitioner:
    def merge_partitions(self, partitions: List[RPPartition]) -> Any:
        """Merge the model inference results from different partitions as it runs on the original frame."""
        raise NotImplementedError

    def merge_partitions(self, res):
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
