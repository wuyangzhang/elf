import collections
import concurrent.futures
import random
from concurrent.futures import wait
from typing import Optional, List, Any

import gevent
import numpy as np

from client_model import ClientModelTest, ClientModelDetectron2
from config import Config
from networking import EncoderPickle, EncoderBase, connect_socket, connect_sockets, close_sockets
from partitioning import PartitionManager
from prediction import PredictionManager
from util.helper import scale_in, display_imgs


class ElfControlManager:

    def __init__(self, config: Config):
        self._config: Config = config
        self._predict_mgr: PredictionManager = PredictionManager(config)
        self._partition_mgr: PartitionManager = PartitionManager(config)
        self._app = ClientModelDetectron2(config)
        self._encoder: EncoderBase = EncoderPickle()

        self._sockets = connect_sockets(config.server[:-1])
        self._lrc_socket = connect_socket(config.server[-1][0], config.server[-1][1])
        self.lrc_future = None

        self._total_server_num: int = len(self._sockets)

        self._cur_frame: Optional[np.ndarray] = None
        self._cur_service_id: int = -1

        self.inference_results: List[Any] = [[None] for _ in range(self._total_server_num)]
        self.lrc_inference_result: List[Any] = [None]

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self._end_to_end_latency_record = collections.defaultdict(list)

        self.visualize: bool = False

    def run(self, img: np.ndarray) -> Any:
        """
        The core function of running Elf is to follow the below steps:
        1. query the predicted RPs for the current frame.
        2. collect the available computing resources of different edge servers
        3. perform frame partitioning
        4. offload partitions to servers
        5. render the results
        """

        self._register_new_request(img)
        if self.visualize:
            display_imgs(img)

        if not self._predict_mgr.is_active():
            """Cold start by offloading the full frame to a random edge server."""
            return self._offload_full_frame(img)

        else:
            """Normal Elf flow"""
            self._submit_lrc_offload()

            frame_pars = self._partition_mgr.frame_partition(img, self._predict_mgr.predict_rps())

            if self.visualize:
                display_imgs(frame_pars)

            self._wait_lrc_offload()

            self._offload_rp_boxes(self._partition_mgr._frame_pars)

            return self._merge_partitions()

    def shutdown(self) -> None:
        close_sockets(self._sockets)
        close_sockets([self._lrc_socket])

    def _offload_rp_boxes(self, rp_boxes: List[np.ndarray]) -> None:
        """
        Offload RP boxes into edge servers.
        Get the application results back from the server side.
        :param rp_boxes: RP boxes
        :return: None
        """

        self._submit_offloading_tasks(rp_boxes)

        self._collect_inference_results()

        self._update_rps()

    def _get_inference_results(self):
        return [res[0] for res in self.inference_results]

    def _update_rps(self):
        """Update the new RPs from the detected results"""
        # self._predict_mgr.add_rps(self._predict_mgr.get_gt_rps())
        for inference_result in self._get_inference_results():
            if inference_result is not None:
                self._predict_mgr.add_rps(self._app.extract_rps(inference_result))

        if self._is_lrc_active():
            rps = self._app.extract_rps(self.lrc_inference_result[0])
            scaled_lrc_rps = rps / self._config.lrc_ratio
            self._predict_mgr.update_rps(scaled_lrc_rps)

    def _register_new_request(self, frame: np.ndarray) -> None:
        """Update Elf when receiving a new frame"""
        self._cur_frame = frame
        self._cur_service_id += 1

    def _submit_offloading_tasks(self, rp_boxes: List[np.ndarray]) -> None:
        """Create offloading tasks"""
        ordered_rp_boxes = self._match_rp_boxes_servers(rp_boxes)

        offloading_tasks = list()
        for i in range(len(self._sockets)):
            index = ordered_rp_boxes[i][0]
            offloading_tasks.append(
                self.executor.submit(
                    self._send_frame,
                    self._sockets[index],
                    rp_boxes[index],
                )
            )

        wait(offloading_tasks)

    def _create_inference_results_collection_tasks(self, inference_results: List[Any]):
        tasks = [gevent.spawn(self._receive_inference_results, self._sockets[i], inference_results[i]) for i in
                 range(self._total_server_num)]

        if self._is_lrc_active():
            tasks += [gevent.spawn(self._receive_inference_results, self._lrc_socket, self.lrc_inference_result)]

        return tasks

    def _collect_inference_results(self) -> List[Any]:
        """Get inference results"""

        inference_results_collection_tasks = self._create_inference_results_collection_tasks(self.inference_results)

        gevent.joinall(
            inference_results_collection_tasks
        )

        return self.inference_results

    def _match_rp_boxes_servers(self, rp_boxes: List[np.ndarray], servers=None):
        """Decide the target offloading server for each rp box"""
        index_boxes_size = {index: rp_boxes[index].nbytes for index in range(len(rp_boxes))}
        ordered_rp_boxes = sorted(index_boxes_size.items(), key=lambda x: -x[1])
        return ordered_rp_boxes

    def _offload_full_frame(self, frame: np.ndarray) -> Any:
        """
        Offload the request to an edge server
        :param frame: requested image
        :return: image processing.
        """
        soc = self._get_offload_server()
        self._send_frame(soc, frame)

        self._receive_inference_results(soc, self.inference_results[0])

        self._update_rps()

        return self._get_inference_results()[0]

    def _send_frame(self, soc, img: np.ndarray) -> None:
        arr = self._encoder.encode(img)

        soc.send(arr, copy=False)

    @staticmethod
    def _receive_inference_results(soc, ans: list) -> Any:
        ans[0] = soc.recv_pyobj()

    def _merge_partitions(self) -> Any:
        """ Merge Partitions
        this function takes as the input the historical e2e latency in order to
        evaluate the computation capability of all the involved nodes.
        :return N partitions
        """
        return self._partition_mgr.merge_partition(self._get_inference_results())

    def _get_offload_server(self) -> Any:
        return self._get_random_socket()

    def _get_random_socket(self) -> Any:
        return random.choice(self._sockets)

    def clean_cache(self) -> None:
        self._end_to_end_latency_record = collections.defaultdict(list)
        self.inference_results.clear()

    """LRC components"""

    def _is_lrc_active(self) -> bool:
        return self._predict_mgr.is_active() and self._cur_service_id % 3 == 0

    def _wait_lrc_offload(self) -> None:
        if self._is_lrc_active():
            wait([self.lrc_future])

    def _create_lrc_frame(self, img: np.ndarray) -> np.ndarray:
        return scale_in(img, self._config.lrc_ratio)

    def _submit_lrc_offload(self) -> None:
        if self._is_lrc_active():
            self.lrc_future = self.executor.submit(self._send_frame, self._lrc_socket,
                                                   self._create_lrc_frame(self._cur_frame))
