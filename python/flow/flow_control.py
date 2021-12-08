import random
from typing import List, Any

import numpy as np

from config import Config
from flow.flow_data import FlowData
from flow.lrc import create_lrc_frame, rescale_lrc_results
from flow.offload import (
    send_frame,
    submit_offloading_tasks,
    wait_offload_finished,
    collect_inference_results,
    receive_inference_results,
    get_random_socket,
)
from flow.offload_schedule import schedule_offloading
from inference_model import InferenceModelTest, InferenceModelInterface
from networking import EncoderPickle, EncoderBase, connect_sockets, close_sockets
from rp_partition import RPPartitioner
from rp_predict import RPPredictor
from rp_predict.util import render_bbox

from util.helper import display_imgs


class FlowControl:
    """This class controls how Elf processes each video frame."""

    def __init__(
        self,
        config: Config = Config(),
        inference_model: InferenceModelInterface = InferenceModelTest()
    ):
        self._config: Config = config
        self._rp_predictor: RPPredictor = RPPredictor(config)
        self._rp_partitioner: RPPartitioner = RPPartitioner(config)
        self._flow_data: FlowData = FlowData()
        self._flow_data.create_executor()

        """A machine learning model to run inference."""
        self.inference_model = inference_model

        """An encoder to encode&decode video frames."""
        self._frame_encoder: EncoderBase = EncoderPickle()

        """If the flag is set True, it shows the received frame and frame partitions."""
        self.visualization_mode: bool = config.visualization_mode

    def run(
        self,
        frame: np.ndarray
    ) -> Any:
        """
        It takes a video frame as the input and returns the model inference result.
        :param frame: np.ndarray, an input video frame.
        :return: model inference result.
        """
        self._register_new_frames(frame)

        if not self._rp_predictor.is_active():
            self._run_cold_start(frame)
            return self._flow_data.inference_results[0]

        """Warm up Elf flow starts."""
        if self._is_lrc_active():
            print("Run LRC flow")
            self._submit_lrc_offloading()
            self._wait_offloading_finished()

        predicted_rps = self.rp_predictor.predict_rps()

        rp_partitions = self._rp_partitioner.partition_frame(
            frame=frame,
            rps=predicted_rps,
        )

        frame_partitions = [rp_partition.partition for rp_partition in rp_partitions]
        frame_offsets = [rp_partition.offset for rp_partition in rp_partitions]

        if self.visualization_mode:
            frame_with_box_render = render_bbox(predicted_rps.astype(int), self._flow_data.cur_frame)
            frame_partition_coordinates = np.stack([rp_partition.coordinates for rp_partition in rp_partitions])
            frame_with_box_render = render_bbox(
                frame_partition_coordinates.astype(int),
                frame_with_box_render,
                color=(0, 255, 0),
            )

            display_imgs(
                frame_with_box_render,
                "Elf internal process (prediction + partitioning)"
            )

        # TODO: Update dynamic offloading
        #offloading_partitions: List[np.ndarray] = schedule_offloading(frame_partitions)

        self._submit_offloading_tasks(frame_partitions)

        self._wait_offloading_finished()

        self._collect_inference_results()

        self._update_inference_results(frame_offsets)

        return self._merge_partitions(frame_offsets)

    def __enter__(self):
        self.connect_remote_servers()
        return self

    def __exit__(self, type, value, traceback):
        self.shutdown()

    def connect_remote_servers(self) -> None:
        """Connect remote servers that are available for model inference and LRC service."""
        self._flow_data.sockets = connect_sockets(
            self._config.servers
        )

        # TODO: Dynamically assign LRC server.
        self._flow_data._lrc_socket = connect_sockets(
            [random.choice(self._config.servers)]
        )[0]

    def shutdown(self) -> None:
        self._flow_data.executor.shutdown()
        close_sockets(
            self._flow_data.sockets
        )

    def _register_new_frames(self, frame: np.ndarray) -> None:
        """Update Elf status when receiving a new frame."""
        self._flow_data.cur_frame = frame
        self._flow_data.frame_count += 1
        self._flow_data.offloading_tasks = []
        self._flow_data.inference_results = []
        self._config.frame_height, self._config.frame_width = frame.shape[:2]
        print(f"Process frame index {self._flow_data.frame_count}.")

    def _run_cold_start(
        self,
        frame: np.ndarray
    ) -> None:
        socket = get_random_socket(
            self._flow_data.sockets
        )

        send_frame(
            socket,
            frame,
            self._frame_encoder,
        )

        inference_result = receive_inference_results(socket)
        self._flow_data.inference_results.append(inference_result)

        self._update_inference_results()

    def _submit_lrc_offloading(self):
        lrc_offloading_task = submit_offloading_tasks(
            frames=[
                create_lrc_frame(
                    self._flow_data.cur_frame,
                    self._config.lrc_downsample_ratio,
                ),
            ],
            frame_encoder=self._frame_encoder,
            sockets=[self._flow_data.lrc_socket],
            executor=self._flow_data.executor
        )
        self._flow_data.offloading_tasks += lrc_offloading_task

    def _submit_offloading_tasks(self, offloading_partitions: List[np.ndarray]):
        tasks = submit_offloading_tasks(
            frames=offloading_partitions,
            frame_encoder=self._frame_encoder,
            sockets=self._flow_data.sockets,
            executor=self._flow_data.executor,
        )
        self._flow_data.offloading_tasks += tasks

    def _wait_offloading_finished(self) -> None:
        wait_offload_finished(self._flow_data.offloading_tasks)

    def _collect_inference_results(self) -> None:
        """
        Collect inference results for both normal offloading tasks and LRC task.
        """
        if self._is_lrc_active():
            inference_results = collect_inference_results(
                sockets=self._flow_data.sockets,
                lrc_socket=self._flow_data.lrc_socket
            )
            self._flow_data.inference_results = inference_results[:-1]
            self._flow_data.lrc_inference_result = inference_results[-1]
        else:
            self._flow_data.inference_results = collect_inference_results(
                sockets=self._flow_data.sockets
            )

    def _is_lrc_active(self) -> bool:
        """Check if a LRC service should be performed at this round."""
        return self._rp_predictor.is_active() and self._flow_data.frame_count % self._config.lrc_window == 0

    def _update_inference_results(
        self,
        offsets: List[List[int]] = [[0, 0]],
    ) -> None:
        """Update the inference results from both normal and LRC flows."""
        frames_rps = []
        is_new_rp_detected = False
        for inference_result, offset in zip(self._flow_data.inference_results, offsets):
            rps = self.inference_model.extract_rps(inference_result)
            if len(rps) == 0:
                continue
            is_new_rp_detected = True
            rps = self._add_offsets(rps, offset)
            frames_rps.append(rps)

        if is_new_rp_detected:
            frames_rps = np.concatenate(frames_rps)
            self._rp_predictor.add_new_rps(frames_rps)

        if not self._is_lrc_active():
            return None

        lrc_rps = self.inference_model.extract_rps(
            self._flow_data.lrc_inference_result
        )

        if not is_new_rp_detected:
            self._rp_predictor.add_new_rps(
                rescale_lrc_results(
                    lrc_rps,
                    self._config.lrc_downsample_ratio,
                )
            )
        else:
            self._rp_predictor.update_new_rps_from_lrc(
                rescale_lrc_results(
                    lrc_rps,
                    self._config.lrc_downsample_ratio,
                )
            )

        return None

    def _merge_partitions(
        self,
        offsets: List[int],
    ) -> Any:
        """ Merge the model inference results from different servers."""
        return self.inference_model.merge(
            self._flow_data.inference_results,
            offsets,
            frame_height=self._config.frame_height,
            frame_width=self._config.frame_width,
            merge_mask=self._config.merge_mask,
        )

    def _add_offsets(
        self,
        source_rp: np.ndarray,
        offset: List[int],
    ) -> np.ndarray:
        rp = source_rp.copy()
        rp[:, 0] += offset[0]
        rp[:, 1] += offset[1]
        rp[:, 2] += offset[0]
        rp[:, 3] += offset[1]
        return rp

    @property
    def rp_partitioner(self) -> RPPartitioner:
        return self._rp_partitioner

    @property
    def rp_predictor(self) -> RPPredictor:
        return self._rp_predictor

