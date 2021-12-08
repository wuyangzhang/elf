import concurrent.futures
import random
from typing import List, Any, Optional

import gevent
import numpy as np
from zmq.sugar.socket import Socket

from networking import EncoderBase


def send_frame(
    socket: Socket,
    frame: np.ndarray,
    frame_encoder: EncoderBase,
) -> None:
    """Encode a frame and send it to a remote socket."""
    encoded_frame = frame_encoder.encode(
        frame
    )

    socket.send(
        encoded_frame,
        copy=False
    )

    return None


def submit_offloading_tasks(
    frames: List[np.ndarray],
    frame_encoder: EncoderBase,
    sockets: List[Socket],
    executor: concurrent.futures.ThreadPoolExecutor,
) -> List[concurrent.futures.Future]:
    """Submit offloading tasks to the connected sockets."""
    offloading_tasks = [
        executor.submit(
            send_frame,
            socket,
            frame,
            frame_encoder,
        ) for frame, socket in zip(frames, sockets)
    ]

    return offloading_tasks


def wait_offload_finished(
    offloading_tasks: List[concurrent.futures.Future]
) -> None:
    """Wait offloading to be done."""

    concurrent.futures.wait(offloading_tasks)


def receive_inference_results(
    socket: Socket,
) -> Any:
    """Save the result returned by a remote socket to a list."""
    return socket.recv_pyobj()


def schedule_inference_results_collection(
    sockets: List[Socket],
) -> List[Any]:
    """Schedule to collect inference results from remote servers."""

    tasks = [
        gevent.spawn(
            receive_inference_results,
            sockets[i],
        ) for i in range(len(sockets))
    ]

    return tasks


def collect_inference_results(
    sockets: List[Socket],
    lrc_socket: Optional[Socket] = None,
) -> List[Any]:
    """Collect inference results from remote servers."""

    inference_results_collection_tasks = schedule_inference_results_collection(
        sockets,
    )

    if lrc_socket is not None:
        inference_results_collection_tasks += schedule_inference_results_collection(
            [lrc_socket],
        )

    gevent.joinall(
        inference_results_collection_tasks
    )

    return [task.value for task in inference_results_collection_tasks]


def get_random_socket(sockets: List[Socket]) -> Socket:
    return random.choice(sockets)
