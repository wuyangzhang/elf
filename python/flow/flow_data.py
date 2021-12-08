import collections
import concurrent
import concurrent.futures
from dataclasses import dataclass, field
from typing import List, Any, Optional

import numpy as np
from zmq.sugar.socket import Socket

from config import Config


@dataclass
class FlowData:
    config: Config = Config()

    total_server_num: int = 0

    cur_frame: Optional[np.ndarray] = None

    frame_count: int = -1
    """The number of frame processed by Elf."""

    executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
    """Executor to handle different tasks."""

    sockets: Optional[List[Socket]] = None
    """Sockets for remote connections."""

    _lrc_socket: Optional[Socket] = None

    inference_results: List[Any] = field(default_factory=list)

    offloading_tasks: List[concurrent.futures.Future] = field(default_factory=list)

    lrc_future: Optional[concurrent.futures.Future] = None
    """Asynchronously LRC request."""

    lrc_inference_result: Any = None

    e2e_latency = collections.defaultdict(list)
    """Key: socket, value: historical e2e latency from the connected server."""

    def create_executor(self) -> None:
        self.total_server_num = len(self.config.servers)

        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.total_server_num + 1
        )

    @property
    def lrc_socket(self):
        return self._lrc_socket
