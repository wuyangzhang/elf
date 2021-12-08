import concurrent
import concurrent.futures
from unittest.mock import patch
import asyncio

import numpy as np
import zmq
from zmq.tests import BaseZMQTestCase
from numpy.testing import assert_array_equal
import zmq.asyncio as zaio

from flow.offload import (
    send_frame,
    submit_offloading_tasks,
    receive_inference_results,
    schedule_inference_results_collection,
    collect_inference_results,
)

from networking import EncoderPickle


class TestSocket(BaseZMQTestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        super(TestSocket, self).setUp()

    def tearDown(self):
        super().tearDown()
        self.loop.close()
        assert zaio._selectors == {}

    def test_send_frame(self) -> None:
        async def test() -> None:
            with patch.object(
                EncoderPickle,
                "encode",
                return_value=b"hi",
            ):
                sender, receiver = self.create_bound_pair(zmq.PUSH, zmq.PULL)
                send_frame(
                    socket=sender,
                    frame=np.random.rand(100, 100, 3),
                    frame_encoder=EncoderPickle(),
                )

                recv_message = receiver.recv()
                self.assertEqual(recv_message, b"hi")

        self.loop.run_until_complete(test())

    def test_receive_frame(self) -> None:
        async def test() -> None:
            frame = np.random.rand(100, 100, 3)
            sender, receiver = self.create_bound_pair(zmq.PUSH, zmq.PULL)
            send_frame(
                socket=sender,
                frame=frame,
                frame_encoder=EncoderPickle(),
            )

            recv_message = receiver.recv()
            decode_message = EncoderPickle().decode(recv_message)
            assert_array_equal(decode_message, frame)

        self.loop.run_until_complete(test())

    def test_submit_offloading_tasks(self) -> None:
        async def test() -> None:
            frame_0 = np.random.rand(100, 100, 3)
            frame_1 = np.random.rand(100, 100, 3)

            rp_boxes = [
                frame_0,
                frame_1,
            ]
            sender_0, receiver_0 = self.create_bound_pair(zmq.PUSH, zmq.PULL)
            sender_1, receiver_1 = self.create_bound_pair(zmq.PUSH, zmq.PULL)

            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=2
            )

            submit_offloading_tasks(
                frames=rp_boxes,
                sockets=[sender_0, sender_1],
                executor=executor,
                frame_encoder=EncoderPickle(),
            )

            recv_message_0 = receiver_0.recv()
            decode_message_0 = EncoderPickle().decode(recv_message_0)
            assert_array_equal(decode_message_0, frame_0)

            recv_message_1 = receiver_1.recv()
            decode_message_1 = EncoderPickle().decode(recv_message_1)
            assert_array_equal(decode_message_1, frame_1)

        self.loop.run_until_complete(test())

    def test_receive_inference_results(self) -> None:
        async def test() -> None:
            sender, receiver = self.create_bound_pair(zmq.PUSH, zmq.PULL)

            sender.send_pyobj(['hello'], copy=False)

            receive_message = receive_inference_results(
                receiver,
            )

            self.assertEqual(
                receive_message,
                ['hello'],
            )

        self.loop.run_until_complete(test())

    def test_schedule_inference_results_collection(self) -> None:
        async def test() -> None:
            sender_0, receiver_0 = self.create_bound_pair(zmq.PUSH, zmq.PULL)
            sender_1, receiver_1 = self.create_bound_pair(zmq.PUSH, zmq.PULL)

            sender_0.send_pyobj(['hello'], copy=False)
            sender_1.send_pyobj(['world'], copy=False)

            schedule_inference_results_collection(
                [
                    receiver_0,
                    receiver_1,
                ]
            )

        self.loop.run_until_complete(test())

    def test_collect_inference_results_no_lrc(self) -> None:
        async def test() -> None:
            sender_0, receiver_0 = self.create_bound_pair(zmq.PUSH, zmq.PULL)
            sender_1, receiver_1 = self.create_bound_pair(zmq.PUSH, zmq.PULL)

            sender_0.send_pyobj(['hello'], copy=False)
            sender_1.send_pyobj(['world'], copy=False)

            results = collect_inference_results(
                [
                    receiver_0,
                    receiver_1,
                ],
            )

            self.assertEqual(
                results,
                ([['hello'], ['world']])
            )

        self.loop.run_until_complete(test())

    def test_collect_inference_results_with_lrc(self) -> None:
        async def test() -> None:
            sender_0, receiver_0 = self.create_bound_pair(zmq.PUSH, zmq.PULL)
            sender_1, receiver_1 = self.create_bound_pair(zmq.PUSH, zmq.PULL)
            sender_2, receiver_2 = self.create_bound_pair(zmq.PUSH, zmq.PULL)

            sender_0.send_pyobj(['hello'], copy=False)
            sender_1.send_pyobj(['world'], copy=False)
            sender_2.send_pyobj(['lrc'], copy=False)

            results = collect_inference_results(
                [
                    receiver_0,
                    receiver_1,
                ],
                receiver_2,
            )

            self.assertEqual(
                results,
                ([['hello'], ['world'], ['lrc']])
            )

        self.loop.run_until_complete(test())