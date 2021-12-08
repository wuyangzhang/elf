from unittest.mock import patch

import numpy as np
import zmq
import asyncio

from zmq.tests import BaseZMQTestCase
from numpy.testing import assert_array_equal

from config import Config
from flow import FlowControl
from rp_predict import RPPredictor


class TestFlowControl(BaseZMQTestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        super(TestFlowControl, self).setUp()

    def tearDown(self):
        super().tearDown()
        self.loop.close()

    def test_flow_control_instance_creation(self) -> None:
        with patch(
                "networking.socket.connect_sockets",
                return_value=None
        ) and patch(
            "networking.socket.close_sockets",
            return_value=None
        ):
            FlowControl(
                config=Config()
            )

    def test_connect_remote_servers(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://*:%s" % 5053)

        flow_control = FlowControl(
            config=Config(
                servers=[
                    ('localhost', 5053),
                    ('localhost', 5053),
                ],
                lrc_server=('localhost', 5053)
            )
        )
        flow_control.connect_remote_servers()

        flow_control._flow_data.sockets[0].send(b'hello')
        recv_message = socket.recv()

        self.assertEqual(
            recv_message,
            b'hello',
        )

        socket.send(b'world')
        recv_message = flow_control._flow_data.sockets[0].recv()

        self.assertEqual(
            recv_message,
            b'world',
        )

        socket.close()
        context.destroy()
        flow_control.shutdown()

    def test_register_new_frames(self) -> None:
        flow_control = FlowControl(config=Config())
        frame = np.random.rand(100, 100, 3)
        flow_control._register_new_frames(frame)

        self.assertEqual(
            flow_control._flow_data.frame_count,
            0,
        )

        self.assertEqual(
            flow_control._flow_data.offloading_tasks,
            [],
        )

        assert_array_equal(
            flow_control._flow_data.cur_frame,
            frame,
        )

    def test_is_lrc_active_true(self) -> None:
        async def test() -> None:
            with patch.object(
                RPPredictor,
                "is_active",
                return_value=True,
            ):
                flow_control = FlowControl(
                    config=Config(
                        lrc_window=2,
                    )
                )
                flow_control._flow_data.frame_count = 2

                self.assertTrue(
                    flow_control._is_lrc_active(),
                )

        self.loop.run_until_complete(test())

    def test_is_lrc_active_false(self) -> None:
        async def test() -> None:
            with patch.object(
                RPPredictor,
                "is_active",
                return_value=False,
            ):
                flow_control = FlowControl(
                    config=Config()
                )
                self.assertFalse(
                    flow_control._is_lrc_active(),
                )

            with patch.object(
                RPPredictor,
                "is_active",
                return_value=True,
            ):
                flow_control = FlowControl(
                    config=Config(
                        lrc_window=2,
                    )
                )
                flow_control._flow_data.frame_count = 1

                self.assertFalse(
                    flow_control._is_lrc_active(),
                )

        self.loop.run_until_complete(test())

    def test_offloading_tasks(self) -> None:
        async def test() -> None:
            context = zmq.Context()
            socket = context.socket(zmq.REP)
            socket.bind("tcp://*:%s" % 5053)

            flow_control = FlowControl(
                config=Config(
                    servers=[
                        ('localhost', 5053),
                        ('localhost', 5053),
                    ],
                    lrc_server=('localhost', 5053)
                )
            )
            flow_control.connect_remote_servers()

            frame_0 = np.random.rand(100, 100, 3)
            frame_1 = np.random.rand(100, 100, 3)

            rp_boxes = [
                frame_0,
                frame_1,
            ]

            flow_control._submit_offloading_tasks(
                rp_boxes,
            )

            recv_message = socket.recv()
            decode_recv_message = flow_control._frame_encoder.decode(recv_message)

            self.assertEqual(
                decode_recv_message.shape,
                (100, 100, 3),
            )
            socket.send(b'finished')

            recv_message = socket.recv()
            decode_recv_message = flow_control._frame_encoder.decode(recv_message)
            self.assertEqual(
                decode_recv_message.shape,
                (100, 100, 3),
            )

            socket.send(b'finished')

            socket.close()
            context.destroy()
            flow_control.shutdown()

        self.loop.run_until_complete(test())

    def test_collect_inference_results(self) -> None:
        async def test() -> None:
            context = zmq.Context()
            socket = context.socket(zmq.REP)
            socket.bind("tcp://*:%s" % 5053)

            flow_control = FlowControl(
                config=Config(
                    servers=[
                        ('localhost', 5053),
                        ('localhost', 5053),
                    ],
                    lrc_server=('localhost', 5053)
                )
            )
            flow_control.connect_remote_servers()

            frame_0 = np.random.rand(30, 30, 3)
            frame_1 = np.random.rand(100, 100, 3)

            rp_boxes = [
                frame_0,
                frame_1,
            ]

            flow_control._submit_offloading_tasks(
                rp_boxes,
            )

            socket.recv()
            socket.send_pyobj(['finished_0'])

            socket.recv()
            socket.send_pyobj(['finished_1'])

            with patch.object(
                FlowControl,
                "_is_lrc_active",
                return_value=False,
            ):
                flow_control._collect_inference_results()
                self.assertListEqual(
                    flow_control._flow_data.inference_results,
                    [
                        [
                            'finished_0'
                        ],
                        [
                            'finished_1'
                        ]
                    ],
                )

            socket.close()
            context.destroy()
            flow_control.shutdown()

        self.loop.run_until_complete(test())

    def test_context_manager(self) -> None:
        with FlowControl(
            Config()
        ) as flow_control:
            self.assertTrue(
                isinstance(
                    flow_control,
                    FlowControl,
                )
            )

    def test_update_rps(self) -> None:
        with FlowControl(
            Config(),
        ) as flow_control:
            flow_control._flow_data.inference_results = [
                np.random.rand(
                    2, 30, 30,
                ),
                np.random.rand(
                    3, 30, 30,
                ),
                np.random.rand(
                    1, 30, 30,
                ),
            ]

            flow_control._update_inference_results()

            with patch.object(
                FlowControl,
                "_is_lrc_active",
                return_value=True,
            ), patch.object(
                RPPredictor,
                "update_new_rps_from_lrc",
                return_value=None,
            ):
                flow_control._update_inference_results()