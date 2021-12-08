import unittest
import pickle

import torch
import cv2

from inference_model.inference_model_detectron2 import InferenceModelDetectron2


class TestInferenceModelDetectron2(unittest.TestCase):
    def test_run(self) -> None:
        model = InferenceModelDetectron2()
        model.create_model()
        image = cv2.imread("./tests/test_images/test_image_0.jpeg")
        result = model.run(
            image
        )
        self.assertIsNotNone(result)

    def test_merge(self) -> None:
        example_inference_0 = pickle.load(
            open('./tests/inference_model/detectron2_example_inference.pkl', 'rb')
        )
        example_inference_1 = pickle.load(
            open('./tests/inference_model/detectron2_example_inference_1.pkl', 'rb')
        )
        """
        {'instances': Instances(num_instances=4, image_height=94, image_width=328, fields=[pred_boxes: Boxes(tensor([[123.1625,  42.0007, 142.1791,  57.2983],
        [197.0344,  41.1073, 202.7980,  55.3988],
        [175.7267,  37.2345, 188.9875,  45.5555],
        [152.2210,  40.4764, 159.2845,  55.1196]])), scores: tensor([0.9486, 0.8838, 0.6848, 0.5458]), pred_classes: tensor([2, 0, 2, 0]), 
        """
        model = InferenceModelDetectron2()

        merge_result = model.merge(
            inference_results=[
                example_inference_0,
                example_inference_1,
            ],
            offsets=[
                [100, 100],
                [200, 200]
            ],
            frame_height=680,
            frame_width=1280,
            merge_mask=False,
        )

        self.assertTrue(
            torch.equal(
                merge_result['instances'].get_fields()['pred_boxes'].tensor,
                torch.cat(
                    [
                        torch.tensor(
                            [
                                [123.1625, 42.0007, 142.1791, 57.2983],
                                [197.0344, 41.1073, 202.7980, 55.3988],
                                [175.7267, 37.2345, 188.9875, 45.5555],
                                [152.2210, 40.4764, 159.2845, 55.1196]
                            ]
                        ) + 100,
                        torch.tensor(
                            [
                                [123.1625, 42.0007, 142.1791, 57.2983],
                                [197.0344, 41.1073, 202.7980, 55.3988],
                                [175.7267, 37.2345, 188.9875, 45.5555],
                                [152.2210, 40.4764, 159.2845, 55.1196]
                            ]
                        ) + 200
                    ]
                )

            )
        )
