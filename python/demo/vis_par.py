# from config import Config
# from rp_partition import RPPartitioner
#
# from dataset import VideoLoader
# from rp_predict.helper import render_bbox
#
# import sys
# from pathlib import Path
#
# import torch
# import cv2
#
# sys.path.append(str(Path.home()) + '/detectron2')
#
#
# def visualize_par_process() -> None:
#     config = Config()
#     dataset = VideoLoader(config)
#     partition_mgr: RPPartitioner = RPPartitioner(config)
#
#     for index, data in enumerate(dataset):
#         tensor = torch.load(f"../output/tensor_{index}.pt")
#         img, _ = data
#         render_img = render_bbox(tensor, img)
#         cv2.imshow("render", render_img)
#         pars = partition_mgr.partition_frame(img, tensor.numpy())
#         for i, par in enumerate(pars):
#             cv2.imshow(f"par{i}", par)
#         cv2.waitKey(10000)
#
#
# if __name__ == "__main__":
#     visualize_par_process()
#
#
#
#
