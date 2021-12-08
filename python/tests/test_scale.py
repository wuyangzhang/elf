# import sys
# from pathlib import Path
#
# sys.path.append(str(Path.home()) + '/detectron2')
#
# from config import config
# from inference_model import ClientModelInterface
#
# from eval import *
# from rp_predict.lrc import *
# from rp_predict import RPPredictor
# from rp_predict.helper import render_bbox
#
# import time
#
#
# def run():
#     app = ClientModelInterface(config, client=False)
#     tensors = RPPredictor.get_tensors()
#
#     dataset = VideoLoader(config)
#
#     for img, path in dataset:
#         width, height = 1242, 375 * 2
#         img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
#         t = tensors.pop(0)
#         t[:, 0] *= width / 1242
#         t[:, 1] *= height / 375
#         t[:, 2] *= width / 1242
#         t[:, 3] *= height / 375
#
#         img = render_bbox(t, img)
#         cv2.imshow("partition results", img)
#         cv2.waitKey(300)
#
#
# if __name__ == "__main__":
#     run()
