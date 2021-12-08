from config import Config
from dataset import VideoLoader
from flow import FlowControl
from inference_model import InferenceModelDetectron2
from util.helper import display_imgs

run_ground_truth_mode = True

if __name__ == "__main__":
    config = Config()

    if config.visualization_mode:
        import cv2

    video_dataset = VideoLoader(
        video_dir_root=config.video_dataset_dir,
    )

    model = InferenceModelDetectron2()
    if run_ground_truth_mode:
        model.create_model()

    with FlowControl(config, model) as flow_control:
        for img, _ in video_dataset:
            inference_result = flow_control.run(img)
            render_img = model.render(img, inference_result)
            if run_ground_truth_mode:
                gt_inference_result = model.run(img)
                gt_render_img = model.render(img, gt_inference_result)

            if config.visualization_mode:
                display_imgs(render_img, "Elf")
                if run_ground_truth_mode:
                    display_imgs(gt_render_img, "Full frame (non-Elf) offloading")
                cv2.waitKey(100000)
