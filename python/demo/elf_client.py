from config import Config
from control import ElfControlManager
from dataset import VideoLoader
from client_model import ClientModelDetectron2


if __name__ == "__main__":
    config = Config()

    video_dataset = VideoLoader(config)

    elf_control = ElfControlManager(config)

    detectron2 = ClientModelDetectron2(config)

    for img, _ in video_dataset:
        elf_control.run(img)
