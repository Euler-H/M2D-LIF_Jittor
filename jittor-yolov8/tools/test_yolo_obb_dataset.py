import sys
sys.path.insert(0, "./python")

from jdet.config import init_cfg, get_cfg
from jdet.utils.registry import build_from_cfg, DATASETS

import jdet.data


def main():
    init_cfg("projects/yolov8_obb/configs/yolov8m_obb_dronevehicle_rgb_teacher.py")
    cfg = get_cfg()

    dataset = build_from_cfg(cfg.dataset.train, DATASETS)

    img, labels, path = dataset[0]

    print("Image path:", path)
    print("Image shape:", img.shape)
    print("Labels shape:", labels.shape)
    print("Labels first rows:")
    print(labels[:5])

    assert img.shape[0] == 3
    assert img.shape[1] == cfg.imgsz
    assert img.shape[2] == cfg.imgsz

    if len(labels):
        assert labels.shape[1] == 9
        assert labels[:, 1:].min() >= 0
        assert labels[:, 1:].max() <= 1

    print("YoloOBBDataset single sample test success.")


if __name__ == "__main__":
    main()