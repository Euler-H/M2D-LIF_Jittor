import sys
sys.path.insert(0, "./python")

from jdet.config import init_cfg, get_cfg
from jdet.utils.registry import build_from_cfg, DATASETS

import jdet.data


def main():
    init_cfg("projects/yolov8_obb/configs/yolov8m_obb_dronevehicle_rgb_teacher.py")
    cfg = get_cfg()

    dataset = build_from_cfg(cfg.dataset.train, DATASETS)

    batch = [dataset[i] for i in range(4)]
    imgs, labels = dataset.collate_batch(batch)

    print("Batch image shape:", imgs.shape)
    print("Batch labels shape:", labels.shape)
    print("First path:", dataset.get_img_path(0))
    print("Second path:", dataset.get_img_path(1))
    print("Labels first rows:")
    print(labels[:10])

    assert imgs.shape[0] == 4
    assert imgs.shape[1] == 3
    assert imgs.shape[2] == cfg.imgsz
    assert imgs.shape[3] == cfg.imgsz
    assert labels.shape[1] == 10

    print("YoloOBBDataset batch test success.")


if __name__ == "__main__":
    main()