import sys
sys.path.insert(0, "./python")

from jdet.config import init_cfg, get_cfg
from jdet.utils.registry import build_from_cfg, DATASETS

import jdet.data
from jdet.ops.yolo_obb_ops import labels_poly_to_xywhr


def main():
    init_cfg("projects/yolov8_obb/configs/yolov8m_obb_dronevehicle_rgb_teacher.py")
    cfg = get_cfg()

    dataset = build_from_cfg(cfg.dataset.train, DATASETS)

    batch = [dataset[i] for i in range(4)]
    imgs, labels, paths = dataset.collate_batch(batch)

    xywhr_labels = labels_poly_to_xywhr(labels)

    print("Original labels shape:", labels.shape)
    print("XYWHR labels shape:", xywhr_labels.shape)

    print("Original first rows:")
    print(labels[:5])

    print("XYWHR first rows:")
    print(xywhr_labels[:5])

    assert xywhr_labels.shape[1] == 7
    assert xywhr_labels[:, 2].min() >= 0.0
    assert xywhr_labels[:, 2].max() <= 1.0
    assert xywhr_labels[:, 3].min() >= 0.0
    assert xywhr_labels[:, 3].max() <= 1.0

    print("poly2xywhr test success.")


if __name__ == "__main__":
    main()