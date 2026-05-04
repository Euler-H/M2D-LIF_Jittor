import sys
sys.path.insert(0, "./python")

import jittor as jt

from jdet.config import init_cfg, get_cfg
from jdet.utils.registry import build_from_cfg, MODELS, DATASETS

import jdet.models.networks
import jdet.data


def main():
    init_cfg("projects/yolov8_obb/configs/yolov8m_obb_dronevehicle_rgb_teacher.py")
    cfg = get_cfg()

    dataset = build_from_cfg(cfg.dataset.train, DATASETS)
    model = build_from_cfg(cfg.model, MODELS)

    model.train()

    batch = [dataset[i] for i in range(2)]
    images, targets = dataset.collate_batch(batch)

    losses = model(images, targets)

    print("Loss dict:")
    for k, v in losses.items():
        print(k, v)

    total = sum(losses.values())
    print("Total loss:", total)

    assert "box_loss" in losses
    assert "cls_loss" in losses
    assert "dfl_loss" in losses
    assert "angle_loss" in losses

    # 触发一次反向图构建
    total.sync()

    print("YOLOv8-OBB loss step test success.")


if __name__ == "__main__":
    main()