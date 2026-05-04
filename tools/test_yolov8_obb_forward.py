import sys
sys.path.insert(0, "./python")

import jittor as jt

from jdet.config import init_cfg, get_cfg
from jdet.utils.registry import build_from_cfg, MODELS

# 关键：触发模型注册
import jdet.models.networks


def main():
    init_cfg(
        "projects/yolov8_obb/configs/yolov8m_obb_dronevehicle_rgb_teacher.py"
    )

    cfg = get_cfg()

    model = build_from_cfg(cfg.model, MODELS)
    model.eval()

    x = jt.randn((1, 3, 640, 640))

    with jt.no_grad():
        y = model(x)

    print("Forward success.")
    print("Output type:", type(y))

    if hasattr(y, "shape"):
        print("Output shape:", y.shape)
    elif isinstance(y, (list, tuple)):
        print("Output list length:", len(y))
        for i, item in enumerate(y):
            print(i, item.shape)
    else:
        print(y)


if __name__ == "__main__":
    main()