import os
import sys
sys.path.insert(0, "./python")

import cv2
import numpy as np
import jittor as jt

from jdet.config import init_cfg, get_cfg
from jdet.utils.registry import build_from_cfg, MODELS

import jdet.models.networks

from jdet.ops.yolo_obb_ops import (
    postprocess_obb_np,
    xywhr2poly_np,
)


CLASS_NAMES = ["car", "truck", "freight_car", "bus", "van"]


def letterbox(img, new_shape=640, color=(114, 114, 114)):
    h0, w0 = img.shape[:2]

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / h0, new_shape[1] / w0)

    new_unpad = (
        int(round(w0 * r)),
        int(round(h0 * r)),
    )

    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    if (w0, h0) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    img = cv2.copyMakeBorder(
        img,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color,
    )

    meta = dict(
        r=r,
        pad=(left, top),
        orig_shape=(h0, w0),
        input_shape=new_shape,
    )

    return img, meta


def preprocess_image(img_path, imgsz=640):
    img0 = cv2.imread(img_path)
    assert img0 is not None, "Image not found: {}".format(img_path)

    img, meta = letterbox(img0, imgsz)

    img_rgb = img[:, :, ::-1]
    img_rgb = img_rgb.transpose(2, 0, 1)
    img_rgb = np.ascontiguousarray(img_rgb, dtype=np.float32) / 255.0

    img_tensor = jt.array(img_rgb[None])

    return img0, img_tensor, meta


def scale_dets_back(dets, meta):
    """
    将 letterbox 输入尺度下的 dets 映射回原图尺度。

    dets: [M, 7], cx, cy, w, h, theta, score, cls
    """
    if dets is None or len(dets) == 0:
        return dets

    dets = dets.copy()

    r = meta["r"]
    left, top = meta["pad"]
    h0, w0 = meta["orig_shape"]

    dets[:, 0] = (dets[:, 0] - left) / r
    dets[:, 1] = (dets[:, 1] - top) / r
    dets[:, 2] = dets[:, 2] / r
    dets[:, 3] = dets[:, 3] / r

    dets[:, 0] = np.clip(dets[:, 0], 0, w0 - 1)
    dets[:, 1] = np.clip(dets[:, 1], 0, h0 - 1)
    dets[:, 2] = np.clip(dets[:, 2], 1, w0)
    dets[:, 3] = np.clip(dets[:, 3], 1, h0)

    return dets


def draw_dets(img, dets, save_path):
    vis = img.copy()

    if dets is None or len(dets) == 0:
        cv2.imwrite(save_path, vis)
        return

    boxes = dets[:, :5]
    scores = dets[:, 5]
    cls_ids = dets[:, 6].astype(np.int64)

    polys = xywhr2poly_np(boxes)

    for poly, score, cls_id in zip(polys, scores, cls_ids):
        pts = poly.reshape(4, 2).astype(np.int32)

        cv2.polylines(
            vis,
            [pts],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2,
        )

        x, y = pts[0]
        name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)

        cv2.putText(
            vis,
            "{} {:.2f}".format(name, float(score)),
            (int(x), int(y) - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(save_path, vis)


def load_checkpoint(model, ckpt_path):
    assert os.path.exists(ckpt_path), "Checkpoint not found: {}".format(ckpt_path)

    state = jt.load(ckpt_path)

    if isinstance(state, dict):
        if "model" in state:
            model.load_state_dict(state["model"])
        elif "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            model.load_state_dict(state)
    else:
        model.load_state_dict(state)


def main():
    config = "projects/yolov8_obb/configs/yolov8m_obb_dronevehicle_rgb_teacher.py"

    ckpt = "./work_dirs/yolov8m_obb_dronevehicle_rgb_teacher_tal_debug/checkpoints/ckpt_1.pkl"

    img_path = "/root/JDet/DroneVehicle_train_val/images/val/00001.jpg"

    save_path = "./work_dirs/yolov8m_obb_infer_test/00001_pred.jpg"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    init_cfg(config)
    cfg = get_cfg()

    model = build_from_cfg(cfg.model, MODELS)
    load_checkpoint(model, ckpt)

    model.eval()

    img0, img_tensor, meta = preprocess_image(img_path, imgsz=cfg.imgsz_test)

    with jt.no_grad():
        pred = model(img_tensor)

    pred_np = pred[0].numpy()

    dets = postprocess_obb_np(
        pred_np,
        conf_thres=0.05,
        iou_thres=0.5,
        max_det=100,
    )

    dets = scale_dets_back(dets, meta)

    print("Image:", img_path)
    print("Raw pred shape:", pred_np.shape)
    print("Detections:", dets.shape)
    if len(dets):
        print("First dets:")
        print(dets[:10])

    draw_dets(img0, dets, save_path)

    print("Saved:", save_path)


if __name__ == "__main__":
    main()