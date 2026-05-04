# -*- coding: utf-8 -*-
"""
Jittor YOLOv8-OBB single-modal prediction visualization.

This script follows the drawing style of the Jittor M2D-LIF visualization script:
    - fixed class colors
    - anti-aliased OBB polygons
    - readable colored label background
    - PNG output with no compression by default

It is used for the single-modal Jittor YOLOv8-OBB model in Jittor-yolov8-OBB.zip.

Example: RGB teacher


PYTHONPATH=/root/JDet/python \
python tools/vis_single_yolov8_obb_pred.py \
  --weights /root/JDet/work_dirs/RGB-full.pkl \
  --source /root/JDet/5000-DroneVehice/images/val \
  --cfg /root/JDet/projects/yolov8_obb/configs/yolo_configs/yolov8n_obb.yaml \
  --out-dir /root/JDet/runs/vis_single_rgb \
  --imgsz 640 \
  --scale n \
  --nc 5 \
  --conf 0.25 \
  --iou 0.7 \
  --use-cuda 1 \
  --names "car,truck,bus,van,freight_car"


Example: with GT and compare images

cd /root/JDet/jittor-yolov8

PYTHONPATH=/root/JDet/jittor-yolov8/python \
python tools/vis_single_yolov8_obb_pred.py \
  --weights /root/JDet/work_dirs/yolov8n_obb_dronevehicle_rgb_teacher/checkpoints/best.pkl \
  --source /root/JDet/5000-DroneVehice/images/val \
  --label-dir /root/JDet/5000-DroneVehice/labels/val \
  --cfg /root/JDet/jittor-yolov8/projects/yolov8_obb/configs/yolo_configs/yolov8n_obb.yaml \
  --out-dir /root/JDet/jittor-yolov8/runs/vis_single_rgb_with_gt \
  --imgsz 640 \
  --scale n \
  --nc 5 \
  --conf 0.25 \
  --iou 0.7 \
  --use-cuda 1 \
  --names "car,truck,bus,van,freight_car"
"""

import os
import sys
import cv2
import glob
import argparse
import numpy as np
import jittor as jt

# Make it runnable both from repo root and from tools/.
ROOT = os.getcwd()
if os.path.isdir(os.path.join(ROOT, "python")) and os.path.join(ROOT, "python") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "python"))

from jdet.models.networks.yolov8_obb import YOLOv8OBB
from jdet.ops.yolo_obb_ops import postprocess_obb_np, xywhr2poly_np


CLASS_COLORS = {
    0: (0, 205, 0),       # car - green
    1: (0, 205, 205),     # truck - cyan/yellow-like in BGR
    2: (205, 51, 51),     # bus - blue/red-like in BGR
    3: (205, 149, 12),    # van
    4: (139, 0, 139),     # freight_car - magenta
}

DEFAULT_NAMES = {
    0: "car",
    1: "truck",
    2: "bus",
    3: "van",
    4: "freight_car",
}

IMG_FORMATS = [".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"]


def parse_names(names):
    if not names:
        return DEFAULT_NAMES
    items = [x.strip() for x in names.split(",") if x.strip()]
    return {i: name for i, name in enumerate(items)}


def scan_images(source):
    """source can be a folder, an image file, or a txt file containing image paths."""
    source = str(source)

    if os.path.isfile(source) and source.lower().endswith(".txt"):
        with open(source, "r", encoding="utf-8") as f:
            files = [x.strip() for x in f.readlines() if x.strip()]
        return files

    if os.path.isfile(source):
        return [source]

    if os.path.isdir(source):
        files = []
        for fmt in IMG_FORMATS:
            files.extend(glob.glob(os.path.join(source, "**", "*" + fmt), recursive=True))
            files.extend(glob.glob(os.path.join(source, "**", "*" + fmt.upper()), recursive=True))
        return sorted(files)

    raise FileNotFoundError(source)


def letterbox_image(img, new_shape=640, color=(114, 114, 114), scaleup=False):
    """
    Standard YOLO letterbox for single-modal 3-channel image.

    Return:
        img_lb: letterboxed image
        ratio: resize ratio
        pad: (left, top)
    """
    h0, w0 = img.shape[:2]

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(float(new_shape[0]) / h0, float(new_shape[1]) / w0)
    if not scaleup:
        r = min(r, 1.0)

    new_unpad_w = int(round(w0 * r))
    new_unpad_h = int(round(h0 * r))

    dw = (new_shape[1] - new_unpad_w) / 2.0
    dh = (new_shape[0] - new_unpad_h) / 2.0

    if (w0, h0) != (new_unpad_w, new_unpad_h):
        img = cv2.resize(img, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)

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

    return img, r, (left, top)


def preprocess_single(img_bgr, imgsz=640, scaleup=False):
    """
    BGR image -> Jittor tensor [1, 3, imgsz, imgsz].
    """
    img_lb, ratio, pad = letterbox_image(img_bgr, imgsz, scaleup=scaleup)

    # BGR -> RGB, HWC -> CHW
    img_rgb = img_lb[:, :, ::-1].transpose(2, 0, 1)
    img_rgb = np.ascontiguousarray(img_rgb, dtype=np.float32) / 255.0
    x = jt.array(img_rgb[None, ...])

    return x, ratio, pad


def scale_poly_from_letterbox_to_original(poly, orig_shape, ratio, pad):
    """
    poly: [N, 8] in letterbox pixel coordinates.
    Return [N, 8] in original image pixel coordinates.
    """
    if poly is None or len(poly) == 0:
        return np.zeros((0, 8), dtype=np.float32)

    orig_h, orig_w = orig_shape[:2]
    left, top = pad

    poly = poly.reshape(-1, 4, 2).astype(np.float32).copy()
    poly[:, :, 0] = (poly[:, :, 0] - float(left)) / max(float(ratio), 1e-9)
    poly[:, :, 1] = (poly[:, :, 1] - float(top)) / max(float(ratio), 1e-9)

    poly[:, :, 0] = np.clip(poly[:, :, 0], 0, orig_w - 1)
    poly[:, :, 1] = np.clip(poly[:, :, 1], 0, orig_h - 1)

    return poly.reshape(-1, 8).astype(np.float32)


def extract_state_dict(ckpt):
    """
    Compatible with:
        1. pure state_dict
        2. runner checkpoint: {"model": ..., "ema": ...}
        3. saved Module / Sequential
    """
    if isinstance(ckpt, dict):
        if "ema" in ckpt and ckpt["ema"] is not None:
            ema = ckpt["ema"]
            if isinstance(ema, dict):
                return ema
            if hasattr(ema, "state_dict"):
                return ema.state_dict()

        if "model" in ckpt and ckpt["model"] is not None:
            model = ckpt["model"]
            if isinstance(model, dict):
                return model
            if hasattr(model, "state_dict"):
                return model.state_dict()

        if "state_dict" in ckpt and ckpt["state_dict"] is not None:
            return ckpt["state_dict"]

        return ckpt

    if hasattr(ckpt, "state_dict"):
        return ckpt.state_dict()

    raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")


def safe_load_model(model, ckpt_path, strict_shape=True):
    ckpt = jt.load(ckpt_path)
    state = extract_state_dict(ckpt)
    cur = model.state_dict()

    matched = {}
    skipped = []

    for k, v in state.items():
        if k in cur and hasattr(v, "shape") and tuple(v.shape) == tuple(cur[k].shape):
            matched[k] = v
        else:
            if k not in cur:
                reason = "missing_key"
            elif not hasattr(v, "shape"):
                reason = f"not_tensor type={type(v)}"
            else:
                reason = f"shape_mismatch ckpt={tuple(v.shape)} model={tuple(cur[k].shape)}"
            skipped.append((k, reason))

    model.load_parameters(matched)

    print(f"[Load] {ckpt_path}")
    print(f"[Load] matched {len(matched)}/{len(cur)} params")
    if skipped:
        print(f"[Load] skipped {len(skipped)} params. First 10:")
        for item in skipped[:10]:
            print("   ", item)

    if len(matched) == 0:
        raise RuntimeError("No parameters matched. Check cfg/scale/nc/ch/weight path.")

    if strict_shape and len(skipped) > 0:
        print("[Warn] Some parameters were skipped. If the result is abnormal, check scale/nc/cfg.")

    return model


def read_yolo_obb_labels(label_path, img_shape):
    """
    Read YOLO-OBB labels:
        cls x1 y1 x2 y2 x3 y3 x4 y4
    Coordinates are normalized to [0, 1].
    """
    if not label_path or not os.path.exists(label_path):
        return (
            np.zeros((0, 8), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )

    h, w = img_shape[:2]
    polys = []
    clses = []

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f.readlines() if x.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) < 9:
            continue
        cls_id = int(float(parts[0]))
        pts = np.array([float(x) for x in parts[1:9]], dtype=np.float32).reshape(4, 2)
        pts[:, 0] *= w
        pts[:, 1] *= h
        polys.append(pts.reshape(8))
        clses.append(cls_id)

    if len(polys) == 0:
        return (
            np.zeros((0, 8), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )

    return np.asarray(polys, dtype=np.float32), np.asarray(clses, dtype=np.int32)


def infer_label_path(img_path, label_dir):
    if not label_dir:
        return None
    return os.path.join(label_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt")


def draw_obb(
    image,
    polys,
    scores,
    clses,
    names=DEFAULT_NAMES,
    score_thr=0.25,
    class_colors=CLASS_COLORS,
    prefix="",
):
    out = image.copy()
    h, w = out.shape[:2]

    # Same restrained style as the M2D-LIF visualization script.
    line_thickness = max(2, int(round(min(h, w) / 500)))
    font_scale = max(0.45, min(h, w) / 1600.0)
    font_thickness = max(1, line_thickness - 1)

    for poly, score, cls_id in zip(polys, scores, clses):
        if score < score_thr:
            continue

        cls_id = int(cls_id)
        color = class_colors.get(cls_id, (0, 255, 0))
        pts = poly.reshape(4, 2).astype(np.int32)

        cv2.polylines(
            out,
            [pts],
            isClosed=True,
            color=color,
            thickness=line_thickness,
            lineType=cv2.LINE_AA,
        )

        name = names.get(cls_id, f"cls_{cls_id}")
        if prefix:
            text = f"{prefix} {name}"
        else:
            text = f"{name} {score:.2f}"

        x = int(pts[:, 0].min())
        y = int(pts[:, 1].min()) - 4
        y = max(y, 0)

        (tw, th), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_thickness,
        )

        box_x1 = x
        box_y1 = max(y - th - baseline - 4, 0)
        box_x2 = min(x + tw + 6, w - 1)
        box_y2 = min(y + 2, h - 1)

        cv2.rectangle(
            out,
            (box_x1, box_y1),
            (box_x2, box_y2),
            color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

        cv2.putText(
            out,
            text,
            (box_x1 + 3, box_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )

    return out


def make_compare(gt_img, pred_img):
    h = max(gt_img.shape[0], pred_img.shape[0])
    w1, w2 = gt_img.shape[1], pred_img.shape[1]

    canvas = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    canvas[: gt_img.shape[0], :w1] = gt_img
    canvas[: pred_img.shape[0], w1:w1 + w2] = pred_img

    cv2.putText(canvas, "Ground Truth", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Prediction", (w1 + 20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

    return canvas


def save_image(path, image):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, 100])
    elif ext == ".png":
        cv2.imwrite(path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    else:
        cv2.imwrite(path, image)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--weights", type=str, required=True, help="Jittor checkpoint pkl")
    parser.add_argument("--source", type=str, required=True, help="Image folder, image file, or txt file")
    parser.add_argument(
        "--cfg",
        type=str,
        default="projects/yolov8_obb/configs/yolo_configs/yolov8n_obb.yaml",
        help="Single-modal YOLOv8-OBB yaml",
    )
    parser.add_argument("--out-dir", type=str, default="runs/vis_single_yolov8_obb")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--scale", type=str, default="n", help="n/s/m/l/x if YAML has scales")
    parser.add_argument("--nc", type=int, default=5)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--limit", type=int, default=-1, help="Max images to visualize. -1 means all.")
    parser.add_argument("--use-cuda", type=int, default=1)
    parser.add_argument("--scaleup", action="store_true", help="Allow up-sampling during letterbox. Default is False.")
    parser.add_argument("--names", type=str, default=None, help='Class names, e.g. "car,truck,bus,van,freight_car"')
    parser.add_argument("--label-dir", type=str, default=None, help="Optional YOLO-OBB label dir for GT/compare visualization")
    parser.add_argument("--save-empty", action="store_true", help="Save images even if there is no prediction. Default: save all images anyway.")

    args = parser.parse_args()
    jt.flags.use_cuda = int(args.use_cuda)

    names = parse_names(args.names)

    # Output layout:
    #   no label-dir: out-dir/*.png
    #   with label-dir: out-dir/pred, out-dir/gt, out-dir/compare
    os.makedirs(args.out_dir, exist_ok=True)
    pred_dir = os.path.join(args.out_dir, "pred") if args.label_dir else args.out_dir
    gt_dir = os.path.join(args.out_dir, "gt") if args.label_dir else None
    cmp_dir = os.path.join(args.out_dir, "compare") if args.label_dir else None
    os.makedirs(pred_dir, exist_ok=True)
    if gt_dir:
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(cmp_dir, exist_ok=True)

    model = YOLOv8OBB(
        cfg=args.cfg,
        ch=3,
        nc=args.nc,
        imgsz=args.imgsz,
        scale=args.scale,
    )
    model = safe_load_model(model, args.weights)
    model.eval()

    image_files = scan_images(args.source)
    if args.limit > 0:
        image_files = image_files[: args.limit]

    print(f"[Data] Found {len(image_files)} images.")
    print(f"[Model] cfg={args.cfg}, ch=3, nc={args.nc}, scale={args.scale}")

    for idx, img_path in enumerate(image_files):
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] image not found or unreadable: {img_path}")
            continue

        x, ratio, pad = preprocess_single(
            img,
            imgsz=args.imgsz,
            scaleup=args.scaleup,
        )

        with jt.no_grad():
            pred = model(x)

        jt.sync_all()

        if isinstance(pred, (list, tuple)):
            pred = pred[0]

        pred_np = np.asarray(pred.numpy(), dtype=np.float32)

        # pred_np: [B, N, 5 + nc], take B=0
        if pred_np.ndim == 3:
            pred_i = pred_np[0]
        else:
            pred_i = pred_np

        dets = postprocess_obb_np(
            pred_i,
            conf_thres=args.conf,
            iou_thres=args.iou,
            max_det=args.max_det,
        )

        if dets is None or len(dets) == 0:
            pred_vis = img.copy()
            num_det = 0
        else:
            polys_lb = xywhr2poly_np(dets[:, :5])
            polys_orig = scale_poly_from_letterbox_to_original(
                polys_lb,
                img.shape[:2],
                ratio,
                pad,
            )
            scores = dets[:, 5].astype(np.float32)
            clses = dets[:, 6].astype(np.int32)
            num_det = len(polys_orig)

            pred_vis = draw_obb(
                img,
                polys_orig,
                scores,
                clses,
                names=names,
                score_thr=args.conf,
            )

        save_name = os.path.splitext(os.path.basename(img_path))[0] + ".png"
        save_image(os.path.join(pred_dir, save_name), pred_vis)

        if args.label_dir:
            label_path = infer_label_path(img_path, args.label_dir)
            gt_polys, gt_clses = read_yolo_obb_labels(label_path, img.shape[:2])
            gt_scores = np.ones((len(gt_clses),), dtype=np.float32)

            gt_vis = draw_obb(
                img,
                gt_polys,
                gt_scores,
                gt_clses,
                names=names,
                score_thr=0.0,
                prefix="GT",
            )
            cmp_vis = make_compare(gt_vis, pred_vis)

            save_image(os.path.join(gt_dir, save_name), gt_vis)
            save_image(os.path.join(cmp_dir, save_name), cmp_vis)

        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"[{idx + 1}/{len(image_files)}] {os.path.basename(img_path)} | pred={num_det} | saved={os.path.join(pred_dir, save_name)}")

    print(f"[Done] Results saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
