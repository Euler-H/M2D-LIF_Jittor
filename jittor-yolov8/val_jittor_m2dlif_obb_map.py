# -*- coding: utf-8 -*-
"""
Validate mAP for dual-modal Jittor M2D-LIF YOLOv8-OBB.

Expected RGB/IR file pairing:
    --source /path/images/val
    --ir-root /path/images_ir/val     # optional, otherwise images -> images_ir is inferred

Expected label format per line:
    cls x1 y1 x2 y2 x3 y3 x4 y4
where polygon coordinates are normalized to [0, 1].

Example:
PYTHONPATH=/root/JDet/jittor-yolov8/python \
python /root/JDet/jittor-yolov8/val_jittor_m2dlif_obb_map.py \
  --weights /root/JDet/work_dirs/DroneVehicle_M2D-LIF/checkpoints/best.pkl \
  --source /root/JDet/test/images \
  --ir-root /root/JDet/test/images_ir \
  --label-dir /root/JDet/test/labels \
  --cfg /root/JDet/jittor-yolov8/projects/yolov8_obb/configs/yolo_configs/yolov8n_LIF_obb.yaml \
  --imgsz 640 --scale n --nc 5 --conf 0.25 --iou 0.7 --use-cuda 1
"""

import os
import cv2
import glob
import csv
import json
import argparse
import numpy as np
import jittor as jt

from jdet.models.networks.yolov8_obb import YOLOv8OBB
from jdet.ops.yolo_obb_ops import (
    postprocess_obb_np,
    xywhr2poly_np,
    poly2xywhr_np,
    obb_map_eval,
    obb_map_eval_probiou,
)

IMG_FORMATS = [".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"]
DEFAULT_NAMES = {0: "car", 1: "truck", 2: "bus", 3: "van", 4: "freight_car"}


def parse_names(names, nc):
    if names is None or names.strip() == "":
        return {i: DEFAULT_NAMES.get(i, f"cls_{i}") for i in range(nc)}
    parts = [x.strip() for x in names.split(",") if x.strip()]
    return {i: parts[i] if i < len(parts) else f"cls_{i}" for i in range(nc)}


def scan_images(source):
    source = str(source)
    if os.path.isfile(source) and source.lower().endswith(".txt"):
        with open(source, "r", encoding="utf-8") as f:
            return [x.strip() for x in f.readlines() if x.strip()]
    if os.path.isfile(source):
        return [source]
    if os.path.isdir(source):
        files = []
        for fmt in IMG_FORMATS:
            files.extend(glob.glob(os.path.join(source, "**", "*" + fmt), recursive=True))
            files.extend(glob.glob(os.path.join(source, "**", "*" + fmt.upper()), recursive=True))
        return sorted(files)
    raise FileNotFoundError(source)


def infer_ir_path(rgb_path, ir_root=None):
    rgb_path = str(rgb_path)
    if ir_root is not None:
        return os.path.join(ir_root, os.path.basename(rgb_path))
    return rgb_path.replace(os.sep + "images" + os.sep, os.sep + "images_ir" + os.sep)


def image_id_from_path(path):
    return os.path.splitext(os.path.basename(path))[0]


def label_path_from_image(img_path, label_dir):
    return os.path.join(label_dir, image_id_from_path(img_path) + ".txt")


def letterbox_image(img, new_shape=640, color=(114, 114, 114), scaleup=False):
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
        if img.ndim == 3 and img.shape[2] == 6:
            rgb = cv2.resize(img[:, :, :3], (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
            ir = cv2.resize(img[:, :, 3:6], (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
            img = np.concatenate([rgb, ir], axis=2)
        else:
            img = cv2.resize(img, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    pad_value = color[0] if isinstance(color, (tuple, list)) else color
    if img.ndim == 3 and img.shape[2] == 6:
        rgb = cv2.copyMakeBorder(img[:, :, :3], top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad_value, pad_value, pad_value))
        ir = cv2.copyMakeBorder(img[:, :, 3:6], top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad_value, pad_value, pad_value))
        img = np.concatenate([rgb, ir], axis=2)
    else:
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)


def preprocess_pair(rgb_bgr, ir_bgr, imgsz=640, scaleup=False, order="rgb_ir"):
    if ir_bgr.shape[:2] != rgb_bgr.shape[:2]:
        ir_bgr = cv2.resize(ir_bgr, (rgb_bgr.shape[1], rgb_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    if order == "rgb_ir":
        img6 = np.concatenate([rgb_bgr, ir_bgr], axis=2)
    elif order == "ir_rgb":
        img6 = np.concatenate([ir_bgr, rgb_bgr], axis=2)
    else:
        raise ValueError(f"Unsupported order: {order}")
    img6, ratio, pad = letterbox_image(img6, imgsz, scaleup=scaleup)
    img6 = np.concatenate([img6[:, :, :3][:, :, ::-1], img6[:, :, 3:6][:, :, ::-1]], axis=2)
    img6 = img6.transpose(2, 0, 1)
    img6 = np.ascontiguousarray(img6, dtype=np.float32) / 255.0
    x = jt.array(img6[None, ...])
    return x, ratio, pad


def scale_poly_from_letterbox_to_original(poly, orig_shape, ratio, pad):
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


def read_gt_labels(img_path, label_dir, img_shape, nc):
    h, w = img_shape[:2]
    image_id = image_id_from_path(img_path)
    label_path = label_path_from_image(img_path, label_dir)
    gts = []
    if not os.path.exists(label_path):
        return gts
    with open(label_path, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f.readlines() if x.strip()]
    for line in lines:
        parts = line.split()
        if len(parts) < 9:
            continue
        cls_id = int(float(parts[0]))
        if cls_id < 0 or cls_id >= nc:
            continue
        poly = np.array([float(x) for x in parts[1:9]], dtype=np.float32).reshape(4, 2)
        poly[:, 0] *= w
        poly[:, 1] *= h
        poly = poly.reshape(8).astype(np.float32)
        xywhr = poly2xywhr_np(poly.reshape(1, 8))[0].astype(np.float32)
        gts.append({"image_id": image_id, "cls": cls_id, "poly": poly, "xywhr": xywhr})
    return gts


def extract_state_dict(ckpt):
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
        return ckpt
    if hasattr(ckpt, "state_dict"):
        return ckpt.state_dict()
    raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")


def safe_load_model(model, ckpt_path):
    ckpt = jt.load(ckpt_path)
    state = extract_state_dict(ckpt)
    cur = model.state_dict()
    matched, skipped = {}, []
    for k, v in state.items():
        if k in cur and hasattr(v, "shape") and tuple(v.shape) == tuple(cur[k].shape):
            matched[k] = v
        else:
            skipped.append(k)
    model.load_parameters(matched)
    print(f"[Load] {ckpt_path}")
    print(f"[Load] matched {len(matched)}/{len(cur)} params, skipped {len(skipped)}")
    if len(matched) == 0:
        raise RuntimeError("No parameters matched. Check cfg/scale/nc/ch and checkpoint.")
    return model


def normalize_pred_output(pred):
    if isinstance(pred, (list, tuple)):
        for item in pred:
            if hasattr(item, "numpy"):
                pred = item
                break
    pred_np = np.asarray(pred.numpy(), dtype=np.float32) if hasattr(pred, "numpy") else np.asarray(pred, dtype=np.float32)
    if pred_np.ndim == 3:
        pred_np = pred_np[0]
    return pred_np


def append_predictions(predictions, dets, img_path, img_shape, ratio, pad):
    image_id = image_id_from_path(img_path)
    if dets is None or len(dets) == 0:
        return
    polys_lb = xywhr2poly_np(dets[:, :5])
    polys_orig = scale_poly_from_letterbox_to_original(polys_lb, img_shape, ratio, pad)
    xywhr_orig = poly2xywhr_np(polys_orig)
    scores = dets[:, 5].astype(np.float32)
    clses = dets[:, 6].astype(np.int32)
    for poly, xywhr, score, cls_id in zip(polys_orig, xywhr_orig, scores, clses):
        predictions.append({
            "image_id": image_id,
            "cls": int(cls_id),
            "score": float(score),
            "poly": poly.astype(np.float32),
            "xywhr": xywhr.astype(np.float32),
        })


def save_results(out_dir, names, poly_res, probiou_res, predictions, ground_truths):
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    summary = {
        "poly_map50": float(poly_res["map50"]),
        "poly_map": float(poly_res["map"]),
        "probiou_map50": float(probiou_res["map50"]),
        "probiou_map": float(probiou_res["map"]),
        "num_predictions": len(predictions),
        "num_ground_truths": len(ground_truths),
        "n_gt_per_cls": probiou_res["n_gt_per_cls"].astype(int).tolist(),
    }
    with open(os.path.join(out_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "per_class_ap50.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "gt_count", "poly_AP50", "probiou_AP50"])
        for cid in range(len(names)):
            writer.writerow([
                cid, names[cid], int(probiou_res["n_gt_per_cls"][cid]),
                float(poly_res["ap50"][cid]) if not np.isnan(poly_res["ap50"][cid]) else "nan",
                float(probiou_res["ap50"][cid]) if not np.isnan(probiou_res["ap50"][cid]) else "nan",
            ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--source", type=str, required=True, help="RGB image folder, image file, or txt file")
    parser.add_argument("--ir-root", type=str, default=None, help="IR image folder. If omitted, images -> images_ir is inferred")
    parser.add_argument("--label-dir", type=str, required=True, help="YOLO-OBB label folder")
    parser.add_argument("--cfg", type=str, default="/root/JDet/jittor-yolov8/projects/yolov8_obb/configs/yolo_configs/yolov8n_LIF_obb.yaml")
    parser.add_argument("--out-dir", type=str, default="runs/val_m2dlif_obb")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--scale", type=str, default="n")
    parser.add_argument("--nc", type=int, default=5)
    parser.add_argument("--conf", type=float, default=0.001, help="Use 0.001 for mAP evaluation, not 0.25")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--use-cuda", type=int, default=1)
    parser.add_argument("--order", type=str, default="rgb_ir", choices=["rgb_ir", "ir_rgb"], help="Channel order used by the LIF model")
    parser.add_argument("--names", type=str, default="car,truck,bus,van,freight_car")
    args = parser.parse_args()

    jt.flags.use_cuda = int(args.use_cuda)
    names = parse_names(args.names, args.nc)

    model = YOLOv8OBB(cfg=args.cfg, ch=6, nc=args.nc, imgsz=args.imgsz, scale=args.scale)
    model = safe_load_model(model, args.weights)
    model.eval()

    image_files = scan_images(args.source)
    if args.limit > 0:
        image_files = image_files[:args.limit]
    print(f"[Data] Found {len(image_files)} RGB images.")
    print(f"[Data] Modal order: {args.order}")

    predictions, ground_truths = [], []
    missing_ir = 0
    for idx, rgb_path in enumerate(image_files):
        rgb = cv2.imread(rgb_path)
        if rgb is None:
            print(f"[WARN] failed to read RGB image: {rgb_path}")
            continue
        ir_path = infer_ir_path(rgb_path, args.ir_root)
        ir = cv2.imread(ir_path)
        if ir is None:
            missing_ir += 1
            print(f"[WARN] failed to read IR image: {ir_path}")
            continue
        ground_truths.extend(read_gt_labels(rgb_path, args.label_dir, rgb.shape, args.nc))
        x, ratio, pad = preprocess_pair(rgb, ir, imgsz=args.imgsz, scaleup=False, order=args.order)
        with jt.no_grad():
            pred = model(x)
        jt.sync_all()
        pred_np = normalize_pred_output(pred)
        dets = postprocess_obb_np(pred_np, conf_thres=args.conf, iou_thres=args.iou, max_det=args.max_det)
        append_predictions(predictions, dets, rgb_path, rgb.shape, ratio, pad)
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"[{idx + 1}/{len(image_files)}] preds={len(predictions)} gts={len(ground_truths)} missing_ir={missing_ir}")

    print("[Eval] computing polygon IoU mAP...")
    poly_res = obb_map_eval(predictions, ground_truths, num_classes=args.nc)
    print("[Eval] computing ProbIoU mAP...")
    probiou_res = obb_map_eval_probiou(predictions, ground_truths, num_classes=args.nc)

    print("\n================ Jittor M2D-LIF YOLOv8-OBB Validation ================")
    print(f"Images: {len(image_files)} | Missing IR: {missing_ir} | Predictions: {len(predictions)} | GTs: {len(ground_truths)}")
    print(f"ProbIoU mAP50     : {probiou_res['map50']:.6f}")
    print(f"ProbIoU mAP50-95  : {probiou_res['map']:.6f}")
    print(f"Polygon mAP50     : {poly_res['map50']:.6f}")
    print(f"Polygon mAP50-95  : {poly_res['map']:.6f}")
    print("Per-class AP50 by ProbIoU:")
    for cid in range(args.nc):
        ap50 = probiou_res["ap50"][cid]
        ap50_text = "nan" if np.isnan(ap50) else f"{float(ap50):.6f}"
        print(f"  {cid:2d} {names[cid]:12s} GT={int(probiou_res['n_gt_per_cls'][cid]):6d} AP50={ap50_text}")
    save_results(args.out_dir, names, poly_res, probiou_res, predictions, ground_truths)
    print(f"[Done] Saved metrics to: {args.out_dir}")


if __name__ == "__main__":
    main()
