import os
import cv2
import glob
import argparse
import numpy as np
import jittor as jt

from jdet.models.networks.yolov8_obb import YOLOv8OBB
from jdet.ops.yolo_obb_ops import postprocess_obb_np, xywhr2poly_np

'''
RGB:
PYTHONPATH=/root/JDet/jittor-yolov8/python \
python tools/vis_m2d_lif_obb_pred.py \
  --weights /root/JDet/work_dirs/DroneVehicle_M2D-LIF/checkpoints/best.pkl \
  --source /root/JDet/5000-DroneVehice/images/val \
  --view-modal rgb \
  --out-dir /root/JDet/jittor-yolov8/runs/vis_rgb

IR:
PYTHONPATH=/root/JDet/jittor-yolov8/python \
python tools/vis_m2d_lif_obb_pred.py \
  --weights /root/JDet/work_dirs/DroneVehicle_M2D-LIF/checkpoints/best.pkl \
  --source /root/JDet/5000-DroneVehice/images/val \
  --view-modal ir \
  --out-dir /root/JDet/jittor-yolov8/runs/vis_ir

RGB + IR:
PYTHONPATH=/root/JDet/jittor-yolov8/python \
python tools/vis_m2d_lif_obb_pred.py \
  --weights /root/JDet/work_dirs/DroneVehicle_M2D-LIF/checkpoints/best.pkl \
  --source /root/JDet/test/images \
  --view-modal concat \
  --out-dir /root/JDet/jittor-yolov8/runs/vis_concat
'''

CLASS_COLORS = {
    0: (0, 205, 0),      # car - green
    1: (0, 205, 205),      # truck - blue
    2: (205, 51, 51),      # bus - red
    3: (205, 149, 12),    # van - yellow
    4: (139, 0, 139),    # freight_car - magenta
}

IMG_FORMATS = [".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"]


DEFAULT_NAMES = {
    0: "car",
    1: "truck",
    2: "bus",
    3: "van",
    4: "freight_car",
}


def scan_images(source):
    """source can be a folder, an image file, or a txt file containing image paths."""
    source = str(source)

    if os.path.isfile(source) and source.lower().endswith(".txt"):
        with open(source, "r") as f:
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


def infer_ir_path(rgb_path, ir_root=None):
    """
    Default:
        /root/JDet/DV-full/images/val/00001.jpg
    ->  /root/JDet/DV-full/images_ir/val/00001.jpg

    If ir_root is provided:
        use same filename under ir_root.
    """
    rgb_path = str(rgb_path)

    if ir_root is not None:
        return os.path.join(ir_root, os.path.basename(rgb_path))

    return rgb_path.replace(os.sep + "images" + os.sep, os.sep + "images_ir" + os.sep)


def letterbox_image(img, new_shape=640, color=(114, 114, 114), scaleup=False):
    """
    Letterbox for 6-channel paired RGB-IR image.
    Input:
        img: H x W x 6, BGR_RGB + BGR_IR
    Return:
        img_lb: new_shape x new_shape x 6
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
        rgb = cv2.copyMakeBorder(
            img[:, :, :3],
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(pad_value, pad_value, pad_value),
        )
        ir = cv2.copyMakeBorder(
            img[:, :, 3:6],
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(pad_value, pad_value, pad_value),
        )
        img = np.concatenate([rgb, ir], axis=2)
    else:
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


def scale_poly_from_letterbox_to_original(poly, orig_shape, ratio, pad):
    """
    poly: [N, 8] in letterbox image pixel coordinates
    orig_shape: original image shape [H, W]
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


def preprocess_pair(rgb_bgr, ir_bgr, imgsz=640, scaleup=False):
    """
    Return:
        x: jt.Var [1, 6, imgsz, imgsz]
        ratio
        pad
    """
    if ir_bgr.shape[:2] != rgb_bgr.shape[:2]:
        ir_bgr = cv2.resize(ir_bgr, (rgb_bgr.shape[1], rgb_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

    img6 = np.concatenate([rgb_bgr, ir_bgr], axis=2)  # BGR + BGR
    img6, ratio, pad = letterbox_image(img6, imgsz, scaleup=scaleup)

    # BGR+BGR -> RGB+RGB, HWC -> CHW
    img6 = np.concatenate(
        [
            img6[:, :, :3][:, :, ::-1],
            img6[:, :, 3:6][:, :, ::-1],
        ],
        axis=2,
    )

    img6 = img6.transpose(2, 0, 1)
    img6 = np.ascontiguousarray(img6, dtype=np.float32) / 255.0
    x = jt.array(img6[None, ...])

    return x, ratio, pad


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

        return ckpt

    if hasattr(ckpt, "state_dict"):
        return ckpt.state_dict()

    raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")


def safe_load_model(model, ckpt_path):
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
        raise RuntimeError("No parameters matched. Check model cfg/scale/nc/ch.")

    return model


def draw_obb(
    image,
    polys,
    scores,
    clses,
    names=DEFAULT_NAMES,
    score_thr=0.25,
    class_colors=CLASS_COLORS,
):
    out = image.copy()
    h, w = out.shape[:2]

    # # 自适应线宽和字号，避免小图/大图效果差异太大
    # line_thickness = max(2, int(round(min(h, w) / 300)))
    # font_scale = max(0.6, min(h, w) / 900.0)
    # font_thickness = max(1, line_thickness - 1)

    # 更克制的显示参数
    line_thickness = max(2, int(round(min(h, w) / 500)))
    font_scale = max(0.45, min(h, w) / 1600.0)
    font_thickness = max(1, line_thickness - 1)

    for poly, score, cls_id in zip(polys, scores, clses):
        if score < score_thr:
            continue

        cls_id = int(cls_id)
        color = class_colors.get(cls_id, (0, 255, 0))
        pts = poly.reshape(4, 2).astype(np.int32)

        # 画旋转框，使用抗锯齿
        cv2.polylines(
            out,
            [pts],
            isClosed=True,
            color=color,
            thickness=line_thickness,
            lineType=cv2.LINE_AA,
        )

        # 取文本内容
        name = names.get(cls_id, f"cls_{cls_id}")
        text = f"{name} {score:.2f}"

        # 文本放在框上方
        x = int(pts[:, 0].min())
        y = int(pts[:, 1].min()) - 4
        y = max(y, 0)

        # 计算文本框大小
        (tw, th), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_thickness,
        )

        # 文字背景框，提升可读性
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

        # 白字，更清楚
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

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Distilled Jittor checkpoint pkl, e.g. work_dirs/.../checkpoints/best.pkl",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="RGB val folder, image file, or txt file. IR path is inferred from RGB path.",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="/root/JDet/jittor-yolov8/projects/yolov8_obb/configs/yolo_configs/yolov8n_LIF_obb.yaml",
        help="Student LIF YOLOv8-OBB yaml.",
    )
    parser.add_argument("--ir-root", type=str, default=None, help="Optional IR image folder.")
    parser.add_argument("--out-dir", type=str, default="runs/vis_m2d_lif_obb")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--scale", type=str, default="n")
    parser.add_argument("--nc", type=int, default=5)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--limit", type=int, default=-1, help="Max images to visualize. -1 means all.")
    parser.add_argument("--use-cuda", type=int, default=1)
    parser.add_argument("--view-modal", type=str, default="rgb", choices=["rgb", "ir", "concat"], help="Visualization background: rgb, ir, or concat.",
)

    args = parser.parse_args()

    jt.flags.use_cuda = int(args.use_cuda)

    os.makedirs(args.out_dir, exist_ok=True)

    # Inference only needs the student LIF model.
    # We instantiate YOLOv8OBB with LIF yaml and 6 input channels.
    model = YOLOv8OBB(
        cfg=args.cfg,
        ch=6,
        nc=args.nc,
        imgsz=args.imgsz,
        scale=args.scale,
    )
    model = safe_load_model(model, args.weights)
    model.eval()

    image_files = scan_images(args.source)
    if args.limit > 0:
        image_files = image_files[: args.limit]

    print(f"[Data] Found {len(image_files)} RGB images.")

    for idx, rgb_path in enumerate(image_files):
        rgb = cv2.imread(rgb_path)
        if rgb is None:
            print(f"[WARN] RGB image not found: {rgb_path}")
            continue

        ir_path = infer_ir_path(rgb_path, args.ir_root)
        ir = cv2.imread(ir_path)
        if ir is None:
            print(f"[WARN] IR image not found: {ir_path}")
            continue

        x, ratio, pad = preprocess_pair(
            rgb,
            ir,
            imgsz=args.imgsz,
            scaleup=False,
        )

        with jt.no_grad():
            pred = model(x)

        jt.sync_all()

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
            if args.view_modal == "rgb":
                vis = rgb.copy()
            elif args.view_modal == "ir":
                vis = ir.copy()
            elif args.view_modal == "concat":
                vis = np.concatenate([rgb, ir], axis=1)
            else:
                vis = rgb.copy()
        else:
            polys_lb = xywhr2poly_np(dets[:, :5])
            polys_orig = scale_poly_from_letterbox_to_original(
                polys_lb,
                rgb.shape[:2],
                ratio,
                pad,
            )
            scores = dets[:, 5].astype(np.float32)
            clses = dets[:, 6].astype(np.int32)

            if args.view_modal == "rgb":
                vis = draw_obb(
                    rgb,
                    polys_orig,
                    scores,
                    clses,
                    names=DEFAULT_NAMES,
                    score_thr=args.conf,
                )

            elif args.view_modal == "ir":
                vis = draw_obb(
                    ir,
                    polys_orig,
                    scores,
                    clses,
                    names=DEFAULT_NAMES,
                    score_thr=args.conf,
                )

            elif args.view_modal == "concat":
                # 左边 RGB，右边 IR，两边都画同一组检测框
                rgb_vis = draw_obb(
                    rgb,
                    polys_orig,
                    scores,
                    clses,
                    names=DEFAULT_NAMES,
                    score_thr=args.conf,
                )

                ir_polys = polys_orig.copy().reshape(-1, 4, 2)
                ir_polys[:, :, 0] += rgb.shape[1]
                ir_polys = ir_polys.reshape(-1, 8)

                concat_img = np.concatenate([rgb, ir], axis=1)

                vis = draw_obb(
                    concat_img,
                    np.concatenate([polys_orig, ir_polys], axis=0),
                    np.concatenate([scores, scores], axis=0),
                    np.concatenate([clses, clses], axis=0),
                    names=DEFAULT_NAMES,
                    score_thr=args.conf,
                )

            else:
                vis = draw_obb(
                    rgb,
                    polys_orig,
                    scores,
                    clses,
                    names=DEFAULT_NAMES,
                    score_thr=args.conf,
                )

        save_name = os.path.splitext(os.path.basename(rgb_path))[0] + ".png"
        save_path = os.path.join(args.out_dir, save_name)
        ext = os.path.splitext(save_path)[1].lower()

        if ext in [".jpg", ".jpeg"]:
            cv2.imwrite(save_path, vis, [cv2.IMWRITE_JPEG_QUALITY, 100])
        elif ext == ".png":
            cv2.imwrite(save_path, vis, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        else:
            cv2.imwrite(save_path, vis)

        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"[{idx + 1}/{len(image_files)}] saved: {save_path}")

    print(f"[Done] Results saved to: {args.out_dir}")


if __name__ == "__main__":
    main()