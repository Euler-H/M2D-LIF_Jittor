import math
import cv2
import numpy as np
import jittor as jt


def poly2xywhr_np(polys):
    """
    polygon 四点框转 xywhr。

    输入:
        polys: numpy array, shape [N, 8]
               x1 y1 x2 y2 x3 y3 x4 y4
               坐标要求已经是归一化坐标或像素坐标均可

    输出:
        xywhr: numpy array, shape [N, 5]
               cx cy w h theta

    说明:
        theta 使用弧度。
        这里先用 OpenCV minAreaRect 作为稳定实现。
    """

    if polys is None or len(polys) == 0:
        return np.zeros((0, 5), dtype=np.float32)

    polys = np.asarray(polys, dtype=np.float32).reshape(-1, 4, 2)
    results = []

    for p in polys:
        rect = cv2.minAreaRect(p)
        (cx, cy), (w, h), angle = rect

        # OpenCV angle 通常在 [-90, 0) 度附近
        # 统一成 w >= h 的形式，减少角度歧义
        if w < h:
            w, h = h, w
            angle += 90.0

        theta = angle / 180.0 * math.pi

        # 将 theta 粗略规整到 [-pi/2, pi/2)
        while theta >= math.pi / 2:
            theta -= math.pi
        while theta < -math.pi / 2:
            theta += math.pi

        results.append([cx, cy, w, h, theta])

    return np.asarray(results, dtype=np.float32)


def labels_poly_to_xywhr(labels):
    """
    labels:
        jt.Var or numpy array, [M, 10]
        batch_id, cls, x1, y1, ..., x4, y4

    return:
        numpy array [M, 7]
        batch_id, cls, cx, cy, w, h, theta
    """

    if isinstance(labels, jt.Var):
        labels_np = labels.numpy()
    else:
        labels_np = labels

    if labels_np is None or len(labels_np) == 0:
        return np.zeros((0, 7), dtype=np.float32)

    batch_cls = labels_np[:, :2]
    polys = labels_np[:, 2:10]
    xywhr = poly2xywhr_np(polys)

    return np.concatenate([batch_cls, xywhr], axis=1).astype(np.float32)


def xywhr_to_xyxy_np(xywhr):
    """
    临时用于第一版 loss：
    将旋转框近似转水平外接框 xyxy。

    输入:
        [N, 5] cx cy w h theta

    输出:
        [N, 4] x1 y1 x2 y2

    注意:
        这是训练初版的近似方案，后续需要替换为 rotated IoU。
    """

    if xywhr is None or len(xywhr) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    cx = xywhr[:, 0]
    cy = xywhr[:, 1]
    w = xywhr[:, 2]
    h = xywhr[:, 3]

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)


def xywhr2poly_np(xywhr):
    """
    xywhr -> polygon

    Args:
        xywhr: [N, 5], cx, cy, w, h, theta
               坐标为像素坐标或归一化坐标均可，但必须前后一致。

    Returns:
        polys: [N, 8], x1 y1 x2 y2 x3 y3 x4 y4
    """
    if xywhr is None or len(xywhr) == 0:
        return np.zeros((0, 8), dtype=np.float32)

    xywhr = np.asarray(xywhr, dtype=np.float32).reshape(-1, 5)
    polys = []

    for cx, cy, w, h, theta in xywhr:
        cos_t = math.cos(float(theta))
        sin_t = math.sin(float(theta))

        dx = float(w) / 2.0
        dy = float(h) / 2.0

        pts = np.array(
            [
                [-dx, -dy],
                [ dx, -dy],
                [ dx,  dy],
                [-dx,  dy],
            ],
            dtype=np.float32,
        )

        rot = np.array(
            [
                [cos_t, -sin_t],
                [sin_t,  cos_t],
            ],
            dtype=np.float32,
        )

        pts = pts @ rot.T
        pts[:, 0] += float(cx)
        pts[:, 1] += float(cy)

        polys.append(pts.reshape(-1))

    return np.asarray(polys, dtype=np.float32)


def rotated_iou_cv2(box1, box2):
    """
    使用 OpenCV 计算两个旋转框 IoU。

    Args:
        box1, box2: [5], cx, cy, w, h, theta
                    theta 为弧度，cx/cy/w/h 为像素坐标或同一尺度坐标。

    Returns:
        iou: float
    """
    cx1, cy1, w1, h1, t1 = box1
    cx2, cy2, w2, h2, t2 = box2

    w1 = max(float(w1), 1e-6)
    h1 = max(float(h1), 1e-6)
    w2 = max(float(w2), 1e-6)
    h2 = max(float(h2), 1e-6)

    rect1 = (
        (float(cx1), float(cy1)),
        (w1, h1),
        float(t1) * 180.0 / math.pi,
    )
    rect2 = (
        (float(cx2), float(cy2)),
        (w2, h2),
        float(t2) * 180.0 / math.pi,
    )

    area1 = w1 * h1
    area2 = w2 * h2

    inter_type, inter_pts = cv2.rotatedRectangleIntersection(rect1, rect2)

    if inter_pts is None:
        inter_area = 0.0
    else:
        inter_area = abs(cv2.contourArea(inter_pts))

    union = area1 + area2 - inter_area

    if union <= 0:
        return 0.0

    return float(inter_area / union)


def probiou_np_matrix_for_nms(boxes, eps=1e-7):
    boxes = np.asarray(boxes, dtype=np.float32)

    x = boxes[:, 0]
    y = boxes[:, 1]
    w = np.maximum(boxes[:, 2], 1e-6)
    h = np.maximum(boxes[:, 3], 1e-6)
    theta = boxes[:, 4]

    cos = np.cos(theta)
    sin = np.sin(theta)

    w2 = (w ** 2) / 12.0
    h2 = (h ** 2) / 12.0

    a = w2 * cos ** 2 + h2 * sin ** 2
    b = w2 * sin ** 2 + h2 * cos ** 2
    c = (w2 - h2) * sin * cos

    a1 = a[:, None]
    b1 = b[:, None]
    c1 = c[:, None]
    x1 = x[:, None]
    y1 = y[:, None]

    a2 = a[None, :]
    b2 = b[None, :]
    c2 = c[None, :]
    x2 = x[None, :]
    y2 = y[None, :]

    aa = a1 + a2
    bb = b1 + b2
    cc = c1 + c2

    denominator = aa * bb - cc ** 2 + eps

    dx = x1 - x2
    dy = y1 - y2

    t1 = 0.25 * (aa * dy ** 2 + bb * dx ** 2) / denominator
    t2 = 0.5 * cc * dx * dy / denominator

    det1 = np.maximum(a1 * b1 - c1 ** 2, eps)
    det2 = np.maximum(a2 * b2 - c2 ** 2, eps)
    det = np.maximum(denominator, eps)

    t3 = 0.5 * np.log(det / (4.0 * np.sqrt(det1 * det2) + eps) + eps)

    bd = np.clip(t1 + t2 + t3, eps, 100.0)
    hd = np.sqrt(1.0 - np.exp(-bd) + eps)
    probiou = 1.0 - hd

    return np.clip(probiou, 0.0, 1.0).astype(np.float32)


def probiou_np_matrix(boxes1, boxes2, eps=1e-7):
    """
    Pairwise ProbIoU between two groups of rotated boxes.

    boxes1: [N, 5], [cx, cy, w, h, theta]
    boxes2: [M, 5], [cx, cy, w, h, theta]
    return: [N, M]
    """
    boxes1 = np.asarray(boxes1, dtype=np.float32)
    boxes2 = np.asarray(boxes2, dtype=np.float32)

    if boxes1.ndim == 1:
        boxes1 = boxes1.reshape(1, 5)
    if boxes2.ndim == 1:
        boxes2 = boxes2.reshape(1, 5)

    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    x1 = boxes1[:, 0]
    y1 = boxes1[:, 1]
    w1 = np.maximum(boxes1[:, 2], 1e-6)
    h1 = np.maximum(boxes1[:, 3], 1e-6)
    r1 = boxes1[:, 4]

    x2 = boxes2[:, 0]
    y2 = boxes2[:, 1]
    w2 = np.maximum(boxes2[:, 2], 1e-6)
    h2 = np.maximum(boxes2[:, 3], 1e-6)
    r2 = boxes2[:, 4]

    cos1 = np.cos(r1)
    sin1 = np.sin(r1)
    cos2 = np.cos(r2)
    sin2 = np.sin(r2)

    w1_2 = (w1 ** 2) / 12.0
    h1_2 = (h1 ** 2) / 12.0
    w2_2 = (w2 ** 2) / 12.0
    h2_2 = (h2 ** 2) / 12.0

    a1 = w1_2 * cos1 ** 2 + h1_2 * sin1 ** 2
    b1 = w1_2 * sin1 ** 2 + h1_2 * cos1 ** 2
    c1 = (w1_2 - h1_2) * sin1 * cos1

    a2 = w2_2 * cos2 ** 2 + h2_2 * sin2 ** 2
    b2 = w2_2 * sin2 ** 2 + h2_2 * cos2 ** 2
    c2 = (w2_2 - h2_2) * sin2 * cos2

    a1 = a1[:, None]
    b1 = b1[:, None]
    c1 = c1[:, None]
    x1 = x1[:, None]
    y1 = y1[:, None]

    a2 = a2[None, :]
    b2 = b2[None, :]
    c2 = c2[None, :]
    x2 = x2[None, :]
    y2 = y2[None, :]

    aa = a1 + a2
    bb = b1 + b2
    cc = c1 + c2

    denominator = aa * bb - cc ** 2 + eps

    dx = x1 - x2
    dy = y1 - y2

    t1 = 0.25 * (aa * dy ** 2 + bb * dx ** 2) / denominator
    t2 = 0.5 * cc * dx * dy / denominator

    det1 = np.maximum(a1 * b1 - c1 ** 2, eps)
    det2 = np.maximum(a2 * b2 - c2 ** 2, eps)
    det = np.maximum(denominator, eps)

    t3 = 0.5 * np.log(det / (4.0 * np.sqrt(det1 * det2) + eps) + eps)

    bd = np.clip(t1 + t2 + t3, eps, 100.0)
    hd = np.sqrt(1.0 - np.exp(-bd) + eps)

    probiou = 1.0 - hd
    return np.clip(probiou, 0.0, 1.0).astype(np.float32)


def rotated_nms_np(boxes, scores, iou_thres=0.7, max_det=300, max_nms=3000):
    """
    Fast rotated NMS using ProbIoU.
    boxes: [N, 5], xywhr
    scores: [N]
    """
    if boxes is None or len(boxes) == 0:
        return []

    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    order = scores.argsort()[::-1]

    if len(order) > max_nms:
        order = order[:max_nms]

    boxes_sorted = boxes[order]
    ious = probiou_np_matrix_for_nms(boxes_sorted)

    ious = np.triu(ious, k=1)
    keep_sorted = np.where(ious.max(axis=0) < iou_thres)[0]

    keep = order[keep_sorted]

    if len(keep) > max_det:
        keep = keep[:max_det]

    return keep.astype(np.int64).tolist()


def postprocess_obb_np(
    pred,
    conf_thres=0.001,
    iou_thres=0.7,
    max_det=300,
    max_nms=30000,
):
    if pred is None or len(pred) == 0:
        return np.zeros((0, 7), dtype=np.float32)

    pred = np.asarray(pred, dtype=np.float32)

    boxes = pred[:, :5]
    cls_scores = pred[:, 5:]

    box_idx, cls_ids = np.where(cls_scores >= conf_thres)

    if len(box_idx) == 0:
        return np.zeros((0, 7), dtype=np.float32)

    boxes = boxes[box_idx]
    scores = cls_scores[box_idx, cls_ids]
    cls_ids = cls_ids.astype(np.int64)

    if len(scores) > max_nms:
        order = scores.argsort()[::-1][:max_nms]
        boxes = boxes[order]
        scores = scores[order]
        cls_ids = cls_ids[order]

    final_boxes = []
    final_scores = []
    final_cls = []

    for c in np.unique(cls_ids):
        inds = np.where(cls_ids == c)[0]

        keep_inds = rotated_nms_np(
            boxes[inds],
            scores[inds],
            iou_thres=iou_thres,
            max_det=max_det,
        )

        if len(keep_inds) == 0:
            continue

        final_boxes.append(boxes[inds][keep_inds])
        final_scores.append(scores[inds][keep_inds])
        final_cls.append(cls_ids[inds][keep_inds])

    if len(final_boxes) == 0:
        return np.zeros((0, 7), dtype=np.float32)

    final_boxes = np.concatenate(final_boxes, axis=0)
    final_scores = np.concatenate(final_scores, axis=0)
    final_cls = np.concatenate(final_cls, axis=0)

    order = final_scores.argsort()[::-1][:max_det]

    dets = np.concatenate(
        [
            final_boxes[order],
            final_scores[order, None],
            final_cls[order, None].astype(np.float32),
        ],
        axis=1,
    )

    return dets.astype(np.float32)


def compute_ap(recall, precision):
    """
    Compute AP from precision-recall curve using 101-point interpolation.
    """
    recall = np.asarray(recall, dtype=np.float64)
    precision = np.asarray(precision, dtype=np.float64)

    # Add sentinel endpoints
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # 101-point interpolation
    x = np.linspace(0, 1, 101)
    ap = np.trapz(np.interp(x, mrec, mpre), x)

    return float(ap)


def polygon_iou_cv2(poly1, poly2):
    """
    polygon IoU for OBB.

    Args:
        poly1: [8], x1 y1 ... x4 y4
        poly2: [8]

    Returns:
        iou: float
    """
    p1 = np.asarray(poly1, dtype=np.float32).reshape(4, 2)
    p2 = np.asarray(poly2, dtype=np.float32).reshape(4, 2)

    area1 = abs(cv2.contourArea(p1))
    area2 = abs(cv2.contourArea(p2))

    if area1 <= 0 or area2 <= 0:
        return 0.0

    inter_type, inter_pts = cv2.intersectConvexConvex(p1, p2)

    if inter_pts is None:
        inter_area = 0.0
    else:
        inter_area = abs(float(inter_type))

    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0

    return float(inter_area / union)


def obb_map_eval(
    predictions,
    ground_truths,
    num_classes=5,
    iou_thres_list=None,
):
    """
    OBB mAP 评估。

    Args:
        predictions:
            list of dict
            [
                {
                    "image_id": str,
                    "cls": int,
                    "score": float,
                    "poly": np.ndarray shape [8],
                },
                ...
            ]

        ground_truths:
            list of dict
            [
                {
                    "image_id": str,
                    "cls": int,
                    "poly": np.ndarray shape [8],
                },
                ...
            ]

        num_classes:
            类别数。

        iou_thres_list:
            IoU 阈值列表。
            None 时默认 COCO 风格 0.50:0.05:0.95。

    Returns:
        results: dict
    """
    if iou_thres_list is None:
        iou_thres_list = np.arange(0.50, 0.96, 0.05)

    iou_thres_list = np.asarray(iou_thres_list, dtype=np.float32)

    # 按类别和 image_id 组织 GT
    gt_by_cls_img = {}
    n_gt_per_cls = np.zeros((num_classes,), dtype=np.int64)

    for gt in ground_truths:
        c = int(gt["cls"])
        img_id = gt["image_id"]

        if c < 0 or c >= num_classes:
            continue

        key = (c, img_id)
        gt_by_cls_img.setdefault(key, [])
        gt_by_cls_img[key].append(gt["poly"].astype(np.float32))

        n_gt_per_cls[c] += 1

    ap = np.zeros((len(iou_thres_list), num_classes), dtype=np.float32)

    for ti, iou_thres in enumerate(iou_thres_list):
        for c in range(num_classes):
            preds_c = [p for p in predictions if int(p["cls"]) == c]
            preds_c = sorted(preds_c, key=lambda x: x["score"], reverse=True)

            n_gt = int(n_gt_per_cls[c])
            if n_gt == 0:
                ap[ti, c] = np.nan
                continue

            if len(preds_c) == 0:
                ap[ti, c] = 0.0
                continue

            tp = np.zeros((len(preds_c),), dtype=np.float32)
            fp = np.zeros((len(preds_c),), dtype=np.float32)

            # 每个 IoU 阈值下，GT 只能被匹配一次
            matched = {}

            for pi, pred in enumerate(preds_c):
                img_id = pred["image_id"]
                key = (c, img_id)
                gt_polys = gt_by_cls_img.get(key, [])

                if len(gt_polys) == 0:
                    fp[pi] = 1.0
                    continue

                ious = np.array(
                    [polygon_iou_cv2(pred["poly"], g) for g in gt_polys],
                    dtype=np.float32,
                )

                best_i = int(ious.argmax())
                best_iou = float(ious[best_i])

                match_key = (c, img_id, best_i)

                if best_iou >= float(iou_thres) and match_key not in matched:
                    tp[pi] = 1.0
                    matched[match_key] = True
                else:
                    fp[pi] = 1.0

            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)

            recall = tp_cum / (n_gt + 1e-16)
            precision = tp_cum / (tp_cum + fp_cum + 1e-16)

            ap[ti, c] = compute_ap(recall, precision)

    ap50 = ap[0]
    map50 = np.nanmean(ap50)
    map5095 = np.nanmean(ap)

    return {
        "ap": ap,
        "ap50": ap50,
        "map50": float(map50),
        "map": float(map5095),
        "iou_thres_list": iou_thres_list,
        "n_gt_per_cls": n_gt_per_cls,
    }


def obb_map_eval_probiou(
    predictions,
    ground_truths,
    num_classes=5,
    iou_thres_list=None,
):
    """
    OBB mAP with ProbIoU, closer to Ultralytics OBBValidator.
    predictions item requires:
        image_id, cls, score, xywhr
    ground_truths item requires:
        image_id, cls, xywhr
    """
    if iou_thres_list is None:
        iou_thres_list = np.arange(0.50, 0.96, 0.05)

    iou_thres_list = np.asarray(iou_thres_list, dtype=np.float32)

    gt_by_cls_img = {}
    n_gt_per_cls = np.zeros((num_classes,), dtype=np.int64)

    for gt in ground_truths:
        c = int(gt["cls"])
        img_id = gt["image_id"]
        if c < 0 or c >= num_classes:
            continue
        key = (c, img_id)
        gt_by_cls_img.setdefault(key, [])
        gt_by_cls_img[key].append(gt["xywhr"].astype(np.float32))
        n_gt_per_cls[c] += 1

    ap = np.zeros((len(iou_thres_list), num_classes), dtype=np.float32)

    for ti, iou_thres in enumerate(iou_thres_list):
        for c in range(num_classes):
            preds_c = [p for p in predictions if int(p["cls"]) == c]
            preds_c = sorted(preds_c, key=lambda x: x["score"], reverse=True)

            n_gt = int(n_gt_per_cls[c])
            if n_gt == 0:
                ap[ti, c] = np.nan
                continue

            if len(preds_c) == 0:
                ap[ti, c] = 0.0
                continue

            tp = np.zeros((len(preds_c),), dtype=np.float32)
            fp = np.zeros((len(preds_c),), dtype=np.float32)
            matched = {}

            for pi, pred in enumerate(preds_c):
                img_id = pred["image_id"]
                key = (c, img_id)
                gt_boxes = gt_by_cls_img.get(key, [])

                if len(gt_boxes) == 0:
                    fp[pi] = 1.0
                    continue

                gt_boxes = np.asarray(gt_boxes, dtype=np.float32)
                pred_box = np.asarray(pred["xywhr"], dtype=np.float32).reshape(1, 5)

                # probiou_np_matrix expects pred_boxes [A,5], gt_boxes [G,5]
                # returns [G,A]
                ious = probiou_np_matrix(pred_box, gt_boxes).reshape(-1)

                best_i = int(ious.argmax())
                best_iou = float(ious[best_i])
                match_key = (c, img_id, best_i)

                if best_iou >= float(iou_thres) and match_key not in matched:
                    tp[pi] = 1.0
                    matched[match_key] = True
                else:
                    fp[pi] = 1.0

            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)

            recall = tp_cum / (n_gt + 1e-16)
            precision = tp_cum / (tp_cum + fp_cum + 1e-16)

            ap[ti, c] = compute_ap(recall, precision)

    return {
        "ap": ap,
        "ap50": ap[0],
        "map50": float(np.nanmean(ap[0])),
        "map": float(np.nanmean(ap)),
        "iou_thres_list": iou_thres_list,
        "n_gt_per_cls": n_gt_per_cls,
    }