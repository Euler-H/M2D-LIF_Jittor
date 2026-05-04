import math
import numpy as np
import jittor as jt
from jittor import nn

from jdet.ops.yolo_obb_ops import labels_poly_to_xywhr
from jdet.models.utils.yolov8_obb_modules import dist2rbox


def bce_with_logits_sum(pred, target):
    """
    Stable BCEWithLogits with sum reduction.
    Jittor 的 binary_cross_entropy_with_logits 不同版本接口不完全一致，
    这里手动实现 sum reduction，避免 reduction 参数兼容问题。
    """
    loss = jt.maximum(pred, jt.zeros_like(pred)) - pred * target + jt.log(
        1.0 + jt.exp(-jt.abs(pred))
    )
    return loss.sum()

def bbox2dist(anchor_points, bbox_xyxy, reg_max):
    """
    anchor_points: [N, 2], normalized feature-grid coordinate
    bbox_xyxy:     [N, 4], x1 y1 x2 y2, normalized to grid coordinate
    return:        [N, 4], l t r b
    """
    x1y1 = bbox_xyxy[:, :2]
    x2y2 = bbox_xyxy[:, 2:]
    return jt.concat(
        [
            anchor_points - x1y1,
            x2y2 - anchor_points,
        ],
        dim=1,
    ).clamp(0, reg_max - 1 - 0.01)


def xywhr_to_xyxy_np(xywhr):
    """
    简化版：用旋转框的水平外接框近似监督 DFL。
    后续再替换为旋转 IoU。
    xywhr: [N, 5], normalized cx cy w h theta
    """
    cx = xywhr[:, 0]
    cy = xywhr[:, 1]
    w = xywhr[:, 2]
    h = xywhr[:, 3]

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)


def dfl_loss(pred_dist, target):
    """
    pred_dist: [N, 4, reg_max]
    target:    [N, 4], continuous distance
    """
    reg_max = pred_dist.shape[-1]

    target = target.clamp(0, reg_max - 1 - 0.01)
    tl = target.floor().int()
    tr = tl + 1

    wl = tr.float() - target
    wr = target - tl.float()

    pred = pred_dist.reshape(-1, reg_max)
    tl = tl.reshape(-1)
    tr = tr.reshape(-1)
    wl = wl.reshape(-1)
    wr = wr.reshape(-1)

    loss_l = nn.cross_entropy_loss(pred, tl, reduction="none") * wl
    loss_r = nn.cross_entropy_loss(pred, tr, reduction="none") * wr

    return (loss_l + loss_r).mean()

def dfl_loss_per_sample(pred_dist, target):
    """
    pred_dist:
        [N, 4, reg_max]
    target:
        [N, 4]

    return:
        [N, 1]
    """
    reg_max = pred_dist.shape[-1]

    target = target.clamp(0, reg_max - 1 - 0.01)
    tl = target.floor().int()
    tr = tl + 1

    wl = tr.float() - target
    wr = target - tl.float()

    pred = pred_dist.reshape(-1, reg_max)

    tl_flat = tl.reshape(-1)
    tr_flat = tr.reshape(-1)
    wl_flat = wl.reshape(-1)
    wr_flat = wr.reshape(-1)

    loss_l = nn.cross_entropy_loss(pred, tl_flat, reduction="none") * wl_flat
    loss_r = nn.cross_entropy_loss(pred, tr_flat, reduction="none") * wr_flat

    loss = (loss_l + loss_r).reshape(-1, 4).mean(dim=1)

    return loss.view(-1, 1)

def make_anchors(feats, strides, grid_cell_offset=0.5):
    """
    为 P3/P4/P5 生成 anchor points。

    feats:
        list of feature maps:
        P3 [B, C, H3, W3]
        P4 [B, C, H4, W4]
        P5 [B, C, H5, W5]

    return:
        anchor_points: [A, 2], grid 坐标
        stride_tensor: [A, 1], 每个点对应 stride
    """
    anchor_points = []
    stride_tensor = []

    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape

        sx = jt.arange(w).float() + grid_cell_offset
        sy = jt.arange(h).float() + grid_cell_offset

        yy, xx = jt.meshgrid([sy, sx])
        points = jt.stack([xx, yy], dim=-1).view(-1, 2)

        anchor_points.append(points)
        stride_tensor.append(jt.ones((h * w, 1)).float() * stride)

    anchor_points = jt.concat(anchor_points, dim=0)
    stride_tensor = jt.concat(stride_tensor, dim=0)

    return anchor_points, stride_tensor

def dist_decode(pred_distri, reg_max):
    """
    pred_distri:
        [B, A, 4 * reg_max]

    return:
        pred_dist:
        [B, A, 4]
    """
    b, a, c = pred_distri.shape
    pred = pred_distri.view(b, a, 4, reg_max)
    pred = nn.softmax(pred, dim=3)

    proj = jt.arange(reg_max).float().view(1, 1, 1, reg_max)
    pred_dist = (pred * proj).sum(dim=3)

    return pred_dist

def dist2bbox(distance, anchor_points):
    """
    distance:
        [B, A, 4] or [N, 4]
        l, t, r, b

    anchor_points:
        [A, 2] or [N, 2]

    return:
        xyxy:
        [B, A, 4] or [N, 4]
    """
    lt = distance[..., :2]
    rb = distance[..., 2:]

    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb

    return jt.concat([x1y1, x2y2], dim=-1)


def xyxytheta2xywhr(xyxy, theta):
    """
    xyxy:
        [B, A, 4] 或 [N, 4]
    theta:
        [B, A, 1] 或 [N, 1]

    return:
        xywhr:
        [..., 5]
    """
    x1, y1, x2, y2 = xyxy[..., 0], xyxy[..., 1], xyxy[..., 2], xyxy[..., 3]

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = (x2 - x1).clamp(1e-4)
    h = (y2 - y1).clamp(1e-4)

    return jt.stack([cx, cy, w, h, theta.squeeze(-1)], dim=-1)

def xywhr_to_xyxy_jt(xywhr):
    """
    xywhr:
        [..., 5]

    return:
        xyxy:
        [..., 4]
    """
    cx = xywhr[..., 0]
    cy = xywhr[..., 1]
    w = xywhr[..., 2]
    h = xywhr[..., 3]

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    return jt.stack([x1, y1, x2, y2], dim=-1)


def probiou_jt_matrix(pred_boxes, gt_boxes, eps=1e-7):
    """
    Jittor matrix ProbIoU for TaskAlignedAssigner.

    Args:
        pred_boxes: [A, 5], cx, cy, w, h, theta, image scale
        gt_boxes:   [G, 5], cx, cy, w, h, theta, image scale

    Returns:
        probiou: [G, A]
    """
    px = pred_boxes[None, :, 0]
    py = pred_boxes[None, :, 1]
    pw = pred_boxes[None, :, 2].clamp(1e-6)
    ph = pred_boxes[None, :, 3].clamp(1e-6)
    pt = pred_boxes[None, :, 4]

    gx = gt_boxes[:, None, 0]
    gy = gt_boxes[:, None, 1]
    gw = gt_boxes[:, None, 2].clamp(1e-6)
    gh = gt_boxes[:, None, 3].clamp(1e-6)
    gt = gt_boxes[:, None, 4]

    def _cov(w, h, theta):
        cos = jt.cos(theta)
        sin = jt.sin(theta)
        w2 = (w ** 2) / 12.0
        h2 = (h ** 2) / 12.0
        a = w2 * cos ** 2 + h2 * sin ** 2
        b = w2 * sin ** 2 + h2 * cos ** 2
        c = (w2 - h2) * sin * cos
        return a, b, c

    a1, b1, c1 = _cov(pw, ph, pt)
    a2, b2, c2 = _cov(gw, gh, gt)

    a = a1 + a2
    b = b1 + b2
    c = c1 + c2

    denominator = a * b - c ** 2 + eps
    dx = px - gx
    dy = py - gy

    t1 = 0.25 * (a * dy ** 2 + b * dx ** 2) / denominator
    t2 = 0.5 * c * dx * dy / denominator

    det1 = (a1 * b1 - c1 ** 2).clamp(eps)
    det2 = (a2 * b2 - c2 ** 2).clamp(eps)
    det = denominator.clamp(eps)

    t3 = 0.5 * jt.log(det / (4.0 * jt.sqrt(det1 * det2 + eps) + eps) + eps)
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = jt.sqrt(1.0 - jt.exp(-bd) + eps)

    return (1.0 - hd).clamp(0.0, 1.0)


def xywhr2xyxyxyxy_jt(xywhr):
    """
    Convert rotated boxes to four corner points with Jittor tensor ops.

    Args:
        xywhr: [G, 5], cx, cy, w, h, theta

    Returns:
        corners: [G, 4, 2]
    """
    cx = xywhr[:, 0]
    cy = xywhr[:, 1]
    w = xywhr[:, 2]
    h = xywhr[:, 3]
    theta = xywhr[:, 4]

    cos = jt.cos(theta)
    sin = jt.sin(theta)
    dx = w / 2.0
    dy = h / 2.0

    local_x = jt.stack([-dx, dx, dx, -dx], dim=1)
    local_y = jt.stack([-dy, -dy, dy, dy], dim=1)

    x = local_x * cos[:, None] - local_y * sin[:, None] + cx[:, None]
    y = local_x * sin[:, None] + local_y * cos[:, None] + cy[:, None]

    return jt.stack([x, y], dim=-1)


def select_candidates_in_rotated_gts_jt(anchor_points, gt_bboxes, eps=1e-9):
    """
    Jittor version of rotated-box point-in-polygon test.

    Args:
        anchor_points: [A, 2], image scale
        gt_bboxes:     [G, 5], image scale xywhr

    Returns:
        inside: [G, A] float mask, value in {0, 1}
    """
    corners = xywhr2xyxyxyxy_jt(gt_bboxes)

    a = corners[:, 0:1, :]
    b = corners[:, 1:2, :]
    d = corners[:, 3:4, :]

    ab = b - a
    ad = d - a
    ap = anchor_points[None, :, :] - a

    norm_ab = (ab * ab).sum(dim=-1)
    norm_ad = (ad * ad).sum(dim=-1)
    ap_dot_ab = (ap * ab).sum(dim=-1)
    ap_dot_ad = (ap * ad).sum(dim=-1)

    inside = (
        (ap_dot_ab >= -eps)
        & (ap_dot_ab <= norm_ab + eps)
        & (ap_dot_ad >= -eps)
        & (ap_dot_ad <= norm_ad + eps)
    )

    return inside.float()

class TaskAlignedAssigner:
    """
    YOLOv8/YOLOv8-OBB Task-Aligned Assigner implemented with Jittor tensors.

    Matching metric:
        alignment_metric = cls_score^alpha * IoU^beta

    Compared with the previous implementation, this version removes the CPU/numpy
    matching path from the assigner itself. Candidate filtering, matrix ProbIoU,
    top-k selection, duplicate-positive conflict resolution, and target score
    normalization are all computed by Jittor tensor ops.
    """

    def __init__(self, topk=10, num_classes=5, alpha=0.5, beta=6.0, eps=1e-9, min_pos_score=0.05, use_nearest_fallback=False,):
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        # Prevent early background-only collapse: when initial ProbIoU is nearly
        # zero, selected positives must still carry non-zero cls/box/DFL weight.
        self.min_pos_score = float(min_pos_score)
        self.use_nearest_fallback = bool(use_nearest_fallback)

    def __call__(
        self,
        pred_scores,
        pred_bboxes,
        anchor_points,
        gt_labels,
        gt_bboxes,
        mask_gt,
    ):
        """
        Args:
            pred_scores:   [B, A, C], sigmoid probabilities, detached
            pred_bboxes:   [B, A, 5], image-pixel xywhr, detached
            anchor_points: [A, 2], image-pixel anchor points
            gt_labels:     [B, M, 1]
            gt_bboxes:     [B, M, 5], image-pixel xywhr
            mask_gt:       [B, M, 1]

        Returns:
            target_labels: [B, A]
            target_bboxes: [B, A, 5]
            target_scores: [B, A, C]
            fg_mask:       [B, A]
            target_gt_idx: [B, A]
        """
        b, a, c = pred_scores.shape

        target_labels_all = []
        target_bboxes_all = []
        target_scores_all = []
        fg_masks_all = []
        target_gt_idx_all = []

        anchor_ids = jt.arange(a).view(1, a)
        class_ids = jt.arange(self.num_classes).view(1, self.num_classes)
        background_labels = jt.full((a,), self.num_classes, dtype=jt.int32)
        zero_bboxes = jt.zeros((a, 5), dtype=pred_bboxes.dtype)
        zero_scores = jt.zeros((a, self.num_classes), dtype=pred_scores.dtype)
        zero_gt_idx = jt.zeros((a,), dtype=jt.int32)

        for bi in range(b):
            ps = pred_scores[bi]
            pb = pred_bboxes[bi]

            # Static-shape assignment path. Do not slice by valid_idx here:
            # Jittor compiles kernels per tensor shape, and per-batch GT counts
            # are different almost every iteration, which can make training look
            # stuck due to repeated compilation. Invalid padded GTs are masked.
            valid_gt = (mask_gt[bi, :, 0] > 0).float().view(-1, 1)  # [M, 1]
            gl = gt_labels[bi, :, 0].int().clamp(0, self.num_classes - 1)  # [M]
            gb = gt_bboxes[bi].float()  # [M, 5]
            g = gb.shape[0]

            # 1. Candidate points inside rotated GTs. If a valid GT contains no
            # anchor point, fall back to its nearest center anchor.
            inside = select_candidates_in_rotated_gts_jt(anchor_points, gb) * valid_gt

            if self.use_nearest_fallback:
                center_dist = (
                    (anchor_points[None, :, 0] - gb[:, None, 0]) ** 2
                    + (anchor_points[None, :, 1] - gb[:, None, 1]) ** 2
                )
                nearest_idx, _ = center_dist.argmin(dim=1)
                nearest_mask = (anchor_ids == nearest_idx.view(g, 1)).float() * valid_gt
                has_inside = (inside.sum(dim=1, keepdims=True) > 0).float()
                inside = (inside * has_inside + nearest_mask * (1.0 - has_inside)) * valid_gt
            else:
                nearest_mask = jt.zeros_like(inside)

            # center_dist = (
            #     (anchor_points[None, :, 0] - gb[:, None, 0]) ** 2
            #     + (anchor_points[None, :, 1] - gb[:, None, 1]) ** 2
            # )
            # nearest_idx, _ = center_dist.argmin(dim=1)
            # nearest_mask = (anchor_ids == nearest_idx.view(g, 1)).float() * valid_gt
            # has_inside = (inside.sum(dim=1, keepdims=True) > 0).float()
            # inside = (inside * has_inside + nearest_mask * (1.0 - has_inside)) * valid_gt

            # 2. Matrix ProbIoU: [M, A]
            overlaps = probiou_jt_matrix(pb, gb) * inside

            # 3. Classification score for each GT class: [M, A]
            cls_scores = ps[:, gl].permute(1, 0).clamp(0.0, 1.0)
            align_metrics = (cls_scores ** self.alpha) * (overlaps ** self.beta) * inside

            # 4. Per-GT top-k candidate selection. The nearest-center fallback is
            # kept when all alignment metrics of a GT are zero.
            topk = min(self.topk, a)
            topk_metrics, _ = align_metrics.topk(topk, dim=1, largest=True)
            kth_metric = topk_metrics[:, topk - 1:topk]
            candidate_mask = ((align_metrics >= kth_metric) & (align_metrics > 0)).float()

            metric_max = align_metrics.max(dim=1, keepdims=True)
            has_metric = (metric_max > 0).float()
            candidate_mask = candidate_mask * has_metric + nearest_mask * (1.0 - has_metric)
            candidate_mask = candidate_mask * inside * valid_gt

            # 5. If one anchor is assigned to multiple GTs, keep the GT with the
            # largest ProbIoU, matching the standard TAL conflict-resolution rule.
            masked_overlaps = overlaps * candidate_mask + (candidate_mask <= 0).float() * (-1.0)
            matched_gt_idx, matched_iou = masked_overlaps.argmax(dim=0)
            fg_mask = matched_iou > -0.5

            matched_gt_idx = matched_gt_idx.int()
            matched_labels = gl[matched_gt_idx]
            matched_bboxes = gb[matched_gt_idx]

            target_labels = jt.where(fg_mask, matched_labels, background_labels)
            target_bboxes = matched_bboxes * fg_mask.float().view(a, 1)

            # 6. YOLOv8 target-score normalization.
            pos_align = (align_metrics * candidate_mask).max(dim=1, keepdims=True)
            pos_overlap = (overlaps * candidate_mask).max(dim=1, keepdims=True)
            norm_metric = align_metrics * pos_overlap / (pos_align + self.eps)
            norm_metric = norm_metric.max(dim=0)

            matched_overlap = overlaps[matched_gt_idx, anchor_ids.reshape(-1)]
            norm_metric = jt.where(norm_metric > 0, norm_metric, matched_overlap)
            norm_metric = norm_metric * fg_mask.float()
            if self.min_pos_score > 0:
                pos_floor = jt.ones_like(norm_metric) * self.min_pos_score
                norm_metric = jt.where(fg_mask, jt.maximum(norm_metric, pos_floor), norm_metric)

            target_scores = (matched_labels.view(a, 1) == class_ids).float()
            target_scores = target_scores * norm_metric.view(a, 1)

            target_labels_all.append(target_labels.int())
            target_bboxes_all.append(target_bboxes.float())
            target_scores_all.append(target_scores.float())
            fg_masks_all.append(fg_mask.int())
            target_gt_idx_all.append(matched_gt_idx.int())

        return (
            jt.stack(target_labels_all, dim=0),
            jt.stack(target_bboxes_all, dim=0),
            jt.stack(target_scores_all, dim=0),
            jt.stack(fg_masks_all, dim=0),
            jt.stack(target_gt_idx_all, dim=0),
        )

def probiou_np(box1, box2, eps=1e-7):
    """
    numpy 版 ProbIoU，用于 TaskAlignedAssigner。
    box1, box2:
        [5] = cx, cy, w, h, theta
    """

    def cov(box):
        w = max(float(box[2]), 1e-6)
        h = max(float(box[3]), 1e-6)
        theta = float(box[4])

        cos = np.cos(theta)
        sin = np.sin(theta)

        w2 = (w ** 2) / 12.0
        h2 = (h ** 2) / 12.0

        a = w2 * cos ** 2 + h2 * sin ** 2
        b = w2 * sin ** 2 + h2 * cos ** 2
        c = (w2 - h2) * sin * cos

        return a, b, c

    x1, y1 = float(box1[0]), float(box1[1])
    x2, y2 = float(box2[0]), float(box2[1])

    a1, b1, c1 = cov(box1)
    a2, b2, c2 = cov(box2)

    a = a1 + a2
    b = b1 + b2
    c = c1 + c2

    denominator = a * b - c ** 2 + eps

    dx = x1 - x2
    dy = y1 - y2

    t1 = 0.25 * (a * dy ** 2 + b * dx ** 2) / denominator
    t2 = 0.5 * c * dx * dy / denominator

    det1 = max(a1 * b1 - c1 ** 2, eps)
    det2 = max(a2 * b2 - c2 ** 2, eps)
    det = max(denominator, eps)

    t3 = 0.5 * np.log(det / (4.0 * np.sqrt(det1 * det2) + eps) + eps)

    bd = np.clip(t1 + t2 + t3, eps, 100.0)
    hd = np.sqrt(1.0 - np.exp(-bd) + eps)
    probiou = 1.0 - hd

    return float(np.clip(probiou, 0.0, 1.0))
    
class YOLOv8OBBLoss:
    """
    YOLOv8-OBB Loss.

    当前版本：
        cls_loss   : BCE
        dfl_loss   : DFL
        box_loss   : ProbIoU loss on xywhr
        angle_loss : periodic angle loss

    注意：
        正样本分配仍然是中心点分配版，后续再替换为 TaskAlignedAssigner。
    """

    def __init__(self, nc=5, reg_max=16, strides=(8, 16, 32), imgsz=640, assigner_min_pos_score=0.05, max_gt=256):
        self.nc = nc
        self.reg_max = reg_max
        self.strides = strides
        self.imgsz = imgsz
        self.assigner_min_pos_score = float(assigner_min_pos_score)
        self.max_gt = int(max_gt)

        self.cls_gain = 1.0
        self.box_gain = 7.5
        self.dfl_gain = 1.5
        self.angle_gain = 0.5
        
        self.assigner = TaskAlignedAssigner(
            topk=10,
            num_classes=nc,
            alpha=0.5,
            beta=6.0,
            min_pos_score=self.assigner_min_pos_score,
            use_nearest_fallback=False, # 标准 yolov8-OBB 格式
        )
        
    def forward_tal(self, preds, targets):
        """
        低显存 TaskAlignedAssigner 版本 YOLOv8-OBB Loss。

        关键原则：
            1. assigner 路径全部 detach；
            2. full anchors 只用于分类 loss；
            3. box / dfl / angle 只对正样本计算；
            4. 不构造全量带梯度 pred_bboxes_img。
        """

# 针对空标签处理
# 空标签分类 loss 的归一化从 / batch_size 改成 / (batch_size * num_anchors)，否则空标签时 cls_loss 仍可能偏大
######################################################################################
        device_zero = preds[0].sum() * 0.0
        batch_size = preds[0].shape[0]

        if targets is None or len(targets) == 0:
            targets_np = None
        else:
            targets_np = labels_poly_to_xywhr(targets)

        # Official YOLOv8-OBB tiny rotated-box filtering.
        # targets_np format: [batch_id, cls, cx, cy, w, h, theta], normalized.
        if targets_np is not None and len(targets_np):
            rw = targets_np[:, 4] * float(self.imgsz)
            rh = targets_np[:, 5] * float(self.imgsz)
            keep = (rw >= 2.0) & (rh >= 2.0)
            targets_np = targets_np[keep]

        empty_targets = targets_np is None or len(targets_np) == 0

        pred_distri, pred_scores_logits, pred_angle_logits = preprocess_preds(
            preds,
            self.nc,
            self.reg_max,
        )

        if empty_targets:
            target_scores = jt.zeros_like(pred_scores_logits)

            cls_loss = bce_with_logits_sum(pred_scores_logits, target_scores)

            # Stable average over anchors and batch.
            num_pred = max(float(pred_scores_logits.shape[1]), 1.0)
            cls_loss = cls_loss / (max(float(batch_size), 1.0) * num_pred)

            return dict(
                box_loss=device_zero,
                cls_loss=self.cls_gain * cls_loss,
                dfl_loss=device_zero,
                angle_loss=device_zero,
            )
######################################################################################

        pred_distri, pred_scores_logits, pred_angle_logits = preprocess_preds(
            preds,
            self.nc,
            self.reg_max,
        )

        anchor_points, stride_tensor = make_anchors(
            preds,
            self.strides,
            grid_cell_offset=0.5,
        )

        stride_expand = stride_tensor.view(1, -1, 1)

        gt_labels, gt_bboxes_norm, mask_gt = build_targets_tensor(
            targets_np,
            batch_size=batch_size,
            max_gt=self.max_gt,
        )

        gt_bboxes_img = gt_to_img_scale(gt_bboxes_norm, self.imgsz)

        # ------------------------------------------------------------
        # 1. assigner 路径：全部 detach，不参与反向传播
        # ------------------------------------------------------------
        pred_scores_assign = jt.sigmoid(pred_scores_logits.detach())

        pred_dist_assign = dist_decode(
            pred_distri.detach(),
            self.reg_max,
        )

        pred_theta_assign = (
            jt.sigmoid(pred_angle_logits.detach()) - 0.25
        ) * math.pi

        pred_rbox_grid_assign = dist2rbox(
            pred_dist_assign,
            pred_theta_assign,
            anchor_points.view(1, -1, 2),
        )

        pred_bboxes_grid_assign = jt.concat(
            [pred_rbox_grid_assign, pred_theta_assign],
            dim=-1,
        )

        pred_bboxes_img_assign = jt.concat(
            [
                pred_bboxes_grid_assign[:, :, 0:4] * stride_expand,
                pred_bboxes_grid_assign[:, :, 4:5],
            ],
            dim=-1,
        )

        anchor_points_img = anchor_points * stride_tensor

        target_labels, target_bboxes_img, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores_assign,
            pred_bboxes_img_assign,
            anchor_points_img,
            gt_labels,
            gt_bboxes_img,
            mask_gt,
        )

        # ------------------------------------------------------------
        # 2. 分类损失：全量 anchors，但是不需要 decode box
        # ------------------------------------------------------------
        target_scores_sum = target_scores.sum().clamp(1.0)

        cls_loss = bce_with_logits_sum(
            pred_scores_logits,
            target_scores,
        ) / target_scores_sum

        # ------------------------------------------------------------
        # 3. 没有正样本时，只返回 cls
        # ------------------------------------------------------------
        fg_indices = jt.nonzero(fg_mask > 0)

        if fg_indices.shape[0] == 0:
            return dict(
                box_loss=device_zero,
                cls_loss=self.cls_gain * cls_loss,
                dfl_loss=device_zero,
                angle_loss=device_zero,
            )

        fg_b_jt = fg_indices[:, 0].int()
        fg_a_jt = fg_indices[:, 1].int()

        # ------------------------------------------------------------
        # 4. 只对正样本 gather 后 decode，避免全量回归图
        # ------------------------------------------------------------
        pred_distri_pos = pred_distri[fg_b_jt, fg_a_jt]  # [N, 4*reg_max]
        pred_distri_pos = pred_distri_pos.view(-1, 4, self.reg_max)

        prob = nn.softmax(pred_distri_pos, dim=2)
        proj = jt.arange(self.reg_max).float().view(1, 1, self.reg_max)
        pred_dist_pos = (prob * proj).sum(dim=2)  # [N, 4]

        anchor_grid_pos = anchor_points[fg_a_jt]  # [N, 2]
        
        pred_theta_pos = (
            jt.sigmoid(pred_angle_logits[fg_b_jt, fg_a_jt]).view(-1, 1) - 0.25
        ) * math.pi

        pred_rbox_grid_pos = dist2rbox(
            pred_dist_pos,
            pred_theta_pos,
            anchor_grid_pos,
        )

        pred_xywhr_grid_pos = jt.concat(
            [pred_rbox_grid_pos, pred_theta_pos],
            dim=1,
        )

        stride_pos = stride_tensor[fg_a_jt]  # [N, 1]

        pred_bboxes_pos = jt.concat(
            [
                pred_xywhr_grid_pos[:, 0:4] * stride_pos,
                pred_xywhr_grid_pos[:, 4:5],
            ],
            dim=1,
        )

        target_bboxes_pos = target_bboxes_img[fg_b_jt, fg_a_jt]  # [N, 5]

        # ------------------------------------------------------------
        # 5. DFL target：target box 从 image scale 转回对应 grid scale
        # ------------------------------------------------------------
        target_grid = jt.concat(
            [
                target_bboxes_pos[:, 0:4] / stride_pos,
                target_bboxes_pos[:, 4:5],
            ],
            dim=1,
        )

        target_xyxy_grid = xywhr_to_xyxy_jt(target_grid)

        target_ltrb_pos = bbox2dist(
            anchor_grid_pos,
            target_xyxy_grid,
            self.reg_max,
        )

        # ------------------------------------------------------------
        # 6. Loss
        # ------------------------------------------------------------
        weight = target_scores[fg_b_jt, fg_a_jt].sum(dim=-1).view(-1, 1)
        target_scores_sum = target_scores.sum().clamp(1.0)

        iou = probiou_jt(pred_bboxes_pos, target_bboxes_pos)
        box_loss = ((1.0 - iou) * weight).sum() / target_scores_sum

        dfl_raw = dfl_loss_per_sample(pred_distri_pos, target_ltrb_pos)
        dfl_loss_value = (dfl_raw * weight).sum() / target_scores_sum

        angle_loss = device_zero

        box_loss = self.box_gain * box_loss
        cls_loss = self.cls_gain * cls_loss
        dfl_loss_value = self.dfl_gain * dfl_loss_value
        angle_loss = self.angle_gain * angle_loss

        return dict(
            box_loss=box_loss,
            cls_loss=cls_loss,
            dfl_loss=dfl_loss_value,
            angle_loss=angle_loss,
        )     
    def __call__(self, preds, targets):
        # return self.forward_center(preds, targets)
        return self.forward_tal(preds, targets)    
    

    def forward_center(self, preds, targets):
        """
        preds:
            list of 3 tensors
            P3: [B, 4*reg_max + nc + 1, 80, 80]
            P4: [B, 4*reg_max + nc + 1, 40, 40]
            P5: [B, 4*reg_max + nc + 1, 20, 20]

        targets:
            [M, 10]
            batch_id, cls, x1, y1, ..., x4, y4
        """

        targets_np = labels_poly_to_xywhr(targets)

        device_zero = preds[0].sum() * 0.0

        cls_losses = []
        box_losses = []
        dfl_losses = []
        angle_losses = []

        if targets_np is None or targets_np.shape[0] == 0:
            return dict(
                box_loss=device_zero,
                cls_loss=device_zero,
                dfl_loss=device_zero,
                angle_loss=device_zero,
            )

        for level, p in enumerate(preds):
            b, c, h, w = p.shape

            box_ch = 4 * self.reg_max
            cls_start = box_ch
            cls_end = cls_start + self.nc
            angle_start = cls_end

            pred_box = p[:, :box_ch, :, :]
            pred_cls = p[:, cls_start:cls_end, :, :]
            pred_angle = p[:, angle_start:angle_start + 1, :, :]

            cls_target = jt.zeros_like(pred_cls)

            matched_box_logits = []
            matched_dist_targets = []
            matched_angle_logits = []
            matched_angle_targets = []

            matched_anchors = []
            matched_target_xywhr = []

            for row in targets_np:
                bidx = int(row[0])
                cls_id = int(row[1])
                cx, cy, bw, bh, theta = row[2:7]

                if bidx < 0 or bidx >= b:
                    continue
                if cls_id < 0 or cls_id >= self.nc:
                    continue

                # 根据目标面积选择尺度
                area = bw * bh
                if area < 0.02 and level != 0:
                    continue
                if 0.02 <= area < 0.08 and level != 1:
                    continue
                if area >= 0.08 and level != 2:
                    continue

                gx = int(cx * w)
                gy = int(cy * h)

                gx = max(0, min(gx, w - 1))
                gy = max(0, min(gy, h - 1))

                cls_target[bidx, cls_id, gy, gx] = 1.0

                # 当前正样本网格点的 box distribution logits
                box_logits = pred_box[bidx:bidx + 1, :, gy:gy + 1, gx:gx + 1]
                box_logits = box_logits.view(4, self.reg_max)
                matched_box_logits.append(box_logits)

                # 当前正样本网格点的 angle logit
                angle_logit = pred_angle[bidx:bidx + 1, :, gy:gy + 1, gx:gx + 1]
                matched_angle_logits.append(angle_logit.view(1))
                matched_angle_targets.append(theta)

                # 构造 l/t/r/b target，单位是当前 feature grid
                xywhr = np.array([[cx, cy, bw, bh, theta]], dtype=np.float32)
                xyxy = xywhr_to_xyxy_np(xywhr)[0]

                x1 = xyxy[0] * w
                y1 = xyxy[1] * h
                x2 = xyxy[2] * w
                y2 = xyxy[3] * h

                anchor = jt.array([[gx + 0.5, gy + 0.5]]).float()
                bbox = jt.array([[x1, y1, x2, y2]]).float()

                dist_target = bbox2dist(anchor, bbox, self.reg_max)
                matched_dist_targets.append(dist_target.view(4))

                matched_anchors.append(anchor.view(2))

                # target xywhr 统一转到当前 feature grid 坐标
                target_xywhr_grid = jt.array([
                    cx * w,
                    cy * h,
                    max(bw * w, 1e-4),
                    max(bh * h, 1e-4),
                    theta
                ]).float()

                matched_target_xywhr.append(target_xywhr_grid)

            # 分类损失：每个 level 都要算，即使当前 level 没有正样本
            cls_loss = nn.binary_cross_entropy_with_logits(pred_cls, cls_target)

            if len(matched_box_logits) == 0:
                box_loss = device_zero
                dfl_loss_value = device_zero
                angle_loss = device_zero
            else:
                matched_box_logits = jt.stack(matched_box_logits, dim=0)
                matched_dist_targets = jt.stack(matched_dist_targets, dim=0)
                matched_anchors = jt.stack(matched_anchors, dim=0)
                matched_target_xywhr = jt.stack(matched_target_xywhr, dim=0)

                # [N, 4, reg_max] -> [N, 4]
                prob = nn.softmax(matched_box_logits, dim=2)
                proj = jt.arange(self.reg_max).float().view(1, 1, self.reg_max)
                pred_dist = (prob * proj).sum(dim=2)

                # pred_dist = l, t, r, b
                lt = pred_dist[:, 0:2]
                rb = pred_dist[:, 2:4]

                pred_xy = matched_anchors + (rb - lt) / 2.0
                pred_wh = (lt + rb).clamp(1e-4)

                matched_angle_logits = jt.concat(matched_angle_logits, dim=0)
                matched_angle_targets = jt.array(matched_angle_targets).float()

                pred_theta = (jt.sigmoid(matched_angle_logits) - 0.25) * math.pi

                pred_xywhr = jt.concat(
                    [
                        pred_xy,
                        pred_wh,
                        pred_theta.reshape(-1, 1),
                    ],
                    dim=1,
                )

                box_loss = probiou_loss(pred_xywhr, matched_target_xywhr)
                dfl_loss_value = dfl_loss(matched_box_logits, matched_dist_targets)
                angle_loss = periodic_angle_loss(pred_theta, matched_angle_targets)

            # 注意：这四个 append 必须在 if/else 外面，但仍在 for level 内
            cls_losses.append(cls_loss)
            box_losses.append(box_loss)
            dfl_losses.append(dfl_loss_value)
            angle_losses.append(angle_loss)

        # 注意：求平均、加权、return 必须在 for level 外面
        cls_loss = sum(cls_losses) / len(cls_losses)
        box_loss = sum(box_losses) / len(box_losses)
        dfl_loss_value = sum(dfl_losses) / len(dfl_losses)
        angle_loss = sum(angle_losses) / len(angle_losses)

        box_loss = self.box_gain * box_loss
        cls_loss = self.cls_gain * cls_loss
        dfl_loss_value = self.dfl_gain * dfl_loss_value
        angle_loss = self.angle_gain * angle_loss

        return dict(
            box_loss=box_loss,
            cls_loss=cls_loss,
            dfl_loss=dfl_loss_value,
            angle_loss=angle_loss,
        )
    
def periodic_angle_loss(pred_theta, target_theta):
    """
    pred_theta:   [N]
    target_theta: [N]

    OBB 角度具有周期性。
    使用 min(|d|, pi-|d|) 避免角度边界处损失异常。
    """
    diff = jt.abs(pred_theta - target_theta)
    diff = jt.minimum(diff, math.pi - diff)
    return diff.mean()

def _get_covariance_matrix(boxes):
    """
    将 xywhr 旋转框转换为二维高斯协方差矩阵参数。

    boxes:
        [N, 5] = cx, cy, w, h, theta

    return:
        a, b, c
        对应协方差矩阵：
        [[a, c],
         [c, b]]
    """
    w = boxes[:, 2].clamp(1e-6)
    h = boxes[:, 3].clamp(1e-6)
    theta = boxes[:, 4]

    cos = jt.cos(theta)
    sin = jt.sin(theta)

    w2 = (w ** 2) / 12.0
    h2 = (h ** 2) / 12.0

    a = w2 * cos ** 2 + h2 * sin ** 2
    b = w2 * sin ** 2 + h2 * cos ** 2
    c = (w2 - h2) * sin * cos

    return a, b, c

def probiou_jt(pred_boxes, target_boxes, eps=1e-7):
    """
    return:
        [N, 1] probiou
    """
    x1, y1 = pred_boxes[:, 0], pred_boxes[:, 1]
    x2, y2 = target_boxes[:, 0], target_boxes[:, 1]

    a1, b1, c1 = _get_covariance_matrix(pred_boxes)
    a2, b2, c2 = _get_covariance_matrix(target_boxes)

    a = a1 + a2
    b = b1 + b2
    c = c1 + c2

    denominator = a * b - c ** 2 + eps

    dx = x1 - x2
    dy = y1 - y2

    t1 = 0.25 * (a * dy ** 2 + b * dx ** 2) / denominator
    t2 = 0.5 * c * dx * dy / denominator

    det1 = a1 * b1 - c1 ** 2
    det2 = a2 * b2 - c2 ** 2
    det = denominator

    t3 = 0.5 * jt.log(
        det / (4.0 * jt.sqrt(det1 * det2 + eps) + eps) + eps
    )

    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = jt.sqrt(1.0 - jt.exp(-bd) + eps)

    iou = 1.0 - hd
    return iou.clamp(0.0, 1.0).view(-1, 1)

def probiou_loss(pred_boxes, target_boxes, eps=1e-7):
    if pred_boxes.shape[0] == 0:
        return pred_boxes.sum() * 0.0
    return (1.0 - probiou_jt(pred_boxes, target_boxes, eps)).mean()

def probiou_np_matrix(pred_boxes, gt_boxes, eps=1e-7):
    """
    向量化 numpy ProbIoU。

    Args:
        pred_boxes: [A, 5], cx, cy, w, h, theta
        gt_boxes:   [G, 5], cx, cy, w, h, theta

    Returns:
        probiou: [G, A]
    """

    pred_boxes = pred_boxes.astype(np.float32)
    gt_boxes = gt_boxes.astype(np.float32)

    # pred: [1, A]
    px = pred_boxes[None, :, 0]
    py = pred_boxes[None, :, 1]
    pw = np.maximum(pred_boxes[None, :, 2], 1e-6)
    ph = np.maximum(pred_boxes[None, :, 3], 1e-6)
    pt = pred_boxes[None, :, 4]

    # gt: [G, 1]
    gx = gt_boxes[:, None, 0]
    gy = gt_boxes[:, None, 1]
    gw = np.maximum(gt_boxes[:, None, 2], 1e-6)
    gh = np.maximum(gt_boxes[:, None, 3], 1e-6)
    gt = gt_boxes[:, None, 4]

    def cov_np(w, h, theta):
        cos = np.cos(theta)
        sin = np.sin(theta)

        w2 = (w ** 2) / 12.0
        h2 = (h ** 2) / 12.0

        a = w2 * cos ** 2 + h2 * sin ** 2
        b = w2 * sin ** 2 + h2 * cos ** 2
        c = (w2 - h2) * sin * cos
        return a, b, c

    a1, b1, c1 = cov_np(pw, ph, pt)
    a2, b2, c2 = cov_np(gw, gh, gt)

    a = a1 + a2
    b = b1 + b2
    c = c1 + c2

    denominator = a * b - c ** 2 + eps

    dx = px - gx
    dy = py - gy

    t1 = 0.25 * (a * dy ** 2 + b * dx ** 2) / denominator
    t2 = 0.5 * c * dx * dy / denominator

    det1 = np.maximum(a1 * b1 - c1 ** 2, eps)
    det2 = np.maximum(a2 * b2 - c2 ** 2, eps)
    det = np.maximum(denominator, eps)

    t3 = 0.5 * np.log(det / (4.0 * np.sqrt(det1 * det2) + eps) + eps)

    bd = np.clip(t1 + t2 + t3, eps, 100.0)
    hd = np.sqrt(1.0 - np.exp(-bd) + eps)

    probiou = 1.0 - hd
    probiou = np.clip(probiou, 0.0, 1.0)

    return probiou.astype(np.float32)

def build_targets_tensor(targets_np, batch_size, max_gt=None):
    """
    targets_np:
        [M, 7] = batch_id, cls, cx, cy, w, h, theta
        坐标归一化到 [0, 1]

    return:
        gt_labels: [B, max_gt, 1]
        gt_bboxes: [B, max_gt, 5]
        mask_gt:   [B, max_gt, 1]
    """
    if targets_np is None or targets_np.shape[0] == 0:
        max_gt = 1 if max_gt is None else max_gt
        gt_labels = np.zeros((batch_size, max_gt, 1), dtype=np.int64)
        gt_bboxes = np.zeros((batch_size, max_gt, 5), dtype=np.float32)
        mask_gt = np.zeros((batch_size, max_gt, 1), dtype=np.float32)
        return jt.array(gt_labels), jt.array(gt_bboxes), jt.array(mask_gt)

    if max_gt is None:
        counts = []
        for b in range(batch_size):
            counts.append(int((targets_np[:, 0] == b).sum()))
        max_gt = max(max(counts), 1)

    gt_labels = np.zeros((batch_size, max_gt, 1), dtype=np.int64)
    gt_bboxes = np.zeros((batch_size, max_gt, 5), dtype=np.float32)
    mask_gt = np.zeros((batch_size, max_gt, 1), dtype=np.float32)

    for b in range(batch_size):
        t = targets_np[targets_np[:, 0] == b]
        n = min(len(t), max_gt)

        if n == 0:
            continue

        gt_labels[b, :n, 0] = t[:n, 1].astype(np.int64)
        gt_bboxes[b, :n, :] = t[:n, 2:7].astype(np.float32)
        mask_gt[b, :n, 0] = 1.0

    return jt.array(gt_labels), jt.array(gt_bboxes), jt.array(mask_gt)

def preprocess_preds(preds, nc, reg_max):
    """
    将 P3/P4/P5 输出整理为 YOLOv8 loss 需要的格式。

    preds:
        list:
            [B, 4*reg_max + nc + 1, H, W]

    return:
        pred_distri: [B, A, 4*reg_max]
        pred_scores: [B, A, nc]
        pred_angle:  [B, A, 1]
    """
    pred_distri_list = []
    pred_scores_list = []
    pred_angle_list = []

    box_ch = 4 * reg_max
    cls_start = box_ch
    cls_end = cls_start + nc
    angle_start = cls_end

    for p in preds:
        b, c, h, w = p.shape

        pred_box = p[:, :box_ch, :, :]
        pred_cls = p[:, cls_start:cls_end, :, :]
        pred_angle = p[:, angle_start:angle_start + 1, :, :]

        pred_box = pred_box.permute(0, 2, 3, 1).reshape(b, h * w, box_ch)
        pred_cls = pred_cls.permute(0, 2, 3, 1).reshape(b, h * w, nc)
        pred_angle = pred_angle.permute(0, 2, 3, 1).reshape(b, h * w, 1)

        pred_distri_list.append(pred_box)
        pred_scores_list.append(pred_cls)
        pred_angle_list.append(pred_angle)

    pred_distri = jt.concat(pred_distri_list, dim=1)
    pred_scores = jt.concat(pred_scores_list, dim=1)
    pred_angle = jt.concat(pred_angle_list, dim=1)

    return pred_distri, pred_scores, pred_angle

def gt_to_img_scale(gt_bboxes, imgsz):
    """
    gt_bboxes:
        [B, M, 5], normalized xywhr

    return:
        [B, M, 5], image pixel scale xywhr
    """
    out = gt_bboxes.clone()
    out[:, :, 0:4] = out[:, :, 0:4] * imgsz
    return out

def xywhr2xyxyxyxy_np(xywhr):
    """
    xywhr:
        [G, 5] = cx, cy, w, h, theta

    return:
        [G, 4, 2]
    """
    xywhr = np.asarray(xywhr, dtype=np.float32).reshape(-1, 5)

    cx = xywhr[:, 0:1]
    cy = xywhr[:, 1:2]
    w = xywhr[:, 2:3]
    h = xywhr[:, 3:4]
    theta = xywhr[:, 4:5]

    cos = np.cos(theta)
    sin = np.sin(theta)

    dx = w / 2.0
    dy = h / 2.0

    corners = np.stack(
        [
            np.concatenate([-dx, -dy], axis=1),
            np.concatenate([ dx, -dy], axis=1),
            np.concatenate([ dx,  dy], axis=1),
            np.concatenate([-dx,  dy], axis=1),
        ],
        axis=1,
    )  # [G, 4, 2]

    rot = np.stack(
        [
            np.concatenate([cos, -sin], axis=1),
            np.concatenate([sin,  cos], axis=1),
        ],
        axis=1,
    )  # [G, 2, 2]

    rotated = corners @ np.transpose(rot, (0, 2, 1))

    rotated[:, :, 0] += cx
    rotated[:, :, 1] += cy

    return rotated.astype(np.float32)

def select_candidates_in_rotated_gts_np(anchor_points, gt_bboxes, eps=1e-9):
    """
    anchor_points:
        [A, 2]
    gt_bboxes:
        [G, 5]

    return:
        [G, A] bool
    """
    corners = xywhr2xyxyxyxy_np(gt_bboxes)

    a = corners[:, 0:1, :]
    b = corners[:, 1:2, :]
    d = corners[:, 3:4, :]

    ab = b - a
    ad = d - a

    ap = anchor_points[None, :, :] - a

    norm_ab = (ab * ab).sum(axis=-1)
    norm_ad = (ad * ad).sum(axis=-1)

    ap_dot_ab = (ap * ab).sum(axis=-1)
    ap_dot_ad = (ap * ad).sum(axis=-1)

    return (
        (ap_dot_ab >= -eps)
        & (ap_dot_ab <= norm_ab + eps)
        & (ap_dot_ad >= -eps)
        & (ap_dot_ad <= norm_ad + eps)
    )