import math
from copy import deepcopy
from pathlib import Path

import yaml
import jittor as jt
from jittor import nn

from jdet.utils.registry import MODELS
from jdet.utils.general import make_divisible, check_img_size
from jdet.ops.yolo_obb_ops import labels_poly_to_xywhr
from jdet.models.losses.yolov8_obb_loss import YOLOv8OBBLoss

from jdet.models.utils.yolov8_obb_modules import (
    Conv,
    C2f,
    SPPF,
    Concat,
    Upsample,
    DFL,
    dist2rbox,
    Add,
    IN,
    Multiin,
    LIF,
    LIFAdd,
)



class OBB(nn.Module):
    """
    YOLOv8-OBB Head.

    输出内容：
    1. bbox distribution: 4 * reg_max
    2. cls logits: nc
    3. angle logits: 1

    训练阶段输出 raw feature；
    推理阶段输出 decoded prediction。
    """

    stride = None

    def __init__(self, nc=80, ne=1, ch=(), reg_max=16):
        super().__init__()

        self.nc = nc
        self.ne = ne
        self.nl = len(ch)
        self.reg_max = reg_max

        self.no = nc + 4 * reg_max
        self.angle_dim = ne

        self.dfl = DFL(reg_max)

        self.cv2 = nn.ModuleList()
        self.cv3 = nn.ModuleList()
        self.cv4 = nn.ModuleList()
        self.bce = nn.BCEWithLogitsLoss()

        c2 = max(16, ch[0] // 4, 4 * reg_max)
        c3 = max(ch[0], min(nc, 100))
        c4 = max(ch[0] // 4, ne)

        for c in ch:
            self.cv2.append(
                nn.Sequential(
                    Conv(c, c2, 3),
                    Conv(c2, c2, 3),
                    nn.Conv(c2, 4 * reg_max, 1)
                )
            )

            self.cv3.append(
                nn.Sequential(
                    Conv(c, c3, 3),
                    Conv(c3, c3, 3),
                    nn.Conv(c3, nc, 1)
                )
            )

            self.cv4.append(
                nn.Sequential(
                    Conv(c, c4, 3),
                    Conv(c4, c4, 3),
                    nn.Conv(c4, ne, 1)
                )
            )

    def execute(self, x):
        outputs = []

        for i in range(self.nl):
            box = self.cv2[i](x[i])
            cls = self.cv3[i](x[i])
            angle = self.cv4[i](x[i])
            outputs.append(jt.concat([box, cls, angle], dim=1))

        if self.is_training():
            return outputs

        return self.decode(outputs)
    
    def bias_init(self):
        """
        YOLOv8-style bias initialization.

        目的：
            让分类分支初始预测为低置信度，
            避免大量背景位置 BCE loss 爆炸。
        """
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            # box branch
            a[-1].bias.assign(jt.ones_like(a[-1].bias) * 1.0)

            # cls branch
            # 先验：每张 640 图像每类大约 5 个目标
            prior = 5.0 / self.nc / (640.0 / float(s)) ** 2
            prior = max(prior, 1e-5)
            bias_value = math.log(prior / (1.0 - prior))
            b[-1].bias.assign(jt.ones_like(b[-1].bias) * bias_value)

        for a in self.cv4:
            # angle branch，先保持中性
            a[-1].bias.assign(jt.zeros_like(a[-1].bias))

    def decode(self, outputs):
        """
        推理阶段解码。

        输出格式：
        [B, N, 5 + nc]
        其中 5 为 cx, cy, w, h, theta。
        """

        z = []

        for i, out in enumerate(outputs):
            b, _, h, w = out.shape
            stride = self.stride[i]

            box = out[:, :4 * self.reg_max, :, :]
            cls = out[:, 4 * self.reg_max:4 * self.reg_max + self.nc, :, :]
            angle = out[:, 4 * self.reg_max + self.nc:, :, :]

            box = box.view(b, 4 * self.reg_max, h * w)
            box = self.dfl(box).permute(0, 2, 1)  # [B, A, 4]

            cls = cls.view(b, self.nc, h * w).permute(0, 2, 1)
            cls = jt.sigmoid(cls)

            angle = angle.view(b, self.ne, h * w).permute(0, 2, 1)
            angle = (jt.sigmoid(angle) - 0.25) * math.pi

            anchor_points = self.make_grid(w, h).view(1, h * w, 2) + 0.5

            rbox = dist2rbox(
                box,
                angle,
                anchor_points,
            )

            rbox = rbox * stride

            pred = jt.concat([rbox, angle, cls], dim=-1)
            z.append(pred)

        return jt.concat(z, dim=1)

    @staticmethod
    def make_grid(nx, ny):
        yv, xv = jt.meshgrid([
            jt.index((ny,), dim=0),
            jt.index((nx,), dim=0)
        ])
        return jt.stack((xv, yv), 2).float()


@MODELS.register_module()
class YOLOv8OBB(nn.Module):
    """
    JDet version YOLOv8-OBB.

    第一阶段目标：
    1. 能构建模型；
    2. 能完成 forward；
    3. 后续再接 OBB loss 和 DroneVehicle dataset。
    """

    def __init__(
        self,
        cfg,
        ch=3,
        nc=80,
        imgsz=640,
        conf_thres=0.25,
        iou_thres=0.45,
        scale=None,
        assigner_min_pos_score=0.05,
        distill_layer_ids=None,
    ):
        super().__init__()

        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding="utf-8") as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)

        # Support Ultralytics-style `scales` while keeping legacy YAMLs with
        # explicit depth_multiple/width_multiple compatible. This also makes the
        # active model scale explicit, avoiding accidental n/m/l checkpoint mismatch.
        if "scales" in self.yaml:
            if scale is None:
                scale = self.yaml.get("scale", None) or next(iter(self.yaml["scales"].keys()))
            if scale not in self.yaml["scales"]:
                raise ValueError(f"Unknown YOLOv8 scale {scale}. Available: {list(self.yaml['scales'].keys())}")
            gd, gw, _max_channels = self.yaml["scales"][scale]
            self.yaml["depth_multiple"] = gd
            self.yaml["width_multiple"] = gw
            self.yaml["scale"] = scale

        ch = self.yaml["ch"] = self.yaml.get("ch", ch)
        self.scale = self.yaml.get("scale", scale)
        print("YOLOv8-OBB active scale:", self.scale,
              "depth_multiple=", self.yaml.get("depth_multiple"),
              "width_multiple=", self.yaml.get("width_multiple"))

        if nc and nc != self.yaml["nc"]:
            print("Overriding model.yaml nc=%g with nc=%g" % (self.yaml["nc"], nc))
            self.yaml["nc"] = nc

        self.nc = self.yaml["nc"]
        self.distill_layer_ids = set(distill_layer_ids or [])
        self.last_distill_features = {}
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])

        m = self.model[-1]
        if isinstance(m, OBB):
            s = 256
            dummy = jt.zeros((1, ch, s, s))
            feats = self.forward_once(dummy)
            if isinstance(feats, list):
                m.stride = jt.array([s / x.shape[-2] for x in feats]).int()
            else:
                # execute once with training mode may return head outputs
                self.train()
                outs = self.forward_once(dummy)
                m.stride = jt.array([s / x.shape[-2] for x in outs]).int()

            self.stride = m.stride
            print("YOLOv8-OBB strides:", self.stride.tolist())
            # 初始化检测头 bias
            m.bias_init()
        
        self.criterion = YOLOv8OBBLoss(
            nc=self.nc,
            reg_max=self.model[-1].reg_max,
            strides=[int(x) for x in self.stride.tolist()],
            imgsz=imgsz,
            assigner_min_pos_score=assigner_min_pos_score,
        )

        self.initialize_weights()

    def forward_once(self, x):
        y = [] # 保存中间层输出

        for m in self.model:
            if m.f != -1: # 输入不是来自上一层
                
                # 构造输入 list
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            
            x = m(x)
            
#######################################################################################

            # Jittor 这里没有用 hook，直接在前向过程中保存指定层输出
            # 根据初始化传入的 distill_layer_ids 来保存指定层特征
            # 具体实现时，会分为 student 和 teacher
            if m.i in self.distill_layer_ids:
                self.last_distill_features[m.i] = x

#######################################################################################

            y.append(x if m.i in self.save else None) 

        return x

    def execute(self, x, targets=None):
        preds = self.forward_once(x)

        if self.is_training():
            if targets is None:
                print("[WARN] YOLOv8OBB got targets=None during training")
                zero = preds[0].sum() * 0.0
                return dict(
                    box_loss=zero,
                    cls_loss=zero,
                    dfl_loss=zero,
                    angle_loss=zero,
                )

            # if not hasattr(self, "_debug_printed"):
            #     self._debug_printed = True
            #     print("[DEBUG] targets shape:", targets.shape)
            #     jt.sync_all(True)
            #     print("[DEBUG] targets first rows:", targets.numpy()[:10])

            return self.criterion(preds, targets)

        return preds

    def initialize_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm):
                m.eps = 1e-3
                m.momentum = 0.03
                
    def simple_obb_loss(self, preds, targets_np):
        """
        targets_np:
            numpy array, [M, 7]
            batch_id, cls, cx, cy, w, h, theta
        """

        device_zero = preds[0].sum() * 0.0

        cls_loss = device_zero
        angle_loss = device_zero

        if targets_np is None or targets_np.shape[0] == 0:
            return dict(
                loss=device_zero,
                box_loss=device_zero,
                cls_loss=device_zero,
                dfl_loss=device_zero,
                angle_loss=device_zero,
            )

        head = self.model[-1]
        nc = head.nc
        reg_max = head.reg_max

        for row in targets_np:
            bidx = int(row[0])
            cls_id = int(row[1])
            cx, cy, w, h, theta = row[2:7]

            area = w * h

            if area < 0.02:
                level = 0
            elif area < 0.08:
                level = 1
            else:
                level = 2

            p = preds[level]
            _, _, gh, gw = p.shape

            gx = int(cx * gw)
            gy = int(cy * gh)

            gx = max(0, min(gx, gw - 1))
            gy = max(0, min(gy, gh - 1))

            cls_start = 4 * reg_max
            cls_end = cls_start + nc

            cls_logits = p[bidx:bidx + 1, cls_start:cls_end, gy:gy + 1, gx:gx + 1]
            cls_logits = cls_logits.view(1, nc)

            cls_target = jt.zeros((1, nc))
            cls_target[:, cls_id] = 1.0

            cls_loss = cls_loss + nn.binary_cross_entropy_with_logits(cls_logits, cls_target)

            angle_logit = p[bidx:bidx + 1, cls_end:cls_end + 1, gy:gy + 1, gx:gx + 1]
            angle_pred = (jt.sigmoid(angle_logit) - 0.25) * math.pi

            theta_target = jt.array([[[[theta]]]]).float()
            angle_loss = angle_loss + jt.abs(angle_pred - theta_target).mean()

        n = max(float(targets_np.shape[0]), 1.0)

        cls_loss = cls_loss / n
        angle_loss = angle_loss / n

        box_loss = device_zero
        dfl_loss = device_zero

        total_loss = cls_loss + 0.1 * angle_loss

        return dict(
            loss=total_loss,
            box_loss=box_loss,
            cls_loss=cls_loss,
            dfl_loss=dfl_loss,
            angle_loss=angle_loss,
        )

def parse_model(d, ch):
    print("\n%3s%18s%3s%10s  %-40s%-30s" %
          ("", "from", "n", "params", "module", "arguments"))

    nc = d["nc"]
    gd = d["depth_multiple"]
    gw = d["width_multiple"]

    layers = []
    save = []
    c2 = ch[-1]

    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        m = eval(m) if isinstance(m, str) else m

        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a
            except Exception:
                pass

        n = max(round(n * gd), 1) if n > 1 else n

        if m in [Conv, C2f, SPPF]:
            c1 = ch[f]
            c2 = args[0]
            c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]

            if m is C2f:
                args.insert(2, n)
                n = 1

        elif m is Concat:
            c2 = sum([ch[x] for x in f])

        elif m is Upsample:
            c2 = ch[f]

        elif m is IN:
            c2 = ch[f]

        elif m is Multiin:
            c2 = 3

        elif m is LIF:
            c2 = 1

        elif m is Add:
            c2 = ch[f[0]]

        elif m is LIFAdd:
            c2 = ch[f[0]]

        elif m is OBB:
            args.append([ch[x] for x in f])
            c2 = None

        else:
            c2 = ch[f]

        module = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)

        t = str(m)[8:-2].replace("__main__.", "")
        np = sum([x.numel() for x in module.parameters()])

        module.i = i
        module.f = f
        module.type = t
        module.np = np

        print("%3s%18s%3s%10.0f  %-40s%-30s" %
              (i, f, n, np, t, args))

        save.extend(
            x % i for x in ([f] if isinstance(f, int) else f)
            if x != -1
        )

        layers.append(module)

        if i == 0:
            ch = []

        if c2 is not None:
            ch.append(c2)
        else:
            ch.append(ch[f[0]])

    return nn.Sequential(*layers), sorted(save)


# 仿BN，省略了某些细节，但是训练效果不是很好
# # 这一步的目的是让 PKD 更关注空间响应关系，而不是特征幅值大小。
# def _feature_standardize(x, eps=1e-5):
#     """BatchNorm2d(affine=False)-like normalization used before PKD/CAD."""
#     mean = x.mean([0, 2, 3], keepdims=True) # 在batch，height，width上求均值，保留通道维度
#     var = ((x - mean) * (x - mean)).mean([0, 2, 3], keepdims=True) # 计算每个通道的方差
#     return (x - mean) / jt.sqrt(var + eps) # 输出标准化后的输出


# PKD 皮尔逊相关蒸馏
# def _pkd_loss_jt(y_s, y_t, eps=1e-6):
#     assert len(y_s) == len(y_t)
#     losses = [] # 保留每个尺度的 PKD loss
#     for s, t in zip(y_s, y_t): # 逐层计算蒸馏损失
#         s = s.float32()

#         # teacher 特征停止梯度，不能被 student 反传更新 对应 pytorch 源码 t = t.detach()
#         t = t.stop_grad().float32() 

#         n, c, h, w = s.shape

#         # [B, C, H, W] -> [B, C, H*W]：PKD 是在每个样本、每个通道内部计算空间响应相关性
#         s_flat = s.reshape(n, c, -1)
#         t_flat = t.reshape(n, c, -1)

#         # 对每个通道的空间响应去均值
#         sm = s_flat - s_flat.mean(dim=-1, keepdims=True)
#         tm = t_flat - t_flat.mean(dim=-1, keepdims=True)

#         num = (sm * tm).sum(dim=-1) # 皮尔逊相关系数分子

#         den = jt.sqrt((sm * sm).sum(dim=-1) * (tm * tm).sum(dim=-1) + eps) # 分母

#         losses.append(1.0 - (num / den).mean()) # 相关性越高，num / den 越接近 1，loss 越小

#     return sum(losses) # 三尺度相加

# CWD 蒸馏损失
def _cwd_loss_jt(y_s, y_t, tau=1.0, eps=1e-12):
    """
    Channel-wise Distillation loss.

    Aligns with PyTorch implementation:

        softmax_pred_T = softmax(t.view(-1, H*W) / tau, dim=1)
        cost = sum(
            softmax_pred_T * log_softmax(t / tau)
            - softmax_pred_T * log_softmax(s / tau)
        ) * tau^2
        loss = cost / (N * C)

    Args:
        y_s: list of student features, each [N, C, H, W]
        y_t: list of teacher features, each [N, C, H, W]
        tau: temperature
    """
    assert len(y_s) == len(y_t)

    losses = []

    for s, t in zip(y_s, y_t):
        s = s.float32()
        t = t.stop_grad().float32()

        assert s.shape == t.shape, "CWD requires student and teacher features to have the same shape."

        n, c, h, w = s.shape

        s_flat = s.reshape(n * c, h * w) / float(tau)
        t_flat = t.reshape(n * c, h * w) / float(tau)

        # teacher spatial probability distribution: [N*C, H*W]
        softmax_t = nn.softmax(t_flat, dim=1)

        # log_softmax(x) = x - logsumexp(x)
        log_softmax_t = t_flat - jt.log(jt.exp(t_flat).sum(1, keepdims=True) + eps)
        log_softmax_s = s_flat - jt.log(jt.exp(s_flat).sum(1, keepdims=True) + eps)

        cost = (softmax_t * (log_softmax_t - log_softmax_s)).sum()
        cost = cost * (float(tau) ** 2)

        losses.append(cost / float(c * n))

    return sum(losses)


def _pkd_loss_jt(y_s, y_t, eps=1e-6):
    assert len(y_s) == len(y_t)
    losses = []

    for s, t in zip(y_s, y_t):
        s = s.float32()
        t = t.stop_grad().float32()

        n, c, h, w = s.shape

        s_flat = s.reshape(n, c, -1)
        t_flat = t.reshape(n, c, -1)

        sm = s_flat - s_flat.mean(-1, keepdims=True)
        tm = t_flat - t_flat.mean(-1, keepdims=True)

        num = (sm * tm).sum(-1)
        den = jt.sqrt((sm * sm).sum(-1) * (tm * tm).sum(-1) + eps)

        losses.append(1.0 - (num / den).mean())

    return sum(losses)


# SimAM 注意力“图”
# def _simam_attention_map_jt(x, e_lambda=1e-4):
#     x = x.float32()
#     b, c, h, w = x.shape
#     n = max(h * w - 1, 1) # SimAM 里用空间位置数量减 1 作为归一化项。
#     x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdims=True)) ** 2 # 计算每个空间位置相对通道空间均值的平方差。
#     denom = 4.0 * (x_minus_mu_square.sum(dim=[2, 3], keepdims=True) / float(n) + e_lambda)
#     y = x_minus_mu_square / denom + 0.5
#     return jt.sigmoid(y)

def _simam_attention_map_jt(x, e_lambda=1e-4):
    x = x.float32()
    b, c, h, w = x.shape

    n = max(h * w - 1, 1)

    mu = x.mean([2, 3], keepdims=True)
    x_minus_mu_square = (x - mu) * (x - mu)

    denom = 4.0 * (
        x_minus_mu_square.sum([2, 3], keepdims=True) / float(n) + e_lambda
    )

    y = x_minus_mu_square / denom + 0.5
    return jt.sigmoid(y)


# 同模态蒸馏损失
class M2DFeatureLoss(nn.Module):
    """FeatureLoss(PKD) aligned with M2D-LIF PyTorch implementation."""
    def __init__(self, channels_s, channels_t, loss_type="PKD", loss_weight=1.0):
        super().__init__()
        self.loss_type = str(loss_type).upper()
        self.loss_weight = float(loss_weight)
        
        # 对齐 student 和 teacher 特征
        self.align = nn.ModuleList([ 
            nn.Conv(cs, ct, 1, 1, 0)
            for cs, ct in zip(channels_s, channels_t)
        ])

        self.norm_t = nn.ModuleList([
                nn.BatchNorm(ct, affine=False)
                for ct in channels_t
            ])

        if self.loss_type not in ("PKD", "CWD"):
            raise NotImplementedError(
                "Unsupported M2D feature distillation loss_type: {}".format(loss_type)
            )

    def execute(self, y_s, y_t):
        assert len(y_s) == len(y_t)

        ss, tt = [], []

        for i, (s, t) in enumerate(zip(y_s, y_t)): # 逐尺度处理
            s = self.align[i](s)

            s = self.norm_t[i](s)
            t = self.norm_t[i](t.stop_grad())

            ss.append(s) 
            tt.append(t) 
        
        if self.loss_type == "PKD":
            loss = _pkd_loss_jt(ss, tt)
        elif self.loss_type == "CWD":
            loss = _cwd_loss_jt(ss, tt)
        else:
            raise NotImplementedError

        return loss * self.loss_weight



class M2DCrossAttentionLoss(nn.Module):
    """Cross-modality SimAM-masked PKD used by M2D-LIF."""
    def __init__(self, channels_s, channels_t, loss_weight=1.0):
        super().__init__()
        self.loss_weight = float(loss_weight)

        self.align = nn.ModuleList([
            nn.Conv(cs, ct, 1, 1, 0)
            for cs, ct in zip(channels_s, channels_t)
        ])

        self.norm_t = nn.ModuleList([
                nn.BatchNorm(ct, affine=False)
                for ct in channels_t
            ])

    def execute(self, y_s, y_t):
        ss, tt = [], []
        for i, (s, t) in enumerate(zip(y_s, y_t)):
            s = self.align[i](s)

            s = self.norm_t[i](s)
            t = self.norm_t[i](t.stop_grad())

            attn = _simam_attention_map_jt(t).stop_grad()
            ss.append(s * attn)
            tt.append(t * attn)
        return _pkd_loss_jt(ss, tt) * self.loss_weight


@MODELS.register_module()
class YOLOv8OBBM2DLIF(YOLOv8OBB):
    """
    Jittor reproduction of the M2D-LIF outer YOLOv8-OBB distillation model.

    Student: dual-modal YOLOv8-OBB with LIF fusion, scale=n by default.
    Teachers: two frozen mono-modal YOLOv8-OBB models, one RGB and one IR.
    Distillation:
        L_IM = PKD(RGB_s, RGB_t) + PKD(IR_s, IR_t)
        L_CM = CAD(IR_s, RGB_t) + CAD(RGB_s, IR_t)
    LIF illumination supervision follows the PyTorch code path using max(RGB) pooled to stride 8.
    """
    def __init__(
        self,
        cfg,
        teacher_rgb_cfg,
        teacher_ir_cfg,
        teacher_rgb_ckpt=None,
        teacher_ir_ckpt=None,
        ch=6,
        nc=80,
        imgsz=640,
        scale='n',
        distill_weight=0.8,
        loss_type="CWD",
        lif_weight=1.3,
        total_epochs=100,
        **kwargs,
    ):
        # LIF student feature indices match the original M2D-LIF scale=n YAML.
        super().__init__(cfg=cfg, ch=ch, nc=nc, imgsz=imgsz, scale=scale,
                         distill_layer_ids=[12, 13, 17, 18, 22, 23], **kwargs)
        self.distill_weight = float(distill_weight)
        self.lif_weight = float(lif_weight)
        self.total_epochs = int(total_epochs)
        self.current_epoch = 0
        teacher_rgb = YOLOv8OBB(cfg=teacher_rgb_cfg, ch=3, nc=nc, imgsz=imgsz, scale=scale,
                                distill_layer_ids=[4, 6, 8], **kwargs)
        teacher_ir = YOLOv8OBB(cfg=teacher_ir_cfg, ch=3, nc=nc, imgsz=imgsz, scale=scale,
                               distill_layer_ids=[4, 6, 8], **kwargs)
        def _extract_state_dict_from_ckpt(ckpt):
            """
            Compatible with:
            1. pure state_dict: {name: jt.Var}
            2. runner checkpoint: {"model": state_dict or Module, "ema": state_dict or Module, ...}
            3. saved Module / Sequential
            """
            if isinstance(ckpt, dict):
                # Prefer EMA if it is available and valid.
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


        def _safe_load_teacher(model, ckpt_path, name="teacher"):
            print(f"[M2D-LIF] Loading {name} from: {ckpt_path}")

            ckpt = jt.load(ckpt_path)
            state = _extract_state_dict_from_ckpt(ckpt)
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
                        reason = f"not_tensor_value type={type(v)}"
                    else:
                        reason = f"shape_mismatch ckpt={tuple(v.shape)} model={tuple(cur[k].shape)}"
                    skipped.append((k, reason))

            model.load_parameters(matched)

            print(f"[M2D-LIF] {name}: loaded {len(matched)}/{len(cur)} parameters.")
            if skipped:
                print(f"[M2D-LIF] {name}: skipped {len(skipped)} parameters. First 10:")
                for item in skipped[:10]:
                    print("   ", item)

            if len(matched) == 0:
                raise RuntimeError(
                    f"[M2D-LIF] No parameters were loaded into {name}. "
                    f"Please check whether checkpoint scale/model type matches YOLOv8n-OBB."
                )

            return matched, skipped


        if teacher_rgb_ckpt:
            _safe_load_teacher(teacher_rgb, teacher_rgb_ckpt, "teacher_rgb")

        if teacher_ir_ckpt:
            _safe_load_teacher(teacher_ir, teacher_ir_ckpt, "teacher_ir")
        teacher_rgb.eval()
        teacher_ir.eval()
        for pp in teacher_rgb.parameters():
            pp.stop_grad()
        for pp in teacher_ir.parameters():
            pp.stop_grad()
        # Bypass Module.__setattr__ registration so frozen teachers are not included in the student optimizer.
        object.__setattr__(self, 'teacher_rgb', teacher_rgb)
        object.__setattr__(self, 'teacher_ir', teacher_ir)
        channels = [64, 128, 256]

        self.loss_type = str(loss_type).upper()

        self.d_rgb = M2DFeatureLoss(channels, channels, loss_type=self.loss_type)
        self.d_ir = M2DFeatureLoss(channels, channels, loss_type=self.loss_type)
        self.c_rgb_to_ir = M2DCrossAttentionLoss(channels, channels)  # student IR vs teacher RGB
        self.c_ir_to_rgb = M2DCrossAttentionLoss(channels, channels)  # student RGB vs teacher IR
        self.pool_for_lif = nn.Pool(kernel_size=8, stride=8, op='mean')

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def _epoch_distill_decay(self):
        # Same cosine decay as M2D-LIF PyTorch: from 1.0 to 0.1 over training.
        epochs = max(float(self.total_epochs), 1.0)
        return ((1.0 - math.cos(float(self.current_epoch) * math.pi / epochs)) / 2.0) * (0.1 - 1.0) + 1.0

    def _ordered_features(self, model, ids):
        return [model.last_distill_features[i] for i in ids]

    def execute(self, x, targets=None):
        # Student forward + detection loss.
        self.last_distill_features = {}
        losses = super().execute(x, targets)
        if not self.is_training():
            return losses
        rgb = x[:, :3, :, :]
        ir = x[:, 3:6, :, :]

        # LIF illumination supervision: gt = AvgPool(max RGB channel), weight = model[2](RGB).
        img_v = jt.max(rgb, 1, keepdims=True)
        gt = self.pool_for_lif(img_v).stop_grad()
        pred_b = self.model[2](rgb)
        li_loss = jt.abs(pred_b - gt).mean() * self.lif_weight

        # Frozen teacher forward only for feature caches.
        self.teacher_rgb.last_distill_features = {}
        self.teacher_ir.last_distill_features = {}
        self.teacher_rgb.eval()
        self.teacher_ir.eval()
        _ = self.teacher_rgb.forward_once(rgb)
        _ = self.teacher_ir.forward_once(ir)

        s_rgb = self._ordered_features(self, [12, 17, 22])
        s_ir = self._ordered_features(self, [13, 18, 23])
        t_rgb = self._ordered_features(self.teacher_rgb, [4, 6, 8])
        t_ir = self._ordered_features(self.teacher_ir, [4, 6, 8])

        d_loss = self.d_rgb(s_rgb, t_rgb) + self.d_ir(s_ir, t_ir)
        c_loss = self.c_ir_to_rgb(s_rgb, t_ir) + self.c_rgb_to_ir(s_ir, t_rgb)
        kd_w = self.distill_weight * self._epoch_distill_decay()
        d_loss = d_loss * kd_w
        c_loss = c_loss * kd_w

        losses['illumination_loss'] = li_loss
        losses['m2d_d_loss'] = d_loss
        losses['m2d_c_loss'] = c_loss

        return losses
