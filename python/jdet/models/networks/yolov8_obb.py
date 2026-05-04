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
        y = []

        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [
                    x if j == -1 else y[j] for j in m.f
                ]

            x = m(x)
            y.append(x if m.i in self.save else None)

        return x

    def execute(self, x, targets=None):
        preds = self.forward_once(x)

        if self.is_training():
            if targets is None: # 代表训练阶段没有传标签对象进来，这不是正常训练状态。并非空标签！
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