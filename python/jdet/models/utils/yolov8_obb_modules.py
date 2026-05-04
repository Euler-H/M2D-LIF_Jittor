import math
import jittor as jt
from jittor import nn


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class SiLU(nn.Module):
    def execute(self, x):
        return x * jt.sigmoid(x)


class Conv(nn.Module):
    """
    YOLOv8 standard Conv: Conv2d + BN + SiLU
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()

        if isinstance(k, list):
            k = k[0]
        if isinstance(s, list):
            s = s[0]

        self.conv = nn.Conv(
            c1,
            c2,
            k,
            s,
            autopad(k, p),
            groups=g,
            bias=False
        )
        self.bn = nn.BatchNorm(c2)
        self.act = SiLU() if act is True else nn.Identity()

    def execute(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """
    YOLOv8 Bottleneck.
    """

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 3, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def execute(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f(nn.Module):
    """
    YOLOv8 C2f block.

    与 YOLOv5 C3 的区别：
    1. C2f 会保留更多中间层输出；
    2. concat 的特征更多；
    3. 梯度流更丰富；
    4. 是 YOLOv8 backbone/neck 的核心模块。
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList([
            Bottleneck(self.c, self.c, shortcut, g, e=1.0)
            for _ in range(n)
        ])

    def execute(self, x):
        y = list(self.cv1(x).split(self.c, dim=1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(jt.concat(y, dim=1))


class SPPF(nn.Module):
    """
    YOLOv8 SPPF.
    用连续 MaxPool 实现快速空间金字塔池化。
    """

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def execute(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(jt.concat([x, y1, y2, y3], dim=1))


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def execute(self, x):
        return jt.concat(x, self.d)


class Upsample(nn.Module):
    """
    Jittor 版 nearest upsample。
    用于 YOLOv8 neck 的上采样。
    """

    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def execute(self, x):
        return nn.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode="nearest"
        )


class DFL(nn.Module):
    """
    Distribution Focal Loss 的积分层。
    将离散分布 [0, ..., reg_max-1] 转成连续距离。
    """

    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max
        self.proj = jt.arange(reg_max).float().view(1, 1, reg_max, 1)

    def execute(self, x):
        # x: [B, 4 * reg_max, A]
        b, c, a = x.shape

        # [B, 4 * reg_max, A] -> [B, 4, reg_max, A]
        x = x.view(b, 4, self.reg_max, a)

        # 对 reg_max 维做 softmax
        x = nn.softmax(x, dim=2)

        # 积分：sum_i p_i * i
        x = (x * self.proj).sum(dim=2)

        # 输出 [B, 4, A]
        return x
    
def dist2rbox(distance, angle, anchor_points):
    """
    YOLOv8-OBB standard rotated box decode.

    distance:
        [..., 4] = l, t, r, b
    angle:
        [..., 1]
    anchor_points:
        [..., 2]

    return:
        [..., 4] = cx, cy, w, h
    """
    lt = distance[..., :2]
    rb = distance[..., 2:]

    cos = jt.cos(angle)
    sin = jt.sin(angle)

    offset = (rb - lt) / 2.0
    xf = offset[..., 0:1]
    yf = offset[..., 1:2]

    x = xf * cos - yf * sin
    y = xf * sin + yf * cos

    xy = jt.concat([x, y], dim=-1) + anchor_points
    wh = lt + rb

    return jt.concat([xy, wh], dim=-1)