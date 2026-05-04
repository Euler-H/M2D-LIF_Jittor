import os
import cv2
import numpy as np

train_txt = "/root/JDet/DV-full/train.txt"

bad = []
large_label = []

with open(train_txt, "r") as f:
    lines = [x.strip() for x in f.readlines() if x.strip()]

print("total:", len(lines))

for i, line in enumerate(lines):
    if i % 500 == 0:
        print("checking", i, line, flush=True)

    parts = line.split()
    img_path = parts[0]

    if not os.path.exists(img_path):
        bad.append((i, img_path, "missing image"))
        continue

    img = cv2.imread(img_path)
    if img is None:
        bad.append((i, img_path, "cv2.imread failed"))
        continue

    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        bad.append((i, img_path, "bad shape"))
        continue

    # 常见 YOLO txt 路径推断：images -> labels, 后缀改 txt
    label_path = img_path
    label_path = label_path.replace("/images/", "/labels/")
    label_path = os.path.splitext(label_path)[0] + ".txt"

    if not os.path.exists(label_path):
        # 如果你的 train.txt 里第二列就是 label，这里按实际改
        continue

    try:
        arr = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 9)
    except Exception as e:
        bad.append((i, label_path, f"label read failed: {e}"))
        continue

    if len(arr) > 128:
        large_label.append((i, img_path, len(arr)))

    if not np.isfinite(arr).all():
        bad.append((i, label_path, "nan or inf label"))
        continue

    cls = arr[:, 0]
    pts = arr[:, 1:]

    if (cls < 0).any() or (cls >= 5).any():
        bad.append((i, label_path, f"bad class range: min={cls.min()}, max={cls.max()}"))

    if (pts < -0.5).any() or (pts > 1.5).any():
        bad.append((i, label_path, f"coords too far: min={pts.min()}, max={pts.max()}"))

print("bad samples:", len(bad))
for x in bad[:100]:
    print(x)

print("large label samples:", len(large_label))
for x in large_label[:100]:
    print(x)