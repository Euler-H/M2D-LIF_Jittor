import os
import glob
import argparse
import xml.etree.ElementTree as ET


# =========================
# 类别配置
# =========================
CLASSES = [
    "car",
    "bus",
    "truck",
    "van",
    "feright_car",
]


# 类别别名映射
ALIASES = {
    "feright car": "feright_car",
    "feright_car": "feright_car",
}


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def normalize_class_name(cls_name):
    cls_name = cls_name.strip()
    cls_name = ALIASES.get(cls_name, cls_name)
    return cls_name


def polygon_to_bbox(polygon_node):
    """
    polygon 四点框 -> 水平外接矩形
    """
    xs = []
    ys = []

    for i in range(1, 5):
        x_node = polygon_node.find(f"x{i}")
        y_node = polygon_node.find(f"y{i}")

        if x_node is None or y_node is None:
            raise ValueError("polygon 中缺少 x/y 坐标")

        xs.append(float(x_node.text))
        ys.append(float(y_node.text))

    xmin = min(xs)
    ymin = min(ys)
    xmax = max(xs)
    ymax = max(ys)

    return xmin, ymin, xmax, ymax


def bndbox_to_bbox(bndbox_node):
    """
    VOC bndbox -> 水平矩形
    """
    xmin_node = bndbox_node.find("xmin")
    ymin_node = bndbox_node.find("ymin")
    xmax_node = bndbox_node.find("xmax")
    ymax_node = bndbox_node.find("ymax")

    if (
        xmin_node is None
        or ymin_node is None
        or xmax_node is None
        or ymax_node is None
    ):
        raise ValueError("bndbox 中缺少 xmin/ymin/xmax/ymax")

    xmin = float(xmin_node.text)
    ymin = float(ymin_node.text)
    xmax = float(xmax_node.text)
    ymax = float(ymax_node.text)

    return xmin, ymin, xmax, ymax


def get_bbox_from_object(obj):
    """
    从一个 object 中读取 bbox。

    支持两种格式：
    1. polygon 四点框
    2. bndbox 水平框
    """
    polygon_node = obj.find("polygon")
    if polygon_node is not None:
        return polygon_to_bbox(polygon_node), "polygon"

    bndbox_node = obj.find("bndbox")
    if bndbox_node is not None:
        return bndbox_to_bbox(bndbox_node), "bndbox"

    return None, "none"


def bbox_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    """
    bbox -> YOLOv5 格式
    """
    x_center = (xmin + xmax) / 2.0 / img_w
    y_center = (ymin + ymax) / 2.0 / img_h
    box_w = (xmax - xmin) / img_w
    box_h = (ymax - ymin) / img_h

    return x_center, y_center, box_w, box_h


def convert_one_xml(xml_path, save_dir, classes, ignore_difficult=False):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"[Error] 解析失败: {xml_path}, error: {e}")
        return

    size_node = root.find("size")
    if size_node is None:
        print(f"[Skip] {xml_path}: 缺少 <size>")
        return

    try:
        img_w = int(float(size_node.find("width").text))
        img_h = int(float(size_node.find("height").text))
    except Exception as e:
        print(f"[Skip] {xml_path}: 图像尺寸读取失败, error: {e}")
        return

    yolo_lines = []

    polygon_count = 0
    bndbox_count = 0
    skipped_count = 0

    for obj in root.findall("object"):
        name_node = obj.find("name")
        if name_node is None:
            skipped_count += 1
            continue

        raw_cls_name = name_node.text.strip()
        cls_name = normalize_class_name(raw_cls_name)

        if cls_name not in classes:
            print(f"[Warning] {xml_path}: 未知类别 {raw_cls_name} -> {cls_name}，已跳过")
            skipped_count += 1
            continue

        difficult_node = obj.find("difficult")
        difficult = int(difficult_node.text) if difficult_node is not None else 0

        if ignore_difficult and difficult == 1:
            skipped_count += 1
            continue

        bbox, bbox_type = get_bbox_from_object(obj)

        if bbox is None:
            print(
                f"[Warning] {xml_path}: 目标 {cls_name} 既没有 polygon 也没有 bndbox，已跳过"
            )
            skipped_count += 1
            continue

        if bbox_type == "polygon":
            polygon_count += 1
        elif bbox_type == "bndbox":
            bndbox_count += 1

        xmin, ymin, xmax, ymax = bbox

        # 防止坐标越界
        xmin = clamp(xmin, 0, img_w - 1)
        xmax = clamp(xmax, 0, img_w - 1)
        ymin = clamp(ymin, 0, img_h - 1)
        ymax = clamp(ymax, 0, img_h - 1)

        if xmax <= xmin or ymax <= ymin:
            print(
                f"[Warning] {xml_path}: 非法框，已跳过: "
                f"xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}"
            )
            skipped_count += 1
            continue

        cls_id = classes.index(cls_name)

        x_center, y_center, box_w, box_h = bbox_to_yolo(
            xmin, ymin, xmax, ymax, img_w, img_h
        )

        yolo_lines.append(
            f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"
        )

    xml_name = os.path.basename(xml_path)
    txt_name = os.path.splitext(xml_name)[0] + ".txt"
    txt_path = os.path.join(save_dir, txt_name)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_lines))

    print(
        f"[OK] {xml_path} -> {txt_path}, "
        f"objects: {len(yolo_lines)}, "
        f"polygon: {polygon_count}, "
        f"bndbox: {bndbox_count}, "
        f"skipped: {skipped_count}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert XML annotations to YOLOv5 txt labels. Support polygon and bndbox."
    )

    parser.add_argument(
        "--xml_dir",
        type=str,
        required=True,
        help="XML 标注文件夹路径"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="YOLOv5 txt 标签保存文件夹路径"
    )

    parser.add_argument(
        "--ignore_difficult",
        action="store_true",
        help="是否忽略 difficult=1 的目标"
    )

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    xml_paths = sorted(glob.glob(os.path.join(args.xml_dir, "*.xml")))

    if len(xml_paths) == 0:
        print(f"[Error] 没有找到 XML 文件: {args.xml_dir}")
        return

    print("========== 类别映射关系 ==========")
    for i, cls_name in enumerate(CLASSES):
        print(f"{i}: {cls_name}")
    print("=================================")

    print("========== 类别别名映射 ==========")
    for k, v in ALIASES.items():
        print(f"{k} -> {v}")
    print("=================================")

    for xml_path in xml_paths:
        convert_one_xml(
            xml_path=xml_path,
            save_dir=args.save_dir,
            classes=CLASSES,
            ignore_difficult=args.ignore_difficult,
        )

    print("全部转换完成。")


if __name__ == "__main__":
    main()