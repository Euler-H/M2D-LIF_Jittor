import os
import argparse


IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


def is_image_file(filename):
    return os.path.splitext(filename)[1].lower() in IMG_EXTS


def make_txt(image_dir, save_txt):
    image_dir = os.path.abspath(image_dir)
    save_txt = os.path.abspath(save_txt)

    image_paths = []

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if is_image_file(file):
                img_path = os.path.join(root, file)
                img_path = os.path.abspath(img_path)
                image_paths.append(img_path)

    image_paths = sorted(image_paths)

    os.makedirs(os.path.dirname(save_txt), exist_ok=True)

    with open(save_txt, "w", encoding="utf-8") as f:
        for path in image_paths:
            f.write(path + "\n")

    print(f"[OK] 共写入 {len(image_paths)} 张图片")
    print(f"[OK] 保存到: {save_txt}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate YOLOv5 train.txt or val.txt with absolute image paths."
    )

    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="图片文件夹路径"
    )

    parser.add_argument(
        "--save_txt",
        type=str,
        required=True,
        help="输出 txt 文件路径，例如 train.txt 或 val.txt"
    )

    args = parser.parse_args()

    make_txt(args.image_dir, args.save_txt)


if __name__ == "__main__":
    main()