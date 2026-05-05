import os
import shutil
from pathlib import Path


def move_first_n_files(src_dir, dst_dir, n=5000):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    if not src_dir.exists():
        raise FileNotFoundError(f"源文件夹不存在: {src_dir}")

    if not src_dir.is_dir():
        raise NotADirectoryError(f"源路径不是文件夹: {src_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    # 获取源文件夹下的所有文件，不包含子文件夹
    files = [p for p in src_dir.iterdir() if p.is_file()]

    # 按文件名排序，保证“前5000个”是稳定的
    files = sorted(files, key=lambda x: x.name)

    selected_files = files[:n]

    print(f"源文件夹文件总数: {len(files)}")
    print(f"准备迁移文件数: {len(selected_files)}")

    moved_count = 0
    skipped_count = 0

    for file_path in selected_files:
        dst_path = dst_dir / file_path.name

        # 如果目标文件已存在，避免覆盖
        if dst_path.exists():
            print(f"跳过，目标文件已存在: {dst_path}")
            skipped_count += 1
            continue

        shutil.move(str(file_path), str(dst_path))
        moved_count += 1

    print("迁移完成")
    print(f"成功迁移: {moved_count}")
    print(f"跳过文件: {skipped_count}")


if __name__ == "__main__":
    src_folder = r"/root/M2D-LIF_Jittor/DroneVehicle/labels/val"
    dst_folder = r"/root/M2D-LIF_Jittor/5000-DroneVehice/labels/val"

    move_first_n_files(src_folder, dst_folder, n=5000)
