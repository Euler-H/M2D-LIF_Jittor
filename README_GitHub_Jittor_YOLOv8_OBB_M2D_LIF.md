# Jittor YOLOv8-OBB & M2D-LIF

This repository contains a **Jittor-based reproduction and extension** of YOLOv8-OBB for oriented object detection, together with a Jittor implementation of the M2D-LIF-style multimodal distillation training pipeline.

> **Note**
>
> This repository is not a complete one-click experimental package. It mainly contains the core implementation code, training/validation/visualization scripts, and configuration files used in the reproduction. Datasets, large checkpoints, intermediate training outputs, and some local experiment logs are not included and should be prepared separately.

## 1. Project Overview

The project focuses on reproducing and extending a YOLOv8-OBB detection framework under the Jittor deep learning framework, and further adapting it to a dual-modal visible-infrared object detection pipeline.

The main components are:

- Jittor implementation of YOLOv8-OBB.
- Single-modal teacher training for DroneVehicle.
- M2D-LIF-style dual-modal student training.
- Oriented bounding box validation scripts.
- Prediction visualization scripts.
- Jittor `.pkl` checkpoint usage for validation and inference.
- Auxiliary tools for dataset checking and model debugging.

The code is developed based on the JDet project structure and extends it with YOLOv8-OBB and multimodal distillation modules.

## 2. Current Repository Status

This repository currently provides the core code required for:

| Component | Status |
| --- | --- |
| Jittor YOLOv8-OBB model | Provided |
| Single-modal YOLOv8-OBB training | Provided |
| Single-modal OBB validation | Provided |
| Single-modal prediction visualization | Provided |
| M2D-LIF-style dual-modal training | Provided |
| M2D-LIF OBB validation | Provided |
| M2D-LIF prediction visualization | Provided |
| Dataset files | Not included |
| Large checkpoints | Not included |
| Full experiment logs | Not included |
| Complete one-click reproduction pipeline | Not included |

Therefore, before running the code, the user should prepare the dataset paths, checkpoint paths, and working directories according to the local environment.

## 3. Tested Environment

The following environment was used during development and testing.

### Operating System

```bash
Ubuntu 22.04.1 LTS
Linux kernel: 5.15.0-97-generic
```

### Python

```bash
Python 3.10.20
Conda environment: Jittor_py310
Python path: /root/miniconda3/envs/Jittor_py310/bin/python
pip: 26.0.1
```

### GPU and CUDA

```bash
GPU: NVIDIA GeForce RTX 4090
GPU memory: 24564 MiB
NVIDIA Driver: 580.76.05
CUDA runtime reported by nvidia-smi: 13.0
NVCC: 11.8.89
CUDA arch: sm_89
```

### Compiler

```bash
gcc 11.3.0
g++ 11.3.0
```

### Main Python Dependencies

```bash
jittor==1.3.10.0
numpy==1.23.5
opencv-python==4.11.0.86
matplotlib==3.10.9
pillow==12.2.0
pycocotools==2.0.11
PyYAML==6.0.3
scipy==1.15.3
shapely==2.1.2
tensorboardX==2.6.5
terminaltables==3.1.10
tqdm==4.67.3
nvidia-ml-py3==7.352.0
```

PyTorch and Ultralytics were not installed in the tested Jittor environment. This repository mainly runs under Jittor.

## 4. Installation

### 4.1 Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

### 4.2 Create Conda Environment

```bash
conda create -n Jittor_py310 python=3.10 -y
conda activate Jittor_py310
```

### 4.3 Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available or needs to be regenerated, install the core dependencies manually:

```bash
pip install \
  jittor==1.3.10.0 \
  numpy==1.23.5 \
  opencv-python==4.11.0.86 \
  matplotlib \
  pillow \
  pycocotools \
  PyYAML \
  scipy \
  shapely \
  tensorboardX \
  terminaltables \
  tqdm \
  nvidia-ml-py3
```

### 4.4 Install the Project in Editable Mode

From the repository root:

```bash
pip install -e .
```

## 5. Dataset Preparation

The project uses the DroneVehicle-style visible-infrared dataset layout.

A typical dataset directory is:

```text
5000-DroneVehice/
├── images/
│   ├── train/
│   └── val/
├── images_ir/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
├── train.txt
└── val.txt
```

The expected OBB label format is YOLO-style polygon annotation:

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

where the four points are normalized polygon coordinates.

Example class names:

```text
car
truck
bus
van
freight_car
```

Please modify the dataset paths in the config files before training.

## 6. Repository Structure

The project follows the JDet-style layout. Important directories include:

```text
.
├── configs/                         # Original JDet configs
├── docs/                            # JDet documentation
├── projects/
│   ├── yolo/                        # YOLO-related baseline project
│   └── yolov8_obb/                  # Jittor YOLOv8-OBB implementation
│       ├── configs/
│       └── run_net.py
├── python/
│   └── jdet/                        # Core JDet/Jittor detection framework
├── tools/                           # Utility scripts
├── jittor-yolov8/                   # Extended YOLOv8-OBB/M2D-LIF code branch
│   ├── projects/
│   │   └── yolov8_obb/
│   ├── python/
│   │   └── jdet/
│   ├── tools/
│   └── val_jittor_m2dlif_obb_map.py
├── val_jittor_single_obb_map.py     # Single-modal OBB validation script
├── README.md
└── LICENSE.txt
```

## 7. Single-Modal YOLOv8-OBB

### 7.1 Training

Run the following command from the repository root:

```bash
python projects/yolov8_obb/run_net.py \
  --config-file projects/yolov8_obb/configs/yolov8n_obb_dronevehicle_teacher.py
```

This trains a single-modal YOLOv8-OBB teacher model.

Before running, check the following items in the config file:

- Dataset root.
- Train/validation image paths.
- Label paths.
- Number of classes.
- Class names.
- Batch size.
- Image size.
- Work directory.
- Checkpoint save path.

### 7.2 Validation

Example command for validating a single-modal checkpoint:

```bash
PYTHONPATH=/root/JDet/python python val_jittor_single_obb_map.py \
  --weights /root/JDet/work_dirs/RGB-full.pkl \
  --source /root/JDet/5000-DroneVehice/images/val \
  --label-dir /root/JDet/5000-DroneVehice/labels/val \
  --cfg /root/JDet/projects/yolov8_obb/configs/yolo_configs/yolov8n_obb.yaml \
  --imgsz 640 \
  --scale n \
  --nc 5 \
  --conf 0.25 \
  --iou 0.7 \
  --use-cuda 1
```

### 7.3 Visualization

Example command for visualizing single-modal predictions:

```bash
PYTHONPATH=/root/JDet/python python tools/vis_single_yolov8_obb_pred.py \
  --weights /root/JDet/work_dirs/RGB-full.pkl \
  --source /root/JDet/5000-DroneVehice/images/val \
  --cfg /root/JDet/projects/yolov8_obb/configs/yolo_configs/yolov8n_obb.yaml \
  --out-dir /root/JDet/runs/vis_single_rgb \
  --imgsz 640 \
  --scale n \
  --nc 5 \
  --conf 0.25 \
  --iou 0.7 \
  --use-cuda 1 \
  --names "car,truck,bus,van,freight_car"
```

## 8. M2D-LIF Dual-Modal Training

The M2D-LIF branch is placed under `jittor-yolov8/`.

### 8.1 Training

Example command:

```bash
PYTHONPATH=/root/JDet/jittor-yolov8/python python /root/JDet/jittor-yolov8/projects/yolov8_obb/run_net.py \
  --config-file /root/JDet/jittor-yolov8/projects/yolov8_obb/configs/yolov8n_obb_DroneVehicle_M2D-LIF.py
```

Before training, check the config file carefully, especially:

- RGB image root.
- IR image root.
- Label root.
- Teacher checkpoint paths.
- Student model YAML.
- Number of classes.
- Distillation loss weights.
- Batch size.
- Work directory.
- Checkpoint save directory.

### 8.2 Validation

Example command:

```bash
PYTHONPATH=/root/JDet/jittor-yolov8/python python /root/JDet/jittor-yolov8/val_jittor_m2dlif_obb_map.py \
  --weights /root/JDet/work_dirs/DroneVehicle_M2D-LIF/checkpoints/best.pkl \
  --source /root/JDet/test/images \
  --ir-root /root/JDet/test/images_ir \
  --label-dir /root/JDet/test/labels \
  --cfg /root/JDet/jittor-yolov8/projects/yolov8_obb/configs/yolo_configs/yolov8n_LIF_obb.yaml \
  --imgsz 640 \
  --scale n \
  --nc 5 \
  --conf 0.25 \
  --iou 0.7 \
  --use-cuda 1
```

### 8.3 Visualization

Example command:

```bash
PYTHONPATH=/root/JDet/jittor-yolov8/python python tools/vis_m2d_lif_obb_pred.py \
  --weights /root/JDet/work_dirs/DroneVehicle_M2D-LIF/checkpoints/best.pkl \
  --source /root/JDet/test/images \
  --view-modal concat \
  --out-dir /root/JDet/jittor-yolov8/runs/vis_concat
```

The `--view-modal concat` option is used to visualize the dual-modal prediction result in a concatenated view.

## 9. Checkpoint Notes

Jittor checkpoints are saved as `.pkl` files.

Example checkpoint paths used during experiments:

```text
/root/JDet/work_dirs/RGB-full.pkl
/root/JDet/work_dirs/DroneVehicle_M2D-LIF/checkpoints/best.pkl
```

These checkpoint files are usually large and are not recommended to be committed directly to GitHub.

Suggested practice:

```text
work_dirs/
runs/
weights/
*.pkl
*.pt
*.pth
```

should be excluded by `.gitignore`.

## 10. Core Implementation Highlights

### 10.1 YOLOv8-OBB in Jittor

The project implements the main YOLOv8-OBB components in Jittor, including:

- YOLOv8-style backbone and neck.
- OBB detection head.
- Rotated bounding box decoding.
- DFL-based bounding box regression.
- Task-aligned assignment.
- OBB loss calculation.
- OBB post-processing and NMS.
- OBB mAP validation.

### 10.2 M2D-LIF-Style Distillation

The M2D-LIF branch extends the single-modal detector into a dual-modal training pipeline. It supports:

- RGB and IR input branches.
- Teacher checkpoint loading.
- Student model training.
- Feature-level distillation.
- Dual-modal prediction and visualization.
- Jittor-based OBB validation.

## 11. Common Issues

### 11.1 Jittor Compiles Operators at First Run

At the first run, Jittor may spend time compiling operators and external CUDA kernels. This is normal.

Typical logs include:

```text
Compiling jittor_core...
Found nvcc...
Create cache dir...
```

The following runs are usually faster after the cache is built.

### 11.2 PyTorch or Ultralytics Not Found

The tested Jittor environment does not include PyTorch or Ultralytics:

```text
PyTorch import failed: ModuleNotFoundError("No module named 'torch'")
Ultralytics import failed: ModuleNotFoundError("No module named 'ultralytics'")
```

This is acceptable for running the Jittor implementation. Install PyTorch/Ultralytics only if you need cross-framework comparison or checkpoint conversion.

### 11.3 Dataset Path Error

If the dataset cannot be found, check:

- `--source`
- `--label-dir`
- `--ir-root`
- dataset paths in config files
- whether image names match label names

### 11.4 OpenCV Save Error

If `cv2.imwrite()` reports that it cannot find a writer for the specified extension, check whether the output filename has a valid image extension such as:

```text
.jpg
.png
.jpeg
```

## 12. GitHub Upload Suggestions

Before uploading to GitHub, remove cache and temporary files:

```bash
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name ".ipynb_checkpoints" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete
```

Recommended `.gitignore` entries:

```gitignore
__pycache__/
*.pyc
.ipynb_checkpoints/
.cache/
.vscode/

work_dirs/
runs/
weights/
datasets/
data/

*.pkl
*.pt
*.pth
*.onnx
*.engine
*.log
```

Dataset files and trained checkpoints should be provided through external download links if necessary.

## 13. Citation

If this project is used in academic work, please cite the original methods and frameworks related to:

- Jittor
- JDet
- YOLOv8 / Ultralytics YOLO
- M2D-LIF
- DroneVehicle dataset

## 14. License

Please check `LICENSE.txt` for license information.

If this repository contains code adapted from JDet, YOLOv8, or other open-source projects, retain the original license notices and cite the corresponding repositories properly.
