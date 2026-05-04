_base_ = [
    "../../yolo/configs/yolo_optimizer_base.py",
    "../../yolo/configs/yolo_scheduler_base.py",
]

batch_size = 16
max_epoch = 100
log_interval = 10
ema = True
ema_decay = 0.9999
amp = False
nominal_batch_size = 64
grad_accumulate = 1
strict_load = False
save_best = True
metric_key = 'val/map'
eval_interval = 1
checkpoint_interval = 1
max_gt = 256
stride = 32
imgsz = 640
imgsz_test = 640
debug_train_stall = False
debug_batch_start = 70
debug_batch_end = 100

# Paired RGB path: /root/JDet/DV-full/images/train/xxx.jpg
# Paired IR path is inferred as /root/JDet/DV-full/images_ir/train/xxx.jpg
# If your folder name is different, either rename/symlink it to images_ir or override _ir_path in YoloOBBMultiModalDataset.
dataset_type = "YoloOBBMultiModalDataset"

model = dict(
    type="YOLOv8OBBM2DLIF",
    cfg="/root/JDet/jittor-yolov8/projects/yolov8_obb/configs/yolo_configs/yolov8n_LIF_obb.yaml",
    teacher_rgb_cfg="/root/JDet/jittor-yolov8/projects/yolov8_obb/configs/yolo_configs/yolov8n_obb.yaml",
    teacher_ir_cfg="/root/JDet/jittor-yolov8/projects/yolov8_obb/configs/yolo_configs/yolov8n_obb.yaml",
    # Replace with your trained teacher pkl files.
    teacher_rgb_ckpt="/root/JDet/work_dirs/RGB-full.pkl",
    teacher_ir_ckpt="/root/JDet/work_dirs/IR-full.pkl",
    ch=6,
    nc=5,
    imgsz=imgsz,
    scale="n",
    conf_thres=0.25,
    iou_thres=0.7,
    assigner_min_pos_score=0,
    distill_weight=0.8,
    lif_weight=1.3,
    total_epochs=max_epoch,
    loss_type="CWD", # 同模态蒸馏损失类型
)

parameter_groups_generator = dict(
    batch_size=batch_size,
    accumulate=grad_accumulate,
    nominal_batch_size=nominal_batch_size,
)

scheduler = dict(
    max_steps=max_epoch,
    warmup='linear',
    warmup_iters=1000,
    warmup_init_lr_pg=[0.0, 0.0, 0.1],
    warmup_initial_momentum=0.8,
)

dataset = dict(
    train=dict(
        type=dataset_type,
        path="/root/JDet/5000-DroneVehice/train.txt",
        task="train",
        batch_size=batch_size,
        num_workers=4,
        stride=stride,
        imgsz=imgsz,
        augment=True,
        mosaic=1.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.10,
        scale=0.50,
        shear=2.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.0,
        nc=5,
    ),
    val=dict(
        type=dataset_type,
        path="/root/JDet/5000-DroneVehice/val.txt",
        task="val",
        batch_size=batch_size,
        num_workers=4,
        stride=stride,
        imgsz=imgsz_test,
        nc=5,
        conf_thres=0.25,
        iou_thres=0.7,
        max_det=300,
    ),
    test=dict(
        type=dataset_type,
        path="/root/JDet/5000-DroneVehice/val.txt",
        task="test",
        batch_size=batch_size,
        num_workers=4,
        stride=stride,
        imgsz=imgsz_test,
        nc=5,
        conf_thres=0.25,
        iou_thres=0.7,
        max_det=300,
    ),
)

optimizer = dict(
    type="SGD",
    lr=0.01,
    momentum=0.937,
    weight_decay=0.0005,
)

logger = dict(type="RunLogger")
work_dir = "./work_dirs/_DroneVehicle_M2D-LIF"
