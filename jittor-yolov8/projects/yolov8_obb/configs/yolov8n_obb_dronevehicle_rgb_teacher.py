_base_ = [
    "../../yolo/configs/yolo_optimizer_base.py",
    "../../yolo/configs/yolo_scheduler_base.py",
]

batch_size = 32
max_epoch = 200
log_interval = 10

# YOLO-style training-engine options.
ema = True
ema_decay = 0.9999
amp = False  # Jittor AMP support is environment-dependent; Runner enables it only when available.
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

debug_train_stall = False # debug开关

debug_batch_start = 70
debug_batch_end = 100

dataset_type = "YoloOBBDataset"

model = dict(
    type="YOLOv8OBB",
    cfg="/root/M2D-LIF_Jittor/projects/yolov8_obb/configs/yolo_configs/yolov8n_obb.yaml",
    ch=3,
    nc=5,
    imgsz=imgsz,
    conf_thres=0.25,
    iou_thres=0.7,
    assigner_min_pos_score=0, 
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
        path="/root/M2D-LIF_Jittor/DroneVehicle/train.txt",
        task="train",
        batch_size=batch_size,
        num_workers=8,
        stride=stride,
        imgsz=imgsz,
        augment=True,
        # YOLOv8-style train augmentations for OBB polygon labels.
        # OBB rotation itself is supervised by transformed polygons and ProbIoU.
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
        path="/root/M2D-LIF_Jittor/DroneVehicle/val.txt",
        task="val",
        batch_size=batch_size,
        num_workers=8,
        stride=stride,
        imgsz=imgsz_test,
        nc=5,
        conf_thres=0.25,
        iou_thres=0.7,
        max_det=300,
    ),
    test=dict(
        type=dataset_type,
        path="",
        task="test",
        batch_size=batch_size,
        num_workers=8,
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

logger = dict(
    type="RunLogger"
)
work_dir = "./work_dirs/RGB_GPU-LOG-64"
# work_dir = "./work_dirs/yolov8n_obb_dronevehicle_rgb_teacher"
