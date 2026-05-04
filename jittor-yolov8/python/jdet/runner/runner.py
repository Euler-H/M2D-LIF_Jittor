from genericpath import isfile
import time
import jittor as jt
from tqdm import tqdm
import numpy as np
import jdet
import pickle
import datetime
from jdet.config import get_cfg,save_cfg
from jdet.utils.visualization import visualize_results
from jdet.utils.registry import build_from_cfg,MODELS,SCHEDULERS,DATASETS,HOOKS,OPTIMS
from jdet.config import get_classes_by_name
from jdet.utils.general import build_file, current_time, sync,check_file,check_interval,parse_losses,search_ckpt
from jdet.data.devkits.data_merge import data_merge_result
import os
import shutil
from tqdm import tqdm
from jittor_utils import auto_diff
import copy
import csv
import json
import platform
import subprocess


def get_gpu_info_nvml(gpu_id=0):
    """
    Return current GPU memory and utilization by NVML.

    gpu_mem_used_mb: current used GPU memory, MiB
    gpu_mem_total_mb: total GPU memory, MiB
    gpu_util_percent: GPU compute utilization, %
    mem_util_percent: memory controller utilization, %
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_id))

        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)

        return {
            "gpu_mem_used_mb": float(mem.used) / 1024.0 / 1024.0,
            "gpu_mem_total_mb": float(mem.total) / 1024.0 / 1024.0,
            "gpu_mem_used_percent": float(mem.used) / float(mem.total) * 100.0,
            "gpu_util_percent": float(util.gpu),
            "mem_util_percent": float(util.memory),
        }
    except Exception:
        return {
            "gpu_mem_used_mb": -1.0,
            "gpu_mem_total_mb": -1.0,
            "gpu_util_percent": -1.0,
            "mem_util_percent": -1.0,
        }


def append_csv_row(csv_path, row):
    """
    Append one row to csv. Automatically create header.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)

    # Convert unsupported values to printable scalars.
    clean_row = {}
    for k, v in row.items():
        try:
            if isinstance(v, jt.Var):
                clean_row[k] = float(v.numpy().reshape(-1)[0])
            elif isinstance(v, np.ndarray):
                if v.size == 1:
                    clean_row[k] = float(v.reshape(-1)[0])
                else:
                    clean_row[k] = str(v.tolist())
            elif isinstance(v, (np.float32, np.float64, np.int32, np.int64)):
                clean_row[k] = v.item()
            else:
                clean_row[k] = v
        except Exception:
            clean_row[k] = str(v)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(clean_row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(clean_row)


def get_env_info_basic():
    """
    Basic environment information for experiment comparison.
    """
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    try:
        info["jittor_version"] = jt.__version__
    except Exception:
        info["jittor_version"] = "unknown"

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            encoding="utf-8",
        )
        info["nvidia_smi_gpu"] = out.strip()
    except Exception:
        info["nvidia_smi_gpu"] = "unknown"

    return info


class RunnerEMA:
    """Exponential Moving Average for Jittor models."""

    def __init__(self, model, decay=0.9999, updates=0):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.updates = int(updates)
        self.base_decay = float(decay)
        for p in self.ema.parameters():
            p.stop_grad()

    def decay(self):
        return self.base_decay * (1.0 - np.exp(-self.updates / 2000.0))

    def update(self, model):
        with jt.no_grad():
            self.updates += 1
            d = self.decay()
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for k, v in esd.items():
                if k in msd and hasattr(v, 'dtype') and str(v.dtype).startswith('float'):
                    v.assign(v * d + msd[k].detach() * (1.0 - d))

    def update_attr(self, model, include=('yaml', 'nc', 'imgsz', 'stride', 'conf_thres', 'iou_thres')):
        for k in include:
            if hasattr(model, k):
                try:
                    setattr(self.ema, k, copy.deepcopy(getattr(model, k)))
                except Exception:
                    setattr(self.ema, k, getattr(model, k))


class Runner:
    def __init__(self):
        cfg = get_cfg()
        self.cfg = cfg
        self.flip_test = [] if cfg.flip_test is None else cfg.flip_test
        self.work_dir = cfg.work_dir

        self.perf_log_path = os.path.join(self.work_dir, "perf_log.csv")
        self.train_log_path = os.path.join(self.work_dir, "train_log.csv")
        self.env_log_path = os.path.join(self.work_dir, "env.json")

        self.gpu_id = int(getattr(cfg, "gpu_id", 0) or 0)
        self.run_start_time = time.time()
        self.last_train_perf = {}

        self.max_epoch = cfg.max_epoch 
        self.max_iter = cfg.max_iter
        assert (self.max_iter is None)^(self.max_epoch is None),"You must set max_iter or max_epoch"

        self.checkpoint_interval = cfg.checkpoint_interval
        self.eval_interval = cfg.eval_interval
        self.log_interval = cfg.log_interval
        self.resume_path = cfg.resume_path
    
        self.model = build_from_cfg(cfg.model,MODELS)

        self.ema_enabled = bool(getattr(cfg, 'ema', False))
        self.ema = RunnerEMA(self.model, decay=float(getattr(cfg, 'ema_decay', 0.9999))) if self.ema_enabled else None
        self.grad_accumulate = max(int(getattr(cfg, 'grad_accumulate', 1)), 1)
        self.save_best = bool(getattr(cfg, 'save_best', True))
        self.metric_key = getattr(cfg, 'metric_key', 'mAP50-95')
        self.best_metric = -1e100
        self.strict_load = bool(getattr(cfg, 'strict_load', False))
        self.amp = bool(getattr(cfg, 'amp', False))
        if self.amp and jt.rank == 0:
            print('AMP requested, but this Jittor Runner keeps FP32 unless the local Jittor build exposes stable AMP flags.')
        if self.ema_enabled and jt.rank == 0:
            print('EMA enabled: decay=%s' % getattr(cfg, 'ema_decay', 0.9999))
        if jt.rank == 0:
            print('Gradient accumulation: %s' % self.grad_accumulate)
        if (cfg.parameter_groups_generator):
            params = build_from_cfg(cfg.parameter_groups_generator,MODELS,named_params=self.model.named_parameters(), model=self.model)
        else:
            params = self.model.parameters()
        self.optimizer = build_from_cfg(cfg.optimizer,OPTIMS,params=params)
        self.scheduler = build_from_cfg(cfg.scheduler,SCHEDULERS,optimizer=self.optimizer)
        self.train_dataset = build_from_cfg(cfg.dataset.train,DATASETS,drop_last=jt.in_mpi)
        self.val_dataset = build_from_cfg(cfg.dataset.val,DATASETS)
        self.test_dataset = build_from_cfg(cfg.dataset.test,DATASETS)

        if hasattr(self.scheduler, 'warmup_iters') and hasattr(cfg.scheduler, 'warmup_epochs') and self.train_dataset is not None:
            warmup_epochs = getattr(cfg.scheduler, "warmup_epochs", None)
            warmup_iters = getattr(cfg.scheduler, "warmup_iters", 1000)

            if warmup_epochs is not None:
                self.scheduler.warmup_iters = max(
                    int(float(warmup_epochs) * len(self.train_dataset)), 1
                )
            else:
                self.scheduler.warmup_iters = int(warmup_iters)
                
            if jt.rank == 0:
                print('Scheduler warmup_iters set from warmup_epochs: %s' % self.scheduler.warmup_iters)
        
        self.logger = build_from_cfg(self.cfg.logger,HOOKS,work_dir=self.work_dir)

        if jt.rank == 0:
            try:
                env_info = get_env_info_basic()
                env_info.update({
                    "name": self.cfg.name,
                    "work_dir": self.work_dir,
                    "batch_size": getattr(self.cfg.dataset.train, "batch_size", None),
                    "imgsz": getattr(self.cfg.dataset.train, "imgsz", None),
                    "max_epoch": self.max_epoch,
                    "amp": self.amp,
                    "ema": self.ema_enabled,
                    "grad_accumulate": self.grad_accumulate,
                })
                with open(self.env_log_path, "w") as f:
                    json.dump(env_info, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print("[Env log warning]", e)

        save_file = build_file(self.work_dir,prefix="config.yaml")
        save_cfg(save_file)

        self.iter = 0
        self.epoch = 0

        if self.max_epoch:
            if (self.train_dataset):
                self.total_iter = self.max_epoch * len(self.train_dataset)
            else:
                self.total_iter = 0
        else:
            self.total_iter = self.max_iter

        if (cfg.pretrained_weights):
            self.load(cfg.pretrained_weights, model_only=True)
        
        if self.resume_path is None:
            self.resume_path = search_ckpt(self.work_dir)
        if check_file(self.resume_path):
            self.resume()


    @property
    def finish(self):
        if self.max_epoch:
            return self.epoch>=self.max_epoch
        else:
            return self.iter>=self.max_iter
    
    def run(self):
        self.logger.print_log("Start running")
        
        while not self.finish:
            self.train()
            eval_results = None
            if check_interval(self.epoch,self.eval_interval):
                eval_results = self.val()
                if self.save_best and eval_results is not None:
                    self.save_best_checkpoint(eval_results)
            if check_interval(self.epoch,self.checkpoint_interval):
                self.save()
        self.test()

    def test_time(self):
        warmup = 10
        rerun = 100
        self.model.train()
        for batch_idx,(images,targets) in enumerate(self.train_dataset):
            break
        print("warmup...")
        for i in tqdm(range(warmup)):
            losses = self.model(images,targets)
            all_loss,losses = parse_losses(losses)
            self.optimizer.step(all_loss)
            self.scheduler.step(self.iter,self.epoch,by_epoch=True)
        jt.sync_all(True)
        print("testing...")
        start_time = time.time()
        for i in tqdm(range(rerun)):
            losses = self.model(images,targets)
            all_loss,losses = parse_losses(losses)
            self.optimizer.step(all_loss)
            self.scheduler.step(self.iter,self.epoch,by_epoch=True)
        jt.sync_all(True)
        batch_size = len(targets)*jt.world_size
        ptime = time.time()-start_time
        fps = batch_size*rerun/ptime
        print("FPS:", fps)

    @staticmethod
    def _loss_to_float(value):
        try:
            if isinstance(value, jt.Var):
                jt.sync_all(True)
                return float(value.numpy().reshape(-1)[0])
            if isinstance(value, np.ndarray):
                return float(value.reshape(-1)[0])
            return float(value)
        except Exception as e:
            print("[loss_to_float error]", e)
            return 0.0

    def train(self):

        self.model.train()
        if hasattr(self.model, "set_epoch"):
            self.model.set_epoch(self.epoch)

        if hasattr(self.train_dataset, "set_epoch"):
            self.train_dataset.set_epoch(self.epoch)
        try:
            self.train_dataset.max_epoch = self.max_epoch
        except Exception:
            pass

        start_time = time.time()
        epoch_start_time = start_time
        seen_images = 0

        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        last_log_data = None

        peak_gpu_mem_mb = 0.0
        peak_gpu_util = 0.0
        peak_mem_util = 0.0

        # ===== Debug range: only print around suspicious batches =====
        debug_batch_start = int(getattr(self.cfg, "debug_batch_start", 70))
        debug_batch_end = int(getattr(self.cfg, "debug_batch_end", 100))
        debug_enabled = bool(getattr(self.cfg, "debug_train_stall", True))

        progress = tqdm(
            enumerate(self.train_dataset),
            total=len(self.train_dataset),
            disable=(jt.rank != 0),
            dynamic_ncols=True,
            leave=True,
            desc=f"Epoch {self.epoch + 1}/{self.max_epoch if self.max_epoch else '?'}",
        )

        for batch_idx, (images, targets) in progress:

            do_debug = (
                debug_enabled
                and jt.rank == 0
                and debug_batch_start <= batch_idx <= debug_batch_end
            )

            if do_debug:
                try:
                    img_shape = getattr(images, "shape", None)
                    tgt_shape = getattr(targets, "shape", None)
                    print(
                        f"\n[DEBUG] epoch={self.epoch} batch={batch_idx} got batch "
                        f"images_shape={img_shape} targets_shape={tgt_shape}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"\n[DEBUG] epoch={self.epoch} batch={batch_idx} got batch, shape print failed: {e}", flush=True)

            # ======================
            # 1. Forward + loss
            # ======================
            if do_debug:
                print(f"[DEBUG] epoch={self.epoch} batch={batch_idx} before model", flush=True)

            # 进入模型
            losses = self.model(images, targets)

            if do_debug:
                print(f"[DEBUG] epoch={self.epoch} batch={batch_idx} after model", flush=True)
                try:
                    print(f"[DEBUG] epoch={self.epoch} batch={batch_idx} raw losses keys={list(losses.keys()) if isinstance(losses, dict) else type(losses)}", flush=True)
                except Exception as e:
                    print(f"[DEBUG] epoch={self.epoch} batch={batch_idx} raw losses print failed: {e}", flush=True)

            # ======================
            # 2. Parse loss
            # ======================
            if do_debug:
                print(f"[DEBUG] epoch={self.epoch} batch={batch_idx} before parse_losses", flush=True)

            all_loss, losses = parse_losses(losses)

            if do_debug:
                print(f"[DEBUG] epoch={self.epoch} batch={batch_idx} after parse_losses", flush=True)
                try:
                    # 这里会同步一次，只在 debug batch 范围内使用。
                    loss_scalar_dbg = self._loss_to_float(all_loss)
                    print(
                        f"[DEBUG] epoch={self.epoch} batch={batch_idx} all_loss={loss_scalar_dbg:.6f}",
                        flush=True,
                    )
                    print(
                        f"[DEBUG] epoch={self.epoch} batch={batch_idx} parsed losses={losses}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"[DEBUG] epoch={self.epoch} batch={batch_idx} loss scalar print failed: {e}", flush=True)

            # ======================
            # 3. Backward / optimizer
            # ======================
            update_now = ((self.iter + 1) % self.grad_accumulate == 0) or (batch_idx + 1 == len(self.train_dataset))

            if do_debug:
                print(
                    f"[DEBUG] epoch={self.epoch} batch={batch_idx} before optimizer "
                    f"update_now={update_now} grad_accumulate={self.grad_accumulate}",
                    flush=True,
                )

            if self.grad_accumulate > 1:
                self.optimizer.pre_step(all_loss / float(self.grad_accumulate))
                if update_now:
                    if do_debug:
                        print(f"[DEBUG] epoch={self.epoch} batch={batch_idx} before optimizer.step(None)", flush=True)
                    self.optimizer.step(None)
                    if do_debug:
                        print(f"[DEBUG] epoch={self.epoch} batch={batch_idx} after optimizer.step(None)", flush=True)
                    if self.ema is not None:
                        if do_debug:
                            print(f"[DEBUG] epoch={self.epoch} batch={batch_idx} before ema.update", flush=True)
                        self.ema.update(self.model)
                        if do_debug:
                            print(f"[DEBUG] epoch={self.epoch} batch={batch_idx} after ema.update", flush=True)
            else:
                if do_debug:
                    print(f"[DEBUG] epoch={self.epoch} batch={batch_idx} before optimizer.step(all_loss)", flush=True)
                self.optimizer.step(all_loss)
                if do_debug:
                    print(f"[DEBUG] epoch={self.epoch} batch={batch_idx} after optimizer.step(all_loss)", flush=True)
                if self.ema is not None:
                    if do_debug:
                        print(f"[DEBUG] epoch={self.epoch} batch={batch_idx} before ema.update", flush=True)
                    self.ema.update(self.model)
                    if do_debug:
                        print(f"[DEBUG] epoch={self.epoch} batch={batch_idx} after ema.update", flush=True)

            # ======================
            # 4. Scheduler
            # ======================
            if do_debug:
                print(f"[DEBUG] epoch={self.epoch} batch={batch_idx} before scheduler.step", flush=True)

            self.scheduler.step(self.iter, self.epoch, by_epoch=True)

            if do_debug:
                print(f"[DEBUG] epoch={self.epoch} batch={batch_idx} after scheduler.step", flush=True)

            # ======================
            # 5. Progress / log
            # ======================
            batch_size = len(images) * jt.world_size
            seen_images += batch_size

            elapsed = max(time.time() - start_time, 1e-9)
            fps = seen_images / elapsed

            gpu_info_now = get_gpu_info_nvml(self.gpu_id)
            if gpu_info_now["gpu_mem_used_mb"] > peak_gpu_mem_mb:
                peak_gpu_mem_mb = gpu_info_now["gpu_mem_used_mb"]
            if gpu_info_now["gpu_util_percent"] > peak_gpu_util:
                peak_gpu_util = gpu_info_now["gpu_util_percent"]
            if gpu_info_now["mem_util_percent"] > peak_mem_util:
                peak_mem_util = gpu_info_now["mem_util_percent"]

            # Do not sync loss scalar every iteration.
            # value.data/numpy() forces GPU/CPU synchronization in Jittor.
            need_log = jt.rank == 0 and (
                batch_idx == 0 or check_interval(batch_idx + 1, max(1, self.log_interval))
            )

            if need_log:
                loss_float = self._loss_to_float(all_loss)

                box_loss_float = self._loss_to_float(losses.get("box_loss", 0.0)) if isinstance(losses, dict) else 0.0
                cls_loss_float = self._loss_to_float(losses.get("cls_loss", 0.0)) if isinstance(losses, dict) else 0.0
                dfl_loss_float = self._loss_to_float(losses.get("dfl_loss", 0.0)) if isinstance(losses, dict) else 0.0
                m2d_d_loss_float = self._loss_to_float(losses.get("m2d_d_loss", 0.0))
                m2d_c_loss_float = self._loss_to_float(losses.get("m2d_c_loss", 0.0))
                lif_loss_float = self._loss_to_float(losses.get("illumination_loss", 0.0))

                epoch_loss_sum += loss_float
                epoch_loss_count += 1

                progress.set_postfix(
                    loss=f"{loss_float:.4f}",
                    box=f"{box_loss_float:.4f}",
                    cls=f"{cls_loss_float:.4f}",
                    dfl=f"{dfl_loss_float:.4f}",
                    lif=f"{lif_loss_float:.4f}",
                    m2d_d=f"{m2d_d_loss_float:.4f}",
                    m2d_c=f"{m2d_c_loss_float:.4f}",
                    lr=f"{self.optimizer.cur_lr():.3g}",
                )
            else:
                loss_float = None

            if check_interval(self.iter, self.log_interval) and self.iter > 0:
                eta_time = (self.total_iter - self.iter) * elapsed / max(batch_idx + 1, 1)
                eta_str = str(datetime.timedelta(seconds=int(max(eta_time, 0))))

                data = dict(
                    name=self.cfg.name,
                    lr=self.optimizer.cur_lr(),
                    iter=self.iter,
                    epoch=self.epoch,
                    batch_idx=batch_idx,
                    batch_size=batch_size,
                    total_loss=loss_float if loss_float is not None else all_loss,
                    fps=fps,
                    eta=eta_str,
                )

                # 只记录核心 loss，不把 angle_loss 塞进输出
                if isinstance(losses, dict):
                    for k in ["box_loss", "cls_loss", "dfl_loss", "illumination_loss", "m2d_d_loss", "m2d_c_loss"]:
                        if k in losses:
                            data[k] = losses[k]

                last_log_data = data

            if do_debug:
                print(f"[DEBUG] epoch={self.epoch} batch={batch_idx} end batch loop", flush=True)

            self.iter += 1
            if self.finish:
                break

        # Keep one structured record per epoch for logger hooks, but avoid the
        # previous per-iteration console spam.
        train_epoch_time = time.time() - epoch_start_time
        train_fps_epoch = seen_images / max(train_epoch_time, 1e-9)

        if last_log_data is not None:
            last_log_data = sync(last_log_data)
            if jt.rank == 0:
                last_log_data["epoch_avg_loss"] = epoch_loss_sum / max(epoch_loss_count, 1)

                # Performance fields
                last_log_data["train_epoch_time_sec"] = train_epoch_time
                last_log_data["train_fps_epoch"] = train_fps_epoch
                last_log_data["train_seen_images"] = seen_images
                last_log_data["peak_gpu_mem_mb"] = peak_gpu_mem_mb
                last_log_data["peak_gpu_util_percent"] = peak_gpu_util
                last_log_data["peak_mem_util_percent"] = peak_mem_util

                self.last_train_perf = {
                    "epoch": self.epoch + 1,
                    "train_epoch_time_sec": train_epoch_time,
                    "train_fps_epoch": train_fps_epoch,
                    "train_seen_images": seen_images,
                    "peak_gpu_mem_mb": peak_gpu_mem_mb,
                    "peak_gpu_util_percent": peak_gpu_util,
                    "peak_mem_util_percent": peak_mem_util,
                }

                self.logger.log(last_log_data)

                append_csv_row(self.perf_log_path, {
                    "phase": "train",
                    "epoch": self.epoch + 1,
                    "framework": "Jittor",
                    "batch_size": getattr(self.cfg.dataset.train, "batch_size", None),
                    "imgsz": getattr(self.cfg.dataset.train, "imgsz", None),
                    "amp": self.amp,
                    "ema": self.ema_enabled,
                    "grad_accumulate": self.grad_accumulate,
                    "time_sec": train_epoch_time,
                    "fps": train_fps_epoch,
                    "seen_images": seen_images,
                    "peak_gpu_mem_mb": peak_gpu_mem_mb,
                    "peak_gpu_util_percent": peak_gpu_util,
                    "peak_mem_util_percent": peak_mem_util,
                    "lr": self.optimizer.cur_lr(),
                    "epoch_avg_loss": last_log_data["epoch_avg_loss"],
                })

        self.epoch += 1

    @jt.no_grad()
    @jt.single_process_scope()
    def run_on_images(self,save_dir=None,**kwargs):
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        self.model.eval()
        for i,(images,targets) in tqdm(enumerate(self.test_dataset)):
            results = self.model(images,targets)
            if save_dir:
                
                # 更改后：
                sync_results = sync(results)

                # 1. 展平 results: 
                # 原结构可能是 [[det0, det1, ..., det15], [det16, ...]]
                flat_results = []
                for r in sync_results:
                    if isinstance(r, (list, tuple)):
                        flat_results.extend(r)
                    else:
                        flat_results.append(r)

                # 2. 展平 image files
                flat_files = []
                for t in targets:
                    img_file = t["img_file"]
                    if isinstance(img_file, (list, tuple)):
                        flat_files.extend(img_file)
                    else:
                        flat_files.append(img_file)

                # 3. 将 det[N,6] 转成 visualize_results 需要的三元组:
                #    (bboxes, scores, labels)
                vis_results = []
                for det in flat_results:
                    if det is None or len(det) == 0:
                        vis_results.append(([], [], []))
                    else:
                        # det: [N, 6] = x1,y1,x2,y2,score,cls
                        bboxes = det[:, :4]
                        scores = det[:, 4]
                        labels = det[:, 5]
                        vis_results.append((bboxes, scores, labels))

                # 4. 避免 results 和 files 数量不一致导致 zip 截断不易察觉
                n = min(len(vis_results), len(flat_files))
                print(f"visualize images: results={len(vis_results)}, files={len(flat_files)}, use={n}")

                visualize_results(
                    vis_results[:n],
                    get_classes_by_name(self.test_dataset.dataset_type),
                    flat_files[:n],
                    save_dir,
                    **kwargs
                )
                # visualize_results(sync(results),get_classes_by_name(self.test_dataset.dataset_type),[t["img_file"] for t in targets],save_dir, **kwargs)

    @jt.no_grad()
    @jt.single_process_scope()
    def val(self):
        if self.val_dataset is None:
            self.logger.print_log("Please set Val dataset")
        else:
            self.logger.print_log("Validating....")

            val_start_time = time.time()
            val_seen_images = 0
            val_peak_gpu_mem_mb = 0.0
            val_peak_gpu_util = 0.0
            val_peak_mem_util = 0.0

            # TODO: need move eval into this function
            eval_model = self.ema.ema if self.ema is not None else self.model
            if self.ema is not None:
                self.ema.update_attr(self.model)
            eval_model.eval()
            results = []
            for batch_idx, (images, targets) in tqdm(enumerate(self.val_dataset), total=len(self.val_dataset)):
                result = eval_model(images, targets)

                cur_bs = len(images) * jt.world_size
                val_seen_images += cur_bs

                gpu_info_now = get_gpu_info_nvml(self.gpu_id)
                if gpu_info_now["gpu_mem_used_mb"] > val_peak_gpu_mem_mb:
                    val_peak_gpu_mem_mb = gpu_info_now["gpu_mem_used_mb"]
                if gpu_info_now["gpu_util_percent"] > val_peak_gpu_util:
                    val_peak_gpu_util = gpu_info_now["gpu_util_percent"]
                if gpu_info_now["mem_util_percent"] > val_peak_mem_util:
                    val_peak_mem_util = gpu_info_now["mem_util_percent"]

                # YOLOv8-OBB 数据集的 targets 是整批拼接后的 [M, 10]，
                # 不能直接 zip(result, targets)。
                if getattr(self.val_dataset, "is_yolo_obb", False):
                    result_np = sync(result)
                    targets_np = sync(targets)

                    if targets_np is None:
                        targets_np = np.zeros((0, 10), dtype=np.float32)
                    else:
                        targets_np = np.asarray(targets_np, dtype=np.float32)

                    # val 默认 shuffle=False，因此可以用 batch_idx 和 batch_size 找回图像路径
                    start = batch_idx * self.val_dataset.batch_size
                    num_imgs = result_np.shape[0]
                    img_paths = self.val_dataset.img_files[start:start + num_imgs]

                    for bi in range(num_imgs):
                        if targets_np.ndim == 2 and targets_np.shape[0] > 0:
                            labels_i = targets_np[targets_np[:, 0] == bi].copy()
                        else:
                            labels_i = np.zeros((0, 10), dtype=np.float32)

                        img_path = img_paths[bi] if bi < len(img_paths) else str(start + bi)

                        results.append(
                            (
                                result_np[bi],
                                labels_i,
                                img_path,
                            )
                        )

                else:
                    results.extend([(r, t) for r, t in zip(sync(result), sync(targets))])

            eval_results = self.val_dataset.evaluate(results, self.work_dir, self.epoch, logger=self.logger)

            val_time = time.time() - val_start_time
            val_fps = val_seen_images / max(val_time, 1e-9)

            if jt.rank == 0:
                self.logger.print_log(
                    "Val performance: time={:.2f}s, fps={:.2f} images/s, "
                    "peak_gpu_mem={:.0f} MiB, gpu_util_peak={:.0f}%".format(
                        val_time,
                        val_fps,
                        val_peak_gpu_mem_mb,
                        val_peak_gpu_util,
                    )
                )

                append_csv_row(self.perf_log_path, {
                    "phase": "val",
                    "epoch": self.epoch,
                    "framework": "Jittor",
                    "batch_size": getattr(self.cfg.dataset.val, "batch_size", None),
                    "imgsz": getattr(self.cfg.dataset.val, "imgsz", None),
                    "amp": self.amp,
                    "ema": self.ema_enabled,
                    "grad_accumulate": self.grad_accumulate,
                    "time_sec": val_time,
                    "fps": val_fps,
                    "seen_images": val_seen_images,
                    "peak_gpu_mem_mb": val_peak_gpu_mem_mb,
                    "peak_gpu_util_percent": val_peak_gpu_util,
                    "peak_mem_util_percent": val_peak_mem_util,
                    "map50": eval_results.get("val/map50", None),
                    "map": eval_results.get("val/map", None),
                })

                if jt.rank == 0:
                    train_perf = getattr(self, "last_train_perf", {})

                    append_csv_row(self.train_log_path, {
                        "epoch": self.epoch,
                        "iter": self.iter,
                        "framework": "Jittor",
                        "lr": self.optimizer.cur_lr(),
                        "map50": eval_results.get("val/map50", None),
                        "map": eval_results.get("val/map", None),
                        "train_epoch_time_sec": train_perf.get("train_epoch_time_sec", None),
                        "train_fps_epoch": train_perf.get("train_fps_epoch", None),
                        "train_peak_gpu_mem_mb": train_perf.get("peak_gpu_mem_mb", None),
                        "val_time_sec": val_time,
                        "val_fps": val_fps,
                        "val_peak_gpu_mem_mb": val_peak_gpu_mem_mb,
                    })

                # 精度日志单独保持简洁
                self.logger.log(eval_results, iter=self.iter)

            return eval_results

    @jt.no_grad()
    @jt.single_process_scope()
    def test(self):

        if self.test_dataset is None:
            self.logger.print_log("Please set Test dataset")
        else:
            self.logger.print_log("Testing...")
            eval_model = self.ema.ema if self.ema is not None else self.model
            if self.ema is not None:
                self.ema.update_attr(self.model)
            eval_model.eval()
            results = []
            for batch_idx,(images,targets) in tqdm(enumerate(self.test_dataset),total=len(self.test_dataset)):
                result = eval_model(images,targets)
                results.extend([(r,t) for r,t in zip(sync(result),sync(targets))])
                for mode in self.flip_test:
                    images_flip = images.copy()
                    if (mode == 'H'):
                        images_flip = images_flip[:, :, :, ::-1]
                    elif (mode == 'V'):
                        images_flip = images_flip[:, :, ::-1, :]
                    elif (mode == 'HV'):
                        images_flip = images_flip[:, :, ::-1, ::-1]
                    else:
                        assert(False)
                    result = eval_model(images_flip,targets)
                    targets_ = copy.deepcopy(targets)
                    for i in range(len(targets_)):
                        targets_[i]["flip_mode"] = mode
                    results.extend([(r,t) for r,t in zip(sync(result),sync(targets_))])

            save_file = build_file(self.work_dir,f"test/test_{self.epoch}.pkl")
            pickle.dump(results,open(save_file,"wb"))
            if (self.cfg.dataset.test.type == "ImageDataset"):
                dataset_type = self.test_dataset.dataset_type
                data_merge_result(save_file,self.work_dir,self.epoch,self.cfg.name,dataset_type,self.cfg.dataset.test.images_dir)

    @jt.single_process_scope()
    def save(self):
        ema_state = self.ema.ema.state_dict() if self.ema is not None else None
        save_data = {
            "meta":{
                "jdet_version": jdet.__version__,
                "epoch": self.epoch,
                "iter": self.iter,
                "max_iter": self.max_iter,
                "max_epoch": self.max_epoch,
                "best_metric": self.best_metric,
                "save_time":current_time(),
                "config": self.cfg.dump()
            },
            "model":self.model.state_dict(),
            "ema": ema_state,
            "ema_updates": self.ema.updates if self.ema is not None else 0,
            "scheduler": self.scheduler.parameters(),
            "optimizer": self.optimizer.parameters()
        }
        save_file = build_file(self.work_dir,prefix=f"checkpoints/ckpt_{self.epoch}.pkl")
        jt.save(save_data,save_file)
        latest_file = build_file(self.work_dir,prefix="checkpoints/latest.pkl")
        try:
            shutil.copyfile(save_file, latest_file)
        except Exception:
            jt.save(save_data, latest_file)
        if jt.rank == 0:
            print(f"saved checkpoint: {save_file}")

    def save_best_checkpoint(self, eval_results):
        metric = None
        for key in [self.metric_key, 'val/map', 'val/map50', 'mAP50-95', 'mAP', 'mAP50']:
            if isinstance(eval_results, dict) and key in eval_results:
                metric = self._loss_to_float(eval_results[key])
                break
        if metric is None:
            return
        if metric > self.best_metric:
            self.best_metric = metric
            save_data = {
                "meta":{
                    "jdet_version": jdet.__version__,
                    "epoch": self.epoch,
                    "iter": self.iter,
                    "max_iter": self.max_iter,
                    "max_epoch": self.max_epoch,
                    "best_metric": self.best_metric,
                    "metric_key": self.metric_key,
                    "save_time":current_time(),
                    "config": self.cfg.dump()
                },
                "model": self.model.state_dict(),
                "ema": self.ema.ema.state_dict() if self.ema is not None else None,
                "ema_updates": self.ema.updates if self.ema is not None else 0,
                "scheduler": self.scheduler.parameters(),
                "optimizer": self.optimizer.parameters()
            }
            best_file = build_file(self.work_dir,prefix="checkpoints/best.pkl")
            jt.save(save_data,best_file)
            if jt.rank == 0:
                print(f"saved best checkpoint: {best_file}, {self.metric_key}={metric:.6f}")

    def _extract_state_dict(self, resume_data):
        if isinstance(resume_data, dict):
            if "model" in resume_data and resume_data["model"] is not None:
                return resume_data["model"]
            if "state_dict" in resume_data:
                return resume_data["state_dict"]
        return resume_data

    def _safe_load_model(self, state, target_model=None, name="model"):
        if target_model is None:
            target_model = self.model
        if not isinstance(state, dict):
            target_model.load_parameters(state)
            return
        cur = target_model.state_dict()
        matched, skipped = {}, []
        for k, v in state.items():
            if k in cur and hasattr(v, "shape") and tuple(v.shape) == tuple(cur[k].shape):
                matched[k] = v
            else:
                reason = "missing" if k not in cur else f"shape {getattr(v, 'shape', None)} != {cur[k].shape}"
                skipped.append((k, reason))
        if self.strict_load and skipped:
            preview = "; ".join([f"{k}: {r}" for k, r in skipped[:20]])
            raise RuntimeError(f"Strict load failed for {len(skipped)} {name} parameters. {preview}")
        target_model.load_parameters(matched)
        if jt.rank == 0:
            print(f"Loaded {len(matched)}/{len(cur)} {name} parameters; skipped {len(skipped)}.")
            if skipped:
                print("First skipped parameters:", "; ".join([f"{k} ({r})" for k, r in skipped[:8]]))

    def load(self, load_path, model_only=False):
        resume_data = jt.load(load_path)

        if (not model_only) and isinstance(resume_data, dict):
            meta = resume_data.get("meta",dict())
            self.epoch = meta.get("epoch",self.epoch)
            self.iter = meta.get("iter",self.iter)
            self.max_iter = meta.get("max_iter",self.max_iter)
            self.max_epoch = meta.get("max_epoch",self.max_epoch)
            self.best_metric = meta.get("best_metric", self.best_metric)
            self.scheduler.load_parameters(resume_data.get("scheduler",dict()))
            self.optimizer.load_parameters(resume_data.get("optimizer",dict()))

        state = self._extract_state_dict(resume_data)
        self._safe_load_model(state, self.model, name="model")

        if self.ema is not None:
            if isinstance(resume_data, dict) and resume_data.get("ema", None) is not None and not model_only:
                self._safe_load_model(resume_data["ema"], self.ema.ema, name="ema")
                self.ema.updates = int(resume_data.get("ema_updates", self.ema.updates))
            else:
                self.ema = RunnerEMA(self.model, decay=float(getattr(self.cfg, 'ema_decay', 0.9999)))

        self.logger.print_log(f"Loading model parameters from {load_path}")

    def resume(self):
        self.load(self.resume_path)