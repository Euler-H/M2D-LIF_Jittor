#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pack a converted PyTorch state_dict checkpoint into Ultralytics-style .pt.

Input:
    .pt containing {"state_dict": ...} converted from Jittor

Output:
    Ultralytics-style .pt containing {"model": model, ...}
"""
###########################################################################################################

# 蒸馏权重转换时需要添加，并且要使用蒸馏的模型配置文件：/root/M2D-LIF/model_yaml_obb/yolov8_LIF_obb.yaml
import os
import sys

ROOT = "/root/M2D-LIF"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

###########################################################################################################

import argparse
import os
import sys
from copy import deepcopy

import torch


def load_state_ckpt(path):
    ckpt = torch.load(path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]

    if isinstance(ckpt, dict):
        # 也兼容直接保存 state_dict 的情况
        return ckpt

    raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")


def normalize_key(k: str) -> str:
    """
    Normalize checkpoint key to Ultralytics model key.

    Examples:
        0.conv.weight -> model.0.conv.weight
        model.0.conv.weight -> model.0.conv.weight
        module.0.conv.weight -> model.0.conv.weight
        module.model.0.conv.weight -> model.0.conv.weight
        avg_model.0.conv.weight -> model.0.conv.weight
        ema.0.conv.weight -> model.0.conv.weight
    """

    # 先去掉常见外层前缀
    changed = True
    while changed:
        changed = False
        for p in ("module.", "avg_model.", "ema."):
            if k.startswith(p):
                k = k[len(p):]
                changed = True

    # 如果已经是 model.xxx，则不动
    if k.startswith("model."):
        return k

    # 如果是 0.conv.weight / 22.cv3... 这种数字开头，则补 model.
    first = k.split(".", 1)[0]
    if first.isdigit():
        k = "model." + k

    return k


def filter_matched_state_dict(model, state_dict):
    model_state = model.state_dict()

    matched = {}
    skipped = []

    for k, v in state_dict.items():
        nk = normalize_key(k)

        if nk in model_state and tuple(model_state[nk].shape) == tuple(v.shape):
            matched[nk] = v
        else:
            reason = "missing_key"
            if nk in model_state:
                reason = f"shape_mismatch: ckpt={tuple(v.shape)}, model={tuple(model_state[nk].shape)}"
            skipped.append((k, reason))

    missing = [k for k in model_state.keys() if k not in matched]

    return matched, skipped, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state_ckpt", required=True, help="Converted pt containing state_dict")
    parser.add_argument("--model_yaml", required=True, help="Ultralytics model yaml")
    parser.add_argument("--out", required=True, help="Output Ultralytics-style pt")
    parser.add_argument("--task", default="obb", help="task type: obb/detect/segment/etc.")
    args = parser.parse_args()

    # 重要：使用你当前项目里的 Ultralytics
    from ultralytics.nn.tasks import yaml_model_load, guess_model_scale
    from ultralytics.nn.tasks import OBBModel, DetectionModel

    print(f"[Info] Loading state_dict checkpoint: {args.state_ckpt}")
    state_dict = load_state_ckpt(args.state_ckpt)

    print(f"[Info] Building model from yaml: {args.model_yaml}")

    if args.task == "obb":
        model = OBBModel(args.model_yaml)
    elif args.task == "detect":
        model = DetectionModel(args.model_yaml)
    else:
        raise ValueError(f"Unsupported task: {args.task}")

    model.float()

    matched, skipped, missing = filter_matched_state_dict(model, state_dict)

    print("=" * 80)
    print(f"[Info] Model tensors:   {len(model.state_dict())}")
    print(f"[Info] CKPT tensors:    {len(state_dict)}")
    print(f"[Info] Matched tensors: {len(matched)}")
    print(f"[Info] Skipped tensors: {len(skipped)}")
    print(f"[Info] Missing tensors: {len(missing)}")

    print("\n[Preview] First 20 skipped:")
    for k, reason in skipped[:20]:
        print(f"  {k}: {reason}")

    print("\n[Preview] First 20 missing:")
    for k in missing[:20]:
        print(f"  {k}: {tuple(model.state_dict()[k].shape)}")

    model.load_state_dict(matched, strict=False)
    model.args = {"task": "obb"}
    model.task = "obb"

    ckpt = {
        "epoch": -1,
        "best_fitness": None,
        "model": model,
        "ema": None,
        "updates": None,
        "optimizer": None,
        "train_args": {
            "task": "obb",
            "mode": "train",
            "model": args.model_yaml,
        },
        "date": None,
        "version": "jittor-converted",
    }

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"\n[Info] Saving Ultralytics-style checkpoint: {args.out}")
    torch.save(ckpt, args.out)
    print("[Done]")


if __name__ == "__main__":
    main()
