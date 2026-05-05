"""
Convert Jittor .pkl checkpoint to PyTorch .pt checkpoint.

python /root/M2D-LIF/tools/jittor_pkl_to_pt.py \
    --jittor_ckpt /root/M2D-LIF/weights/IR-full.pkl \
    --torch_ckpt /root/M2D-LIF/weights/IR-full.pt 
"""



import argparse
import os
import sys
from collections import OrderedDict

import numpy as np
import torch


def strip_prefix(key: str) -> str:
    """
    Strip common checkpoint prefixes.
    """
    prefixes = [
        "module.",
        "model.",
        "avg_model.",
        "ema.",
    ]

    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if key.startswith(p):
                key = key[len(p):]
                changed = True

    return key


def to_torch_tensor(value):
    """
    Convert Jittor Var / numpy array / scalar to torch.Tensor.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()

    # Jittor Var usually has .numpy()
    if hasattr(value, "numpy"):
        value = value.numpy()

    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).cpu()

    if isinstance(value, (int, float, bool)):
        return torch.tensor(value)

    try:
        return torch.tensor(value)
    except Exception:
        raise TypeError(f"Cannot convert value of type {type(value)} to torch.Tensor")


def load_jittor_pkl(path: str):
    """
    Load Jittor checkpoint.
    """
    try:
        import jittor as jt
    except Exception as e:
        raise RuntimeError(
            "Jittor is required to load .pkl checkpoint. "
            "Please run this script in your Jittor environment."
        ) from e

    return jt.load(path)


def extract_state_dict(ckpt):
    """
    Extract state_dict from different possible Jittor checkpoint formats.
    """
    if not isinstance(ckpt, dict):
        raise TypeError(f"Expected checkpoint to be dict, but got {type(ckpt)}")

    candidate_keys = [
        "state_dict",
        "model",
        "params",
        "net",
        "network",
        "weights",
    ]

    for key in candidate_keys:
        if key in ckpt and isinstance(ckpt[key], dict):
            print(f"[Info] Found state dict under key: {key}")
            return ckpt[key]

    # If all values look like parameters, treat ckpt itself as state_dict
    print("[Info] No explicit state_dict key found. Treat checkpoint itself as state_dict.")
    return ckpt


def convert_jittor_pkl_to_pt(jittor_ckpt_path: str, torch_ckpt_path: str, strip: bool = True):
    if not os.path.exists(jittor_ckpt_path):
        raise FileNotFoundError(f"Jittor checkpoint not found: {jittor_ckpt_path}")

    print(f"[Info] Loading Jittor checkpoint: {jittor_ckpt_path}")
    jt_ckpt = load_jittor_pkl(jittor_ckpt_path)

    state_dict = extract_state_dict(jt_ckpt)

    converted_state_dict = OrderedDict()
    skipped = []

    for key, value in state_dict.items():
        try:
            new_key = strip_prefix(key) if strip else key
            tensor = to_torch_tensor(value)
            converted_state_dict[new_key] = tensor
        except Exception as e:
            skipped.append((key, str(e)))

    out = {
        "state_dict": converted_state_dict,
        "meta": {
            "converted_from": "Jittor",
            "source_path": os.path.abspath(jittor_ckpt_path),
            "num_tensors": len(converted_state_dict),
            "num_skipped": len(skipped),
        },
    }

    out_dir = os.path.dirname(torch_ckpt_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"[Info] Saving PyTorch checkpoint: {torch_ckpt_path}")
    torch.save(out, torch_ckpt_path)

    print("=" * 80)
    print(f"[Done] Converted tensors: {len(converted_state_dict)}")
    print(f"[Done] Skipped tensors: {len(skipped)}")
    print(f"[Done] Output: {torch_ckpt_path}")

    if skipped:
        print("\n[Warning] Some items were skipped:")
        for k, reason in skipped[:20]:
            print(f"  - {k}: {reason}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")

    print("\n[Preview] First 20 converted keys:")
    for i, (k, v) in enumerate(converted_state_dict.items()):
        if i >= 20:
            break
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")

    return torch_ckpt_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Jittor .pkl checkpoint to PyTorch .pt checkpoint."
    )

    parser.add_argument(
        "--jittor_ckpt",
        type=str,
        required=True,
        help="Path to input Jittor .pkl checkpoint.",
    )

    parser.add_argument(
        "--torch_ckpt",
        type=str,
        required=True,
        help="Path to output PyTorch .pt checkpoint.",
    )

    parser.add_argument(
        "--no_strip_prefix",
        action="store_true",
        help="Do not strip common prefixes such as module., model., avg_model.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    try:
        convert_jittor_pkl_to_pt(
            jittor_ckpt_path=args.jittor_ckpt,
            torch_ckpt_path=args.torch_ckpt,
            strip=not args.no_strip_prefix,
        )
    except Exception as e:
        print(f"[Error] Conversion failed: {e}")
        sys.exit(1)