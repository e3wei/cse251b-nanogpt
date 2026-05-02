"""
Concatenate Karpathy build-nanogpt FineWeb shards (.npy) into raw uint16 .bin files
for train.py (np.memmap).

fineweb.py writes edufineweb_train_*.npy and edufineweb_val_000000.npy under edu_fineweb10B/.
This script streams shards to disk so you do not load the full corpus into RAM.

Usage (from repo root):

  python scripts/fineweb_shards_to_bin.py \\
      --input-dir build-nanogpt/edu_fineweb10B \\
      --train-out train.bin \\
      --val-out fineweb_val.bin

Then in config or CLI: dataset_dir="." train_bin=train.bin val_bin=fineweb_val.bin

For the course leaderboard, keep using the provided val.bin with evaluate.py only.
Do not mix official val.bin tokens into train.bin. Using official val.bin as val_bin
during training (monitoring) is fine; training data must stay out of that split.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys


def _sorted_shard_paths(directory: str, prefix: str) -> list[str]:
    pattern = os.path.join(directory, f"{prefix}_*.npy")
    paths = glob.glob(pattern)
    if not paths:
        return []
    return sorted(paths)


def stream_npy_shards_to_bin(paths: list[str], out_path: str) -> int:
    import numpy as np

    total = 0
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as out_f:
        for i, p in enumerate(paths):
            arr = np.load(p, mmap_mode="r")
            if arr.dtype != np.uint16:
                raise ValueError(f"{p}: expected uint16, got {arr.dtype}")
            n = int(arr.size)
            arr.tofile(out_f)
            total += n
            print(f"  [{i + 1}/{len(paths)}] {os.path.basename(p)} -> +{n:_} tokens (total {total:_})")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge FineWeb .npy shards into raw .bin for train.py")
    parser.add_argument(
        "--input-dir",
        default="build-nanogpt/edu_fineweb10B",
        help="Directory containing edufineweb_train_*.npy and edufineweb_val_*.npy",
    )
    parser.add_argument(
        "--train-out",
        default="train.bin",
        help="Output path for training tokens (raw uint16)",
    )
    parser.add_argument(
        "--val-out",
        default="fineweb_val.bin",
        help="Output path for FineWeb holdout shard(s) (raw uint16). Use empty string to skip.",
    )
    args = parser.parse_args()

    input_dir = os.path.normpath(args.input_dir)
    if not os.path.isdir(input_dir):
        print(f"error: input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    train_paths = _sorted_shard_paths(input_dir, "edufineweb_train")
    if not train_paths:
        print(f"error: no edufineweb_train_*.npy under {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Writing {args.train_out} from {len(train_paths)} train shard(s)...")
    n_train = stream_npy_shards_to_bin(train_paths, args.train_out)
    print(f"Done train: {n_train:_} tokens -> {args.train_out}")

    if args.val_out:
        val_paths = _sorted_shard_paths(input_dir, "edufineweb_val")
        if not val_paths:
            print("warning: no edufineweb_val_*.npy found; skipping val export", file=sys.stderr)
        else:
            print(f"Writing {args.val_out} from {len(val_paths)} val shard(s)...")
            n_val = stream_npy_shards_to_bin(val_paths, args.val_out)
            print(f"Done val: {n_val:_} tokens -> {args.val_out}")


if __name__ == "__main__":
    main()
