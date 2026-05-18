#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.predict_window import load_model, predict_window, resolve_chr_file  # noqa: E402


def default_chroms():
    return [f"chr{i}" for i in range(1, 23)] + ["chrX"]


def main():
    parser = argparse.ArgumentParser(description="Run sliding-window ChIANet inference across chromosomes.")
    parser.add_argument("--seq-dir", required=True, help="Directory created by preprocess_fasta.py.")
    parser.add_argument("--chip-dir", required=True, help="Directory created by preprocess_bigwig.py.")
    parser.add_argument("--checkpoint", required=True, help="ChIANet checkpoint .pth file.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--chroms", nargs="*", default=None, help="Chromosomes to predict. Default: chr1-chr22 chrX.")
    parser.add_argument("--window-len", type=int, default=2_097_152)
    parser.add_argument("--window-step", type=int, default=262_144)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    (out_dir / "map").mkdir(parents=True, exist_ok=True)
    (out_dir / "loop").mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, device)
    chroms = args.chroms or default_chroms()

    for chrom in chroms:
        seq_arr = np.load(resolve_chr_file(args.seq_dir, chrom), mmap_mode="r")
        chip_arr = np.load(resolve_chr_file(args.chip_dir, chrom), mmap_mode="r")
        chrom_len = min(seq_arr.shape[1], chip_arr.shape[0])

        start = 0
        while start + args.window_len <= chrom_len:
            end = start + args.window_len
            print(f"predicting {chrom}:{start}-{end}")
            contact_map, loop = predict_window(model, seq_arr, chip_arr, start, args.window_len, device)
            np.save(out_dir / "map" / f"{chrom}_{start}_{end}.npy", contact_map)
            np.save(out_dir / "loop" / f"{chrom}_{start}_{end}.npy", loop)
            start += args.window_step
            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
