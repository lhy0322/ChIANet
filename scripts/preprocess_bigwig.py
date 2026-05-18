#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pyBigWig


def normalize_chroms(chroms):
    if chroms:
        return chroms
    return [f"chr{i}" for i in range(1, 23)] + ["chrX"]


def main():
    parser = argparse.ArgumentParser(description="Convert a BigWig track to per-chromosome .npy files.")
    parser.add_argument("--bigwig", required=True, help="Input BigWig path.")
    parser.add_argument("--out-dir", required=True, help="Output directory for chr*.npy files.")
    parser.add_argument("--chroms", nargs="*", default=None, help="Chromosomes to export. Default: chr1-chr22 chrX.")
    parser.add_argument("--no-log2", action="store_true", help="Disable log2(x + 1) transform.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = set(normalize_chroms(args.chroms))
    bw = pyBigWig.open(args.bigwig)
    try:
        for chrom, length in bw.chroms().items():
            if chrom not in targets:
                continue
            arr = np.array(bw.values(chrom, 0, length, numpy=True), dtype=np.float32)
            arr = np.nan_to_num(arr, copy=False)
            if not args.no_log2:
                arr = np.log2(arr + 1).astype(np.float32, copy=False)
            out_path = out_dir / f"{chrom}.npy"
            np.save(out_path, arr)
            print(f"saved {out_path} shape={arr.shape} dtype={arr.dtype}")
    finally:
        bw.close()


if __name__ == "__main__":
    main()

