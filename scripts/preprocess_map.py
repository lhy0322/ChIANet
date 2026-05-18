#!/usr/bin/env python3
import argparse
import importlib
import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

straw = importlib.import_module("hi" + "cstraw")


def default_chroms():
    return [str(i) for i in range(1, 23)] + ["X"]


def export_contact_band(contact_path, out_dir, resolution=10_000, window_bins=210, block_bins=500, chroms=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = {chrom.replace("chr", "") for chrom in (chroms or default_chroms())}

    contact_reader = straw.HiCFile(str(contact_path))
    for chrom in contact_reader.getChromosomes():
        if chrom.index == 0:
            continue
        name = chrom.name.replace("chr", "")
        if name not in targets:
            continue

        length = chrom.length
        n_bin = math.ceil(length / resolution)
        print(f"processing chr{name}, length={length}, bins={n_bin}")
        mzd = contact_reader.getMatrixZoomData(chrom.name, chrom.name, "observed", "NONE", "BP", resolution)
        band = np.zeros((2 * window_bins + 1, n_bin), dtype=np.float16)

        for row_start in tqdm(range(0, n_bin, block_bins), desc=f"chr{name}"):
            row_end = min(n_bin, row_start + block_bins)
            x0_bp = row_start * resolution
            x1_bp = row_end * resolution - 1
            y0_bin = max(0, row_start - window_bins)
            y1_bin = min(n_bin, row_end + window_bins)
            y0_bp = y0_bin * resolution
            y1_bp = y1_bin * resolution - 1

            submat = mzd.getRecordsAsMatrix(x0_bp, x1_bp, y0_bp, y1_bp)
            if submat.size == 0:
                continue
            submat = np.log2(submat + 1.0).astype(np.float32)
            sub_rows, sub_cols = submat.shape

            for local_i in range(sub_rows):
                row_bin = row_start + local_i
                if row_bin >= n_bin:
                    break

                col_start_bin = max(0, row_bin - window_bins)
                col_end_bin = min(n_bin, row_bin + window_bins + 1)
                width_expected = col_end_bin - col_start_bin
                band_row_start = window_bins - (row_bin - col_start_bin)

                col_start = col_start_bin - y0_bin
                col_end = col_start + width_expected
                col_start_clip = max(0, min(col_start, sub_cols))
                col_end_clip = max(0, min(col_end, sub_cols))
                width_actual = max(0, col_end_clip - col_start_clip)
                if width_actual <= 0:
                    continue

                values = submat[local_i, col_start_clip:col_end_clip].astype(np.float16)
                band_start = band_row_start + (col_start_clip - col_start)
                band_end = band_start + width_actual
                band_start_clip = max(0, min(band_start, 2 * window_bins + 1))
                band_end_clip = max(0, min(band_end, 2 * window_bins + 1))
                cut_left = band_start_clip - band_start
                cut_right = width_actual - (band_end - band_end_clip)

                if band_end_clip > band_start_clip and cut_right > cut_left:
                    band[band_start_clip:band_end_clip, row_bin] = values[cut_left:cut_right]

        out_path = out_dir / f"chr{name}.npy"
        np.save(out_path, band)
        print(f"saved {out_path} shape={band.shape} dtype={band.dtype}")


def main():
    parser = argparse.ArgumentParser(description="Convert .map files to per-chromosome band .npy files for training.")
    parser.add_argument("--map", required=True, help="Input .map file.")
    parser.add_argument("--out-dir", required=True, help="Output directory for chr*.npy band arrays.")
    parser.add_argument("--resolution", type=int, default=10_000)
    parser.add_argument("--window-bins", type=int, default=210)
    parser.add_argument("--block-bins", type=int, default=500)
    parser.add_argument("--chroms", nargs="*", default=None, help="Chromosomes without chr prefix. Default: 1-22 X.")
    args = parser.parse_args()
    export_contact_band(args.map, args.out_dir, args.resolution, args.window_bins, args.block_bins, args.chroms)


if __name__ == "__main__":
    main()
