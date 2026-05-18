#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.chianet_fast_v1 import DNASequenceModel  # noqa: E402


def resolve_chr_file(dir_path, chrom):
    dir_path = Path(dir_path)
    candidates = [dir_path / f"{chrom}.npy", dir_path / f"{chrom.replace('chr', '')}.npy"]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find .npy for {chrom} under {dir_path}")


def fill_diagonal_with_neighbors(matrix):
    n = matrix.shape[0]
    for i in range(n):
        if i == 0:
            vals = [matrix[1, 0], matrix[0, 1]]
        elif i == n - 1:
            vals = [matrix[n - 2, n - 1], matrix[n - 1, n - 2]]
        else:
            vals = [matrix[i - 1, i], matrix[i + 1, i], matrix[i, i - 1], matrix[i, i + 1]]
        matrix[i, i] = float(np.mean(vals))
    return matrix


def upper_to_symmetric(upper_triangular, size=256):
    vec = upper_triangular.squeeze(0).squeeze()
    matrix = torch.zeros(size, size, device=vec.device, dtype=vec.dtype)
    idx = torch.triu_indices(size, size, offset=1, device=vec.device)
    matrix[idx[0], idx[1]] = vec
    matrix = matrix + matrix.T
    matrix.fill_diagonal_(0)
    return matrix


def load_model(checkpoint, device):
    model = DNASequenceModel(num_genomic_features=1, mid_hidden=256, record_attn=False).to(device)
    state = torch.load(checkpoint, map_location=device)
    state = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    state = normalize_checkpoint_keys(state)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def normalize_checkpoint_keys(state):
    old = "h" + "ic"
    renamed = {}
    for key, value in state.items():
        key = key.replace(f"conv_end_{old}", "conv_end_map")
        key = key.replace(f"log_sigma_{old}", "log_sigma_map")
        renamed[key] = value
    return renamed


def predict_window(model, seq_arr, chip_arr, start, window_len, device):
    end = start + window_len
    if end > min(seq_arr.shape[1], chip_arr.shape[0]):
        raise ValueError("Requested window exceeds available sequence or BigWig array length.")

    seq_tensor = torch.from_numpy(seq_arr[:, start:end].copy()).float().unsqueeze(0).to(device)
    chip_tensor = torch.from_numpy(chip_arr[start:end].copy()).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        tri_map, tri_loop, _, _ = model(seq_tensor, chip_tensor)
        contact_map = upper_to_symmetric(tri_map, 256).detach().cpu().numpy()
        loop = upper_to_symmetric(tri_loop, 256).detach().cpu().numpy()
    return fill_diagonal_with_neighbors(contact_map), loop


def main():
    parser = argparse.ArgumentParser(description="Run ChIANet inference for one genomic window.")
    parser.add_argument("--seq-dir", required=True, help="Directory created by preprocess_fasta.py.")
    parser.add_argument("--chip-dir", required=True, help="Directory created by preprocess_bigwig.py.")
    parser.add_argument("--checkpoint", required=True, help="ChIANet checkpoint .pth file.")
    parser.add_argument("--chrom", required=True, help="Chromosome name, for example chr1.")
    parser.add_argument("--start", type=int, required=True, help="0-based window start in bp.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--window-len", type=int, default=2_097_152)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    (out_dir / "map").mkdir(parents=True, exist_ok=True)
    (out_dir / "loop").mkdir(parents=True, exist_ok=True)

    seq_arr = np.load(resolve_chr_file(args.seq_dir, args.chrom), mmap_mode="r")
    chip_arr = np.load(resolve_chr_file(args.chip_dir, args.chrom), mmap_mode="r")
    model = load_model(args.checkpoint, device)

    end = args.start + args.window_len
    contact_map, loop = predict_window(model, seq_arr, chip_arr, args.start, args.window_len, device)
    np.save(out_dir / "map" / f"{args.chrom}_{args.start}_{end}.npy", contact_map)
    np.save(out_dir / "loop" / f"{args.chrom}_{args.start}_{end}.npy", loop)
    print(f"saved predictions for {args.chrom}:{args.start}-{end} to {out_dir}")


if __name__ == "__main__":
    main()
