#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pysam


def build_byte2idx():
    table = np.full(256, 4, dtype=np.uint8)
    for base, idx in (("A", 0), ("T", 1), ("C", 2), ("G", 3), ("N", 4)):
        table[ord(base)] = idx
        table[ord(base.lower())] = idx
    return table


def seq_to_onehot_uint8(seq, byte2idx):
    bases = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    idx = byte2idx[bases]
    onehot = np.zeros((5, idx.size), dtype=np.uint8)
    onehot[idx, np.arange(idx.size)] = 1
    return onehot


def normalize_chroms(chroms):
    if chroms:
        return chroms
    return [f"chr{i}" for i in range(1, 23)] + ["chrX"]


def main():
    parser = argparse.ArgumentParser(description="Convert a reference FASTA to per-chromosome one-hot .npy files.")
    parser.add_argument("--fasta", required=True, help="Reference FASTA path, indexed by samtools faidx or pysam.")
    parser.add_argument("--out-dir", required=True, help="Output directory for chr*.npy files.")
    parser.add_argument("--chroms", nargs="*", default=None, help="Chromosomes to export. Default: chr1-chr22 chrX.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fasta = pysam.FastaFile(args.fasta)
    refs = set(fasta.references)
    targets = [chrom for chrom in normalize_chroms(args.chroms) if chrom in refs]
    if not targets:
        raise RuntimeError("No requested chromosomes were found in the FASTA.")

    byte2idx = build_byte2idx()
    for chrom in targets:
        seq = fasta.fetch(chrom)
        onehot = seq_to_onehot_uint8(seq, byte2idx)
        out_path = out_dir / f"{chrom}.npy"
        np.save(out_path, onehot)
        print(f"saved {out_path} shape={onehot.shape} dtype={onehot.dtype}")


if __name__ == "__main__":
    main()

