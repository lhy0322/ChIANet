#!/usr/bin/env python3
import argparse
from pathlib import Path


HG38_SIZES = {
    "chr1": 248956422, "chr2": 242193529, "chr3": 198295559, "chr4": 190214555,
    "chr5": 181538259, "chr6": 170805979, "chr7": 159345973, "chr8": 145138636,
    "chr9": 138394717, "chr10": 133797422, "chr11": 135086622, "chr12": 133275309,
    "chr13": 114364328, "chr14": 107043718, "chr15": 101991189, "chr16": 90338345,
    "chr17": 83257441, "chr18": 80373285, "chr19": 58617616, "chr20": 64444167,
    "chr21": 46709983, "chr22": 50818468, "chrX": 156040895,
}


def load_sizes(path):
    sizes = {}
    with open(path) as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            chrom, size = line.rstrip("\n").split()[:2]
            sizes[chrom] = int(size)
    return sizes


def main():
    parser = argparse.ArgumentParser(description="Generate a BED file of sliding windows for ChIANet training.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--chrom-sizes", default=None, help="Two-column chrom sizes file. Default: built-in hg38.")
    parser.add_argument("--chroms", nargs="*", default=None, help="Default: chr1-chr22 chrX.")
    parser.add_argument("--window-len", type=int, default=2_100_000)
    parser.add_argument("--step", type=int, default=500_000)
    args = parser.parse_args()

    sizes = load_sizes(args.chrom_sizes) if args.chrom_sizes else HG38_SIZES
    chroms = args.chroms or [f"chr{i}" for i in range(1, 23)] + ["chrX"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as handle:
        for chrom in chroms:
            chrom_len = sizes[chrom]
            start = 0
            while start + args.window_len <= chrom_len:
                handle.write(f"{chrom}\t{start}\t{start + args.window_len}\n")
                start += args.step
    print(f"saved {out}")


if __name__ == "__main__":
    main()

