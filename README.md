# ChIANet Core

This repository contains the minimal ChIANet code needed for preprocessing, training, and inference:

- `model/chianet_fast_v1.py`: ChIANet model architecture.
- `scripts/preprocess_fasta.py`: convert reference FASTA to per-chromosome one-hot `.npy`.
- `scripts/preprocess_bigwig.py`: convert BigWig signal to per-chromosome `.npy`.
- `scripts/preprocess_map.py`: convert `.map` contact maps to per-chromosome band `.npy`.
- `scripts/make_windows_bed.py`: generate sliding-window BED files for training.
- `scripts/train_chianet.py`: train ChIANet from scratch.
- `scripts/predict_window.py`: predict one genomic window.
- `scripts/predict_genome.py`: run sliding-window prediction across chromosomes.

## Install

```bash
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build if GPU inference is needed.

## Preprocess Inputs

Convert genome sequence:

```bash
python scripts/preprocess_fasta.py \
  --fasta /path/to/hg38.fa \
  --out-dir data/sequence
```

Convert a ChIP-seq or other genomic signal BigWig:

```bash
python scripts/preprocess_bigwig.py \
  --bigwig /path/to/signal.bw \
  --out-dir data/chip
```

Convert a `.map` file for training:

```bash
python scripts/preprocess_map.py \
  --map /path/to/contact_map.map \
  --out-dir data/map \
  --resolution 10000
```

The model expects:

- sequence arrays shaped `(5, L)`, saved as `chr*.npy`
- signal arrays shaped `(L,)`, saved as `chr*.npy`
- contact map band arrays shaped `(421, n_bins)` by default, saved as `chr*.npy`

## Train From Scratch

Generate a sliding-window BED:

```bash
python scripts/make_windows_bed.py \
  --out data/hg38_windows.bed \
  --window-len 2100000 \
  --step 500000
```

Train ChIANet:

```bash
python scripts/train_chianet.py \
  --seq-dir data/sequence \
  --chip-dir data/chip \
  --map-dir data/map \
  --loop-file /path/to/loops.bedpe.gz \
  --bed data/hg38_windows.bed \
  --checkpoint checkpoints/chianet_from_scratch.pth \
  --log-file logs/chianet_from_scratch.tsv \
  --batch-size 8 \
  --num-workers 8
```

The loop file should be BEDPE-like with at least six columns:
`chr1 start1 end1 chr2 start2 end2`. Gzipped files are supported.

## Single-Window Inference

```bash
python scripts/predict_window.py \
  --seq-dir data/sequence \
  --chip-dir data/chip \
  --checkpoint checkpoints/chianet_ctcf_gm12878_rpgc_v1.pth \
  --chrom chr1 \
  --start 0 \
  --out-dir outputs/example
```

This writes `outputs/example/map/chr1_0_2097152.npy` and
`outputs/example/loop/chr1_0_2097152.npy`.

## Genome-Wide Sliding-Window Inference

```bash
python scripts/predict_genome.py \
  --seq-dir data/sequence \
  --chip-dir data/chip \
  --checkpoint checkpoints/chianet_ctcf_gm12878_rpgc_v1.pth \
  --out-dir outputs/genome \
  --chroms chr1 chr2 chrX \
  --window-step 262144
```

Default chromosomes are `chr1`-`chr22` and `chrX`. Default window length is
`2,097,152 bp`, producing `256 x 256` contact map and loop score matrices.

## Checkpoints

Pretrained ChIANet checkpoints are available on Google Drive:

https://drive.google.com/drive/folders/1EZ3IDZiEzpeIivL-cBDgvH5Ii0_8IUZA?usp=drive_link

The folder contains trained models for three protein-mediated chromatin contacts:
CTCF, Cohesin, and RNAPII. Download the needed `.pth` file and place it under
`checkpoints/`, for example `checkpoints/chianet_ctcf_gm12878_rpgc_v1.pth`.

Large checkpoint files are intentionally not included in the code-only release
directory.
