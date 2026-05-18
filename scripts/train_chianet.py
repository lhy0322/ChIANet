#!/usr/bin/env python3
import argparse
import math
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.chianet_fast_v1 import DNASequenceModel, UpperTri  # noqa: E402


def normalize_checkpoint_keys(state):
    old = "h" + "ic"
    renamed = {}
    for key, value in state.items():
        key = key.replace(f"conv_end_{old}", "conv_end_map")
        key = key.replace(f"log_sigma_{old}", "log_sigma_map")
        renamed[key] = value
    return renamed


class LinearWarmupCosineAnnealingLR:
    def __init__(self, optimizer, warmup_epochs, max_epochs, warmup_start_lr, eta_min, num_batches_per_epoch):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        self.num_batches_per_epoch = max(1, num_batches_per_epoch)
        self.total_steps = max_epochs * self.num_batches_per_epoch
        for group in self.optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])

    def _get_lr(self, step):
        warmup_steps = self.warmup_epochs * self.num_batches_per_epoch
        if warmup_steps > 0 and step < warmup_steps:
            base_lr = self.optimizer.param_groups[0]["initial_lr"]
            return self.warmup_start_lr + (base_lr - self.warmup_start_lr) * (step / warmup_steps)
        adjusted = max(0, step - warmup_steps)
        total_adjusted = max(1, self.total_steps - warmup_steps)
        cos_out = (1 + math.cos(math.pi * adjusted / total_adjusted)) / 2
        base_lr = self.optimizer.param_groups[0]["initial_lr"]
        return self.eta_min + (base_lr - self.eta_min) * cos_out

    def step(self, step):
        lr = self._get_lr(step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self):
        return {
            "warmup_epochs": self.warmup_epochs,
            "max_epochs": self.max_epochs,
            "warmup_start_lr": self.warmup_start_lr,
            "eta_min": self.eta_min,
            "num_batches_per_epoch": self.num_batches_per_epoch,
            "total_steps": self.total_steps,
        }

    def load_state_dict(self, state):
        self.warmup_epochs = state["warmup_epochs"]
        self.max_epochs = state["max_epochs"]
        self.warmup_start_lr = state["warmup_start_lr"]
        self.eta_min = state["eta_min"]
        self.num_batches_per_epoch = state["num_batches_per_epoch"]
        self.total_steps = state["total_steps"]


class ChIANetDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        bed_file,
        seq_dir,
        chip_dir,
        map_dir,
        loop_file,
        chroms,
        is_train,
        resolution=10_000,
        sequence_len=2_097_152,
        bin_size=8192,
        num_bins=256,
        map_window_bins=210,
        random_shift=360_000,
    ):
        self.resolution = resolution
        self.chroms = chroms
        self.seq_len = sequence_len
        self.is_train = is_train
        self.bin_size = bin_size
        self.num_bins = num_bins
        self.map_window_bins = map_window_bins
        self.random_shift = random_shift
        self.upper_tri = UpperTri()

        self.seq_dict = {chrom: np.load(Path(seq_dir) / f"{chrom}.npy", mmap_mode="r") for chrom in chroms}
        self.bw_dict = {chrom: np.load(Path(chip_dir) / f"{chrom}.npy", mmap_mode="r") for chrom in chroms}
        self.band_dict = {chrom: np.load(Path(map_dir) / f"{chrom}.npy", mmap_mode="r") for chrom in chroms}
        self.chr_len = {chrom: min(self.seq_dict[chrom].shape[1], self.bw_dict[chrom].shape[0]) for chrom in chroms}

        self.bed_df = pd.read_csv(bed_file, sep="\t", header=None, names=["chrom", "start", "end"])
        self.bed_df = self.bed_df[self.bed_df["chrom"].isin(chroms)].reset_index(drop=True)
        if self.bed_df.empty:
            raise ValueError("No BED windows remain after chromosome filtering.")

        self.loop_index = self._preprocess_loops(loop_file)
        self.label_cache = {}

    def __len__(self):
        return len(self.bed_df)

    def _preprocess_loops(self, loop_file):
        loops = pd.read_csv(loop_file, sep="\t", header=None, comment="#", compression="infer")
        if loops.shape[1] < 6:
            raise ValueError("Loop file must have at least 6 BEDPE columns.")
        loops = loops.iloc[:, :6].copy()
        loops.columns = ["chr1", "start1", "end1", "chr2", "start2", "end2"]
        loops = loops[(loops["chr1"] == loops["chr2"]) & (loops["chr1"].isin(self.chroms))].copy()

        index = {}
        for chrom, group in loops.groupby("chr1"):
            chrom_index = {}
            s1 = (group["start1"] // self.bin_size).to_numpy(dtype=np.int64)
            e1 = ((group["end1"] - 1) // self.bin_size).to_numpy(dtype=np.int64)
            s2 = (group["start2"] // self.bin_size).to_numpy(dtype=np.int64)
            e2 = ((group["end2"] - 1) // self.bin_size).to_numpy(dtype=np.int64)
            for b1s, b1e, b2s, b2e in zip(s1, e1, s2, e2):
                if b1e < b1s or b2e < b2s:
                    continue
                for b1 in range(b1s, b1e + 1):
                    chrom_index.setdefault(b1, set()).update(range(b2s, b2e + 1))
            index[chrom] = chrom_index
        return index

    def _build_label_window(self, chrom, start):
        key = (chrom, start)
        if not self.is_train and key in self.label_cache:
            return self.label_cache[key]

        sbin = start // self.bin_size
        ebin = (start + self.seq_len) // self.bin_size
        mat = np.zeros((ebin - sbin, ebin - sbin), dtype=np.uint8)
        for i_global in range(sbin, ebin):
            partners = self.loop_index.get(chrom, {}).get(i_global)
            if not partners:
                continue
            i_local = i_global - sbin
            for j_global in partners:
                if sbin <= j_global < ebin:
                    j_local = j_global - sbin
                    if j_local > i_local:
                        mat[i_local, j_local] = 1
        mat = mat + mat.T
        if not self.is_train:
            self.label_cache[key] = mat
        return mat

    def _reconstruct_window_from_band(self, band, start_bp, end_bp):
        start_bin = start_bp // self.resolution
        end_bin = int(np.ceil(end_bp / self.resolution))
        length = end_bin - start_bin
        matrix = np.zeros((length, length), dtype=np.float32)
        window = self.map_window_bins
        for i in range(length):
            row_bin = start_bin + i
            offsets = (np.arange(length) - i) + window
            valid = (offsets >= 0) & (offsets < band.shape[0])
            if not np.any(valid):
                continue
            cols = np.nonzero(valid)[0]
            matrix[i, cols] = band[offsets[cols].astype(np.int64), row_bin].astype(np.float32)
        return matrix

    @staticmethod
    def _reverse_complement(onehot):
        idx_map = torch.tensor([1, 0, 3, 2, 4], dtype=torch.long, device=onehot.device)
        return torch.flip(onehot.index_select(0, idx_map), dims=[-1])

    def __getitem__(self, idx):
        row = self.bed_df.iloc[idx]
        chrom = row["chrom"]
        start = int(row["start"])

        if self.is_train and self.random_shift > 0:
            start = max(0, start + random.randint(-self.random_shift, self.random_shift))
            if start + self.seq_len > self.chr_len[chrom]:
                start = self.chr_len[chrom] - self.seq_len
            start = max(0, start)

        end = start + self.seq_len
        seq = torch.from_numpy(self.seq_dict[chrom][:, start:end].copy()).float()
        chip = torch.from_numpy(self.bw_dict[chrom][start:end].copy()).float().unsqueeze(0)

        map_matrix = self._reconstruct_window_from_band(self.band_dict[chrom], start, end)
        map_tensor = torch.from_numpy(map_matrix).unsqueeze(0).unsqueeze(0)
        map_tensor = F.interpolate(map_tensor, size=(self.num_bins, self.num_bins), mode="bilinear", align_corners=False)

        label = self._build_label_window(chrom, start)
        label_tensor = torch.from_numpy(label).float().unsqueeze(0).unsqueeze(0)

        if self.is_train and random.random() < 0.5:
            seq = self._reverse_complement(seq)
            chip = torch.flip(chip, dims=[-1])
            map_tensor = map_tensor.transpose(-2, -1).flip(-1).flip(-2)
            label_tensor = label_tensor.transpose(-2, -1).flip(-1).flip(-2)

        return (
            seq,
            chip,
            self.upper_tri(map_tensor).squeeze(0).squeeze(0),
            self.upper_tri(label_tensor).squeeze(0).squeeze(0),
        )


class EarlyStopping:
    def __init__(self, patience=10, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def update(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            return True
        self.counter += 1
        self.should_stop = self.counter >= self.patience
        return False


def save_checkpoint(model, optimizer, scheduler, epoch, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        path,
    )


def load_checkpoint(model, optimizer, scheduler, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(normalize_checkpoint_keys(checkpoint["model_state_dict"]), strict=False)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["epoch"]


def multitask_loss(pred_map, true_map, pred_loop, true_loop, log_sigma_map, log_sigma_loop, mse, bce):
    loss_map = mse(pred_map, true_map)
    loss_loop = bce(pred_loop, true_loop)
    total = (1 / (2 * torch.exp(log_sigma_map))) * loss_map
    total = total + (1 / (2 * torch.exp(log_sigma_loop))) * loss_loop + log_sigma_map + log_sigma_loop
    return total, loss_map, loss_loop


def validate(model, loader, device, mse, bce):
    model.eval()
    total = map_total = loop_total = 0.0
    with torch.no_grad():
        for seq, chip, target_map, label in tqdm(loader, desc="validation", leave=False):
            seq = seq.to(device, non_blocking=True)
            chip = chip.to(device, non_blocking=True)
            target_map = target_map.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            pred_map, pred_loop, log_sigma_map, log_sigma_loop = model(seq, chip)
            loss, loss_map, loss_loop = multitask_loss(
                pred_map, target_map, pred_loop, label, log_sigma_map, log_sigma_loop, mse, bce
            )
            total += loss.item()
            map_total += loss_map.item()
            loop_total += loss_loop.item()
    n = max(1, len(loader))
    return total / n, map_total / n, loop_total / n


def parse_chroms(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def make_loader(dataset, batch_size, shuffle, num_workers):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs.update({"persistent_workers": True, "prefetch_factor": 4})
    return torch.utils.data.DataLoader(dataset, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Train ChIANet from scratch.")
    parser.add_argument("--seq-dir", required=True)
    parser.add_argument("--chip-dir", required=True)
    parser.add_argument("--map-dir", required=True, help="Directory from preprocess_map.py.")
    parser.add_argument("--loop-file", required=True, help="BEDPE loop labels, optionally gzipped.")
    parser.add_argument("--bed", required=True, help="Training window BED.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--train-chroms", default="chr1,chr2,chr4,chr6,chr7,chr8,chr9,chr10,chr11,chr12,chr13,chr14,chr16,chr17,chr19,chr20,chr21,chr22,chrX")
    parser.add_argument("--val-chroms", default="chr3,chr15")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    train_chroms = parse_chroms(args.train_chroms)
    val_chroms = parse_chroms(args.val_chroms)

    train_data = ChIANetDataset(args.bed, args.seq_dir, args.chip_dir, args.map_dir, args.loop_file, train_chroms, True)
    val_data = ChIANetDataset(args.bed, args.seq_dir, args.chip_dir, args.map_dir, args.loop_file, val_chroms, False)
    train_loader = make_loader(train_data, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_data, args.batch_size, False, args.num_workers)

    model = DNASequenceModel(num_genomic_features=1, mid_hidden=256, record_attn=False).to(device)
    mse = nn.MSELoss().to(device)
    bce = nn.BCELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
        warmup_start_lr=1e-5,
        eta_min=1e-6,
        num_batches_per_epoch=len(train_loader),
    )
    early_stop = EarlyStopping(patience=args.patience)

    start_epoch = 0
    if args.resume and Path(args.checkpoint).exists():
        start_epoch = load_checkpoint(model, optimizer, scheduler, args.checkpoint, device)

    log_path = Path(args.log_file) if args.log_file else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as handle:
            handle.write(f"# training started {datetime.now().isoformat()}\n")
            handle.write("epoch\tval_loss\tmap_loss\tloop_loss\tsaved\n")

    step = start_epoch * len(train_loader)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}", leave=False)
        for seq, chip, target_map, label in progress:
            step += 1
            seq = seq.to(device, non_blocking=True)
            chip = chip.to(device, non_blocking=True)
            target_map = target_map.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred_map, pred_loop, log_sigma_map, log_sigma_loop = model(seq, chip)
            loss, loss_map, loss_loop = multitask_loss(
                pred_map, target_map, pred_loop, label, log_sigma_map, log_sigma_loop, mse, bce
            )
            loss.backward()
            optimizer.step()
            scheduler.step(step)
            progress.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "map": f"{loss_map.item():.4f}",
                    "loop": f"{loss_loop.item():.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
            )

        val_loss, val_map, val_loop = validate(model, val_loader, device, mse, bce)
        saved = early_stop.update(val_loss)
        if saved:
            save_checkpoint(model, optimizer, scheduler, epoch, args.checkpoint)

        line = f"{epoch + 1}\t{val_loss:.6f}\t{val_map:.6f}\t{val_loop:.6f}\t{int(saved)}"
        print(line)
        if log_path:
            with open(log_path, "a") as handle:
                handle.write(line + "\n")
        if early_stop.should_stop:
            print("early stopping")
            break


if __name__ == "__main__":
    main()
