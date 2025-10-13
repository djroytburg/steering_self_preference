#!/usr/bin/env python3
"""
train_probe.py (stratified, mask-aware) + ROC plotting
------------------------------------------------------
- Stratified train/val split to avoid single-class validation (keeps AUC defined).
- Ensures non-empty splits even for tiny datasets.
- Passes attn_mask to TransformerProbe.
- Prints class balance for train/val.
- (NEW) Optional Recall-vs-FPR (ROC) plot from the validation set:
    --plot-roc --roc-out roc_val.png --roc-xlim 0.02
    Also saves raw ROC arrays next to the image as <basename>_roc.npz.
"""

from __future__ import annotations
import argparse
import json
import os
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

# For plotting (only used if --plot-roc)
import matplotlib
matplotlib.use("Agg")  # safe in headless; no effect if you open later
import matplotlib.pyplot as plt

# Local import
import sys
sys.path.append(os.path.dirname(__file__))
from probes import LinearProbe, MLPProbe, TransformerProbe  # type: ignore

# ---------------------- Utils ----------------------
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)

def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}

def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC-AUC for binary labels using the Mann–Whitney U statistic.
    Returns NaN if labels are all the same."""
    y_true = y_true.astype(np.int32)
    if y_true.size == 0 or y_true.min() == y_true.max():
        return float('nan')
    # Rank scores (average ranks for ties)
    order = np.argsort(y_score, kind='mergesort')
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(y_score[order], return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    start = cum - counts + 1
    avg = (start + cum) / 2.0
    ranks[order] = avg[inv]
    # Compute AUC
    n1 = int((y_true == 1).sum())
    n0 = int((y_true == 0).sum())
    if n0 == 0 or n1 == 0:
        return float('nan')
    sum_ranks_pos = ranks[y_true == 1].sum()
    U1 = sum_ranks_pos - n1 * (n1 + 1) / 2.0
    auc = U1 / (n0 * n1)
    return float(auc)

def roc_curve_points(y_true: np.ndarray, y_score: np.ndarray):
    """Return (fpr, tpr, thresholds) for binary ROC without sklearn.
    thresholds are unique sorted score values (descending), with an extra +inf row to start at (0,0)."""
    y = y_true.astype(np.int32)
    s = y_score.astype(np.float64)

    if y.size == 0 or y.min() == y.max():
        # Degenerate: no curve possible
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([np.inf, -np.inf])

    # Sort by score descending; keep ties stable
    order = np.argsort(-s, kind='mergesort')
    s_sorted = s[order]
    y_sorted = y[order]

    # Unique thresholds (descending)
    thresh, idx = np.unique(s_sorted, return_index=True)
    thresh = thresh[::-1]
    idx = idx[::-1]

    # Prepend +inf to start at (0,0)
    thresholds = np.concatenate([[np.inf], thresh])

    P = float((y == 1).sum())
    N = float((y == 0).sum())

    # Cumulative TP/FP when threshold moves down across unique indices
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)

    # At each threshold index-1, compute rates; prepend zeros for +inf
    tpr = np.concatenate([[0.0], tp[idx - 1] / max(P, 1.0)])
    fpr = np.concatenate([[0.0], fp[idx - 1] / max(N, 1.0)])

    # Ensure final point (1,1)
    if tpr[-1] < 1.0 or fpr[-1] < 1.0:
        tpr = np.concatenate([tpr, [1.0]])
        fpr = np.concatenate([fpr, [1.0]])
        thresholds = np.concatenate([thresholds, [-np.inf]])

    return fpr, tpr, thresholds

def plot_recall_vs_fpr(fpr: np.ndarray, tpr: np.ndarray, out_path: str, xlim: Optional[float] = None, label: Optional[str] = None):
    plt.figure(figsize=(6,5))
    if label is None:
        plt.plot(fpr, tpr, linewidth=2)
    else:
        plt.plot(fpr, tpr, linewidth=2, label=label)
        plt.legend(loc="lower right", frameon=True)
    # Chance line
    plt.plot([0,1],[0,1], "--", linewidth=1, color="black", alpha=0.5, label=None)
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("Recall (TPR)")
    plt.title("Recall vs FPR (ROC)")
    plt.grid(True, alpha=0.3)
    if xlim is not None:
        plt.xlim(0.0, float(xlim))
        plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

# ---------------------- Dataset ----------------------
class ProbeDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, mask: Optional[np.ndarray] = None):
        assert x.ndim in (2, 3), "x must be (N,D) or (N,T,D)"
        assert y.ndim == 1 and len(y) == len(x)
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y).long()
        self.mask = torch.from_numpy(mask) if mask is not None else None

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx: int):
        item = {"x": self.x[idx], "y": self.y[idx]}
        if self.mask is not None:
            item["mask"] = self.mask[idx]
        return item

def load_data(path: str, x_key: str = "x", y_key: str = "y", mask_key: Optional[str] = "mask"):
    if path.endswith(".npz"):
        data = np.load(path, allow_pickle=True)
        x = data[x_key]
        y = data[y_key]
        m = data[mask_key] if (mask_key is not None and mask_key in data) else None
    elif path.endswith(".pt"):
        data = torch.load(path, map_location="cpu")
        x = data[x_key].numpy() if isinstance(data[x_key], torch.Tensor) else np.asarray(data[x_key])
        y = data[y_key].numpy() if isinstance(data[y_key], torch.Tensor) else np.asarray(data[y_key])
        if mask_key is not None and mask_key in data:
            m_t = data[mask_key]
            m = m_t.numpy() if isinstance(m_t, torch.Tensor) else np.asarray(m_t)
        else:
            m = None
    else:
        raise ValueError("Unsupported data file. Use .npz or .pt")
    x = x.astype(np.float32)
    y = y.astype(np.int64)
    if m is not None:
        m = m.astype(np.int64)
    return x, y, m

# ---------------------- Stratified split ----------------------
def stratified_indices(y: np.ndarray, val_frac: float, seed: int) -> Tuple[List[int], List[int]]:
    """
    Create stratified train/val indices so that both splits contain both classes when possible.
    Ensures both splits are non-empty if len(y) >= 2.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    # Compute desired val size with sanity bounds
    val_size = int(round(n * val_frac))
    val_size = max(1 if n >= 2 else 0, min(val_size, n - 1 if n >= 2 else n))

    # If only one class exists, just do a simple split (AUC will be NaN by definition)
    classes = np.unique(y)
    if len(classes) < 2:
        all_idx = np.arange(n)
        rng.shuffle(all_idx)
        val_idx = list(all_idx[:val_size]) if val_size > 0 else []
        train_idx = list(all_idx[val_size:]) if val_size < n else []
        return train_idx, val_idx

    # Stratify
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    rng.shuffle(idx0); rng.shuffle(idx1)

    # target per-class in val, at least 1 if possible
    n0, n1 = len(idx0), len(idx1)
    val0 = max(1, int(round(val_size * n0 / (n0 + n1)))) if val_size > 1 else 1
    val0 = min(val0, n0)  # can't exceed available
    val1 = val_size - val0
    if val1 < 1 and n1 > 0 and val_size > 0:
        val1 = 1
        if val0 > 1:
            val0 -= 1
    val1 = min(val1, n1)

    val_idx = list(idx0[:val0]) + list(idx1[:val1])
    train_idx = list(np.setdiff1d(np.arange(n), val_idx, assume_unique=False))

    # Final sanity: non-empty splits
    if len(train_idx) == 0 and len(val_idx) > 1:
        train_idx = [val_idx.pop()]  # move one back to train
    if len(val_idx) == 0 and len(train_idx) > 1:
        val_idx = [train_idx.pop()]

    return train_idx, val_idx

# ---------------------- Build probe ----------------------
def build_probe(args, input_dim: int):
    if args.probe == "linear":
        return LinearProbe(input_dim=input_dim, pooling=args.pooling, bias=True)
    elif args.probe == "mlp":
        return MLPProbe(
            input_dim=input_dim,
            hidden_dims=tuple(args.hidden_dims),
            activation=args.activation,
            dropout=args.dropout,
            pooling=args.pooling,
            layernorm=args.layernorm,
        )
    elif args.probe == "transformer":
        return TransformerProbe(
            input_dim=input_dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            mlp_ratio=args.mlp_ratio,
            dropout=args.dropout,
            add_cls_token=not args.no_cls,
            pooling=args.pooling,
        )
    else:
        raise ValueError(f"Unknown probe: {args.probe}")

# ---------------------- Eval ----------------------
@torch.no_grad()
def evaluate(model, loader, device, return_arrays: bool = False):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    logits_all, labels_all = [], []
    loss_fn = nn.BCEWithLogitsLoss()
    for batch in loader:
        batch = to_device(batch, device)
        x = batch["x"]
        y = batch["y"].float()
        mask = batch.get("mask")
        # Pass mask only when the model accepts it (TransformerProbe)
        if isinstance(model, TransformerProbe) and mask is not None:
            logits = model(x, attn_mask=mask)
        else:
            logits = model(x)
        loss = loss_fn(logits, y)
        preds = (logits.sigmoid() >= 0.5).long()
        total += y.numel()
        correct += (preds == y.long()).sum().item()
        loss_sum += loss.item() * y.numel()
        logits_all.append(logits.detach().cpu())
        labels_all.append(y.detach().cpu())
    logits_all = torch.cat(logits_all).numpy() if logits_all else np.array([])
    labels_all = torch.cat(labels_all).numpy() if labels_all else np.array([])
    acc = correct / total if total > 0 else 0.0
    auc = roc_auc_score(labels_all, logits_all) if total > 0 else float('nan')
    out = {"loss": loss_sum / max(total, 1), "acc": acc, "auc": auc}
    if return_arrays:
        out["logits"] = logits_all
        out["labels"] = labels_all
    return out

# ---------------------- Train ----------------------
def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    x, y, m = load_data(args.data, x_key=args.x_key, y_key=args.y_key, mask_key=args.mask_key)
    N = len(x)
    input_dim = x.shape[-1]
    dataset = ProbeDataset(x, y, m)

    # Stratified split
    train_idx, val_idx = stratified_indices(y, args.val_frac, args.seed)
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    def _counts(ds: Subset) -> Dict[int, int]:
        if len(ds) == 0:
            return {0: 0, 1: 0}
        ys = torch.stack([dataset[i]["y"] for i in ds.indices]).numpy()
        return {0: int((ys == 0).sum()), 1: int((ys == 1).sum())}

    train_counts = _counts(train_ds)
    val_counts = _counts(val_ds)

    print(f"[split] N={N}  train={len(train_ds)} (y0={train_counts[0]}, y1={train_counts[1]}), "
          f"val={len(val_ds)} (y0={val_counts[0]}, y1={val_counts[1]})")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = build_probe(args, input_dim=input_dim).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val = float('inf')
    best_metrics = {}
    patience = args.patience
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch in train_loader:
            batch = to_device(batch, device)
            x = batch["x"]
            yb = batch["y"].float()
            mask = batch.get("mask")
            if isinstance(model, TransformerProbe) and mask is not None:
                logits = model(x, attn_mask=mask)
            else:
                logits = model(x)
            loss = loss_fn(logits, yb)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optim.step()

            running_loss += loss.item() * yb.numel()
            seen += yb.numel()

        train_loss = running_loss / max(seen, 1)
        val_metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:03d} | train_loss {train_loss:.4f} | "
              f"val_loss {val_metrics['loss']:.4f} | val_acc {val_metrics['acc']:.4f} | val_auc {val_metrics['auc']:.4f}")

        # Early stopping on val loss
        if val_metrics["loss"] + 1e-6 < best_val:
            best_val = val_metrics["loss"]
            best_metrics = {"epoch": epoch, **val_metrics,
                            "train_counts": train_counts, "val_counts": val_counts}
            torch.save({"state_dict": model.state_dict(),
                        "config": vars(args),
                        "input_dim": input_dim,
                        "metrics": best_metrics},
                       args.save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping.")
                break

    print(f"Best checkpoint metrics: {best_metrics}")
    with open(os.path.splitext(args.save_path)[0] + "_metrics.json", "w") as f:
        json.dump(best_metrics, f, indent=2)

    # ------- NEW: create ROC plot from the best model on validation set -------
    if args.plot_roc and len(val_ds) > 0:
        # Reload best (in case we early-stopped)
        ckpt = torch.load(args.save_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()

        val_metrics_arrays = evaluate(model, val_loader, device, return_arrays=True)
        y_val = val_metrics_arrays.get("labels", np.array([]))
        s_val = val_metrics_arrays.get("logits", np.array([]))

        if y_val.size > 0 and y_val.min() != y_val.max():
            fpr, tpr, thresholds = roc_curve_points(y_val, s_val)
            # Save arrays
            base = os.path.splitext(args.roc_out)[0]
            np.savez_compressed(base + "_roc.npz", fpr=fpr, tpr=tpr, thresholds=thresholds)
            # Plot
            plt.axvline(0.01, linestyle=":", linewidth=1)
            plot_recall_vs_fpr(
                fpr, tpr,
                out_path=args.roc_out,
                xlim=args.roc_xlim,
                label="Validation"
            )
            print(f"[ROC] Saved ROC curve to {args.roc_out} and raw arrays to {base}_roc.npz")
        else:
            print("[ROC] Skipped ROC: validation set has a single class or is empty.")

def parse_args():
    p = argparse.ArgumentParser(description="Train a probe on latent states.")
    p.add_argument("--data", type=str, required=True, help="Path to .npz or .pt file with x,y,(mask)")
    p.add_argument("--x-key", type=str, default="x")
    p.add_argument("--y-key", type=str, default="y")
    p.add_argument("--mask-key", type=str, default="mask")

    p.add_argument("--probe", type=str, choices=["linear", "mlp", "transformer"], default="linear")
    p.add_argument("--pooling", type=str, choices=["mean", "last", "max"], default="mean")

    # MLP settings
    p.add_argument("--hidden-dims", type=int, nargs="*", default=[512], help="e.g., --hidden-dims 1024 256")
    p.add_argument("--activation", type=str, default="gelu", choices=["relu", "gelu", "silu", "tanh"])
    p.add_argument("--layernorm", action="store_true")
    # Transformer settings
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--no-cls", action="store_true", help="disable CLS token and use pooling instead")

    # Optimization
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")

    p.add_argument("--save-path", type=str, default="best_probe.pt")

    # -------- NEW: ROC plot options --------
    p.add_argument("--plot-roc", action="store_true",
                   help="If set, saves a Recall (TPR) vs FPR (ROC) plot from the validation set.")
    p.add_argument("--roc-out", type=str, default="roc_val.png",
                   help="Path to save the ROC figure (PNG). Raw arrays saved as <basename>_roc.npz.")
    p.add_argument("--roc-xlim", type=float, default=None,
                   help="Optional max x-axis (FPR) to zoom, e.g., 0.015 for 1.5%%.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)

