
#!/usr/bin/env python3
"""
train_probe.py
---------------
Train a probe (linear, MLP, or transformer) from probes.py on latent states.

Expected data formats
---------------------
The script expects a file containing arrays for inputs and labels.
  - .npz (NumPy): keys: x (N,T,D) or (N,D), y (N,), optional mask (N,T) with 1 for real tokens, 0 for padding
  - .pt  (PyTorch): a dict with the same keys

Minimal example to create a dummy dataset:
    import numpy as np
    N, T, D = 2000, 1, 512
    x = np.random.randn(N, T, D).astype('float32')
    y = (x.mean(axis=(1,2)) > 0).astype('int64')
    np.savez('toy.npz', x=x, y=y)

Usage
-----
    python train_probe.py --data toy.npz --probe mlp --hidden-dims 512 128 --epochs 10

Outputs
-------
  - Best checkpoint saved to --save-path (default: best_probe.pt)
  - A JSON log with final metrics next to the checkpoint
"""
from __future__ import annotations
import argparse
import json
import os
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# Local import
import sys
sys.path.append(os.path.dirname(__file__))
from probes import LinearProbe, MLPProbe, TransformerProbe

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
    if y_true.min() == y_true.max():
        return float('nan')
    # Rank scores (average ranks for ties)
    order = np.argsort(y_score, kind='mergesort')
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    # average ranks for ties
    unique_vals, inv, counts = np.unique(y_score[order], return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    start = cum - counts + 1
    avg = (start + cum) / 2.0
    ranks[order] = avg[inv]
    # Compute AUC
    n1 = (y_true == 1).sum()
    n0 = (y_true == 0).sum()
    sum_ranks_pos = ranks[y_true == 1].sum()
    U1 = sum_ranks_pos - n1 * (n1 + 1) / 2.0
    auc = U1 / (n0 * n1)
    return float(auc)

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
        data = np.load(path)
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

# ---------------------- Training ----------------------
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

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    logits_all, labels_all = [], []
    loss_fn = nn.BCEWithLogitsLoss()
    for batch in loader:
        batch = to_device(batch, device)
        x = batch["x"]
        y = batch["y"].float()
        mask = batch.get("mask")
        logits = model(x) if mask is None else model(x, attn_mask=mask)
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
    return {"loss": loss_sum / max(total, 1), "acc": acc, "auc": auc}

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    x, y, m = load_data(args.data, x_key=args.x_key, y_key=args.y_key, mask_key=args.mask_key)
    N = len(x)
    input_dim = x.shape[-1]
    dataset = ProbeDataset(x, y, m)
    # Train/val split
    val_size = int(round(N * args.val_frac))
    train_size = N - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
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
            logits = model(x) if mask is None else model(x, attn_mask=mask)
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
            best_metrics = {"epoch": epoch, **val_metrics}
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
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
