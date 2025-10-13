#!/usr/bin/env python3
import argparse, json, re
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    ap = argparse.ArgumentParser(description="Plot linear vs MLP probe accuracy across layers from agg.json")
    ap.add_argument("--agg", required=True, help="Path to aggregate_best_acc.json (agg.json)")
    ap.add_argument("--out", default="acc_by_layer.png", help="Output PNG path")
    ap.add_argument("--title", default="Probe Accuracy by Layer", help="Plot title")
    ap.add_argument("--show", action="store_true", help="Show the plot window as well")
    return ap.parse_args()

def _layer_num(layer_name: str) -> int:
    # expects e.g. "layer_01" or "layer_1"
    m = re.search(r"(\d+)$", layer_name)
    return int(m.group(1)) if m else -1

def load_series(agg_path: str) -> Tuple[List[int], np.ndarray, np.ndarray]:
    with open(agg_path, "r") as f:
        agg = json.load(f)
    rows = agg.get("layers", [])
    # sort by numeric layer index
    rows = sorted(rows, key=lambda r: _layer_num(r.get("layer", "")))
    layers = [_layer_num(r.get("layer", "")) for r in rows]
    lin = np.array([r.get("linear_acc", np.nan) for r in rows], dtype=float)
    mlp = np.array([r.get("mlp_acc",    np.nan) for r in rows], dtype=float)
    return layers, lin, mlp

def main():
    args = parse_args()
    layers, lin, mlp = load_series(args.agg)

    # Basic sanity: drop entries with missing layer numbers
    mask = np.array([i >= 0 for i in layers], dtype=bool)
    layers = np.array(layers)[mask]
    lin = lin[mask]
    mlp = mlp[mask]

    # Plot
    plt.figure(figsize=(7,4.5))
    plt.plot(layers, lin, marker="o", linewidth=2, label="Linear probe")
    plt.plot(layers, mlp, marker="s", linewidth=2, label="MLP probe")
    plt.xlabel("Layer (depth)")
    plt.ylabel("Best validation accuracy")
    plt.title(args.title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=220)
    print(f"[OK] Saved plot to {args.out}")

    # Also print quick summary
    def best_idx(arr):
        idx = int(np.nanargmax(arr)) if np.any(~np.isnan(arr)) else None
        return idx
    bi_lin = best_idx(lin)
    bi_mlp = best_idx(mlp)
    if bi_lin is not None:
        print(f"Best linear: layer {layers[bi_lin]}  acc={lin[bi_lin]:.4f}")
    if bi_mlp is not None:
        print(f"Best MLP:    layer {layers[bi_mlp]}  acc={mlp[bi_mlp]:.4f}")

    if args.show:
        plt.show()

if __name__ == "__main__":
    main()