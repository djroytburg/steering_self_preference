#!/usr/bin/env python3
"""
Analyze Individual Self-Recognition Results.

Metrics:
- True Positive Rate (TPR): Accuracy on Own Responses (Target: 1)
- True Negative Rate (TNR): Accuracy on Other Responses (Target: 0)
- Balanced Accuracy: (TPR + TNR) / 2
- Correlation with Self-Preference Category (LSP vs ILSP)
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

def expand(path: str) -> str:
    import os
    return os.path.abspath(os.path.expanduser(path))

def categorize_preference(record: dict) -> str:
    # Re-implement categorization logic if not present in record
    # But run_individual_self_recognition.py passes 'self_pref_data' through
    sp_data = record.get('self_pref_data', {})
    if not sp_data:
        return "unknown"
    
    # Logic from run_self_recognition.py
    combined_probs = sp_data.get("normalized", {}).get("combined", {})
    judge_prob = combined_probs.get("A", 0.0)
    ref_prob = combined_probs.get("B", 0.0)
    judge_correct = sp_data.get("judge_correct")
    ref_correct = sp_data.get("ref_correct")
    
    prefers_own = judge_prob > ref_prob
    if not prefers_own:
        return "no_self_pref"
    
    if judge_correct == 1 and ref_correct == 0:
        return "LSP"
    elif judge_correct == 0 and ref_correct == 1:
        return "ILSP"
    elif judge_correct == ref_correct:
        return "tie"
    else:
        return "unknown"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--model_name", default="Model")
    args = parser.parse_args()

    results_path = Path(expand(args.results_file))
    if args.output_dir:
        out_dir = Path(expand(args.output_dir))
    else:
        out_dir = results_path.parent / "plots"
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Data
    records = []
    with open(results_path, 'r') as f:
        for line in f:
            records.append(json.loads(line))
            
    print(f"Loaded {len(records)} records.")
    
    # Extract Metrics
    tpr_list = [] # Own -> 1
    tnr_list = [] # Other -> 0
    categories = []
    
    for rec in records:
        indiv = rec['individual_results']
        own = indiv['own_response']
        other = indiv['other_response']
        
        # Prob of saying 1 for Own (True Positive)
        # We use prob_1 directly as a soft score
        tpr_list.append(own['prob_1'])
        
        # Prob of saying 0 for Other (True Negative)
        tnr_list.append(other['prob_0'])
        
        categories.append(categorize_preference(rec))
        
    tpr = np.array(tpr_list)
    tnr = np.array(tnr_list)
    balanced_acc = (tpr + tnr) / 2
    
    # Overall Stats
    print("="*60)
    print(f"INDIVIDUAL RECOGNITION REPORT: {args.model_name}")
    print("="*60)
    print(f"Mean TPR (Accuracy on Own):   {np.mean(tpr):.4f} ± {np.std(tpr):.4f}")
    print(f"Mean TNR (Accuracy on Other): {np.mean(tnr):.4f} ± {np.std(tnr):.4f}")
    print(f"Mean Balanced Accuracy:       {np.mean(balanced_acc):.4f} ± {np.std(balanced_acc):.4f}")
    print("-" * 60)
    
    # By Category
    cat_stats = {}
    unique_cats = set(categories)
    
    for cat in unique_cats:
        indices = [i for i, c in enumerate(categories) if c == cat]
        if not indices:
            continue
            
        cat_tpr = tpr[indices]
        cat_tnr = tnr[indices]
        cat_bal = balanced_acc[indices]
        
        cat_stats[cat] = {
            "count": len(indices),
            "tpr": np.mean(cat_tpr),
            "tnr": np.mean(cat_tnr),
            "balanced": np.mean(cat_bal)
        }
        
        print(f"Category: {cat} (n={len(indices)})")
        print(f"  TPR: {np.mean(cat_tpr):.4f}")
        print(f"  TNR: {np.mean(cat_tnr):.4f}")
        print(f"  Bal: {np.mean(cat_bal):.4f}")
        
    # LSP vs ILSP
    if "LSP" in cat_stats and "ILSP" in cat_stats:
        lsp_idxs = [i for i, c in enumerate(categories) if c == "LSP"]
        ilsp_idxs = [i for i, c in enumerate(categories) if c == "ILSP"]
        
        lsp_bal = balanced_acc[lsp_idxs]
        ilsp_bal = balanced_acc[ilsp_idxs]
        
        u_stat, p_val = stats.mannwhitneyu(lsp_bal, ilsp_bal)
        print("-" * 60)
        print(f"LSP vs ILSP (Balanced Accuracy):")
        print(f"  LSP Mean:  {np.mean(lsp_bal):.4f}")
        print(f"  ILSP Mean: {np.mean(ilsp_bal):.4f}")
        print(f"  Diff:      {np.mean(lsp_bal) - np.mean(ilsp_bal):.4f}")
        print(f"  P-value:   {p_val:.4e}")
        
    # Visualizations (Dark Mode)
    plt.style.use('dark_background')
    sns.set_style("darkgrid", {"axes.facecolor": ".15", "figure.facecolor": ".1"})
    
    # 1. TPR vs TNR Scatter
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=tnr, y=tpr, hue=categories, palette='bright', alpha=0.6)
    plt.title(f"Individual Recognition: TPR vs TNR\n{args.model_name}", color='white')
    plt.xlabel("True Negative Rate (Prob 0 on Other)", color='white')
    plt.ylabel("True Positive Rate (Prob 1 on Own)", color='white')
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.3) # Diagonal
    plt.axhline(0.5, color='gray', linestyle=':')
    plt.axvline(0.5, color='gray', linestyle=':')
    plt.tight_layout()
    plt.savefig(out_dir / "indiv_tpr_vs_tnr.png", dpi=300)
    plt.close()
    
    # 2. Balanced Accuracy Boxplot
    plt.figure(figsize=(10, 6))
    data = {"Category": categories, "Balanced Accuracy": balanced_acc}
    sns.boxplot(x="Category", y="Balanced Accuracy", data=data, palette="viridis")
    plt.title(f"Balanced Accuracy by Category\n{args.model_name}", color='white')
    plt.axhline(0.5, color='red', linestyle='--', label="Random")
    plt.tight_layout()
    plt.savefig(out_dir / "indiv_balanced_acc_boxplot.png", dpi=300)
    plt.close()
    
    print(f"\nPlots saved to {out_dir}")

if __name__ == "__main__":
    main()
