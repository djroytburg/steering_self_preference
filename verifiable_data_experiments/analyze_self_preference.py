


#!/usr/bin/env python3
"""
Analyze and visualize self-preference probabilities from vllm_self_preference.py output.
Generates histograms for legitimate (LSP), illegitimate (ILSP), and all examples.
"""

from datetime import datetime
import os
import sys
import argparse
import json
import logging
import getpass
import socket
import platform
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bootstrap

# ---------------------------
# Logging Setup
# ---------------------------

def setup_logging(output_dir: Path, args) -> logging.Logger:
    """
    Set up logging with comprehensive metadata about the run.
    Creates both console and file handlers.
    Logs are saved in output_dir/logs/
    """
    # Create logs directory within the output directory
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"analysis_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger("analyze_self_preference")
    logger.setLevel(logging.DEBUG)
    
    # Prevent duplicate handlers if function is called multiple times
    if logger.handlers:
        logger.handlers.clear()
    
    # File handler with detailed formatting
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler with simpler formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Log comprehensive run metadata
    logger.info("=" * 80)
    logger.info("ANALYSIS RUN STARTED")
    logger.info("=" * 80)
    
    # System metadata
    logger.info("SYSTEM METADATA:")
    logger.info(f"  User: {getpass.getuser()}")
    logger.info(f"  Hostname: {socket.gethostname()}")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  Python version: {sys.version.split()[0]}")
    
    # Run parameters
    logger.info("RUN PARAMETERS:")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"  {arg}: {value}")
    
    logger.info(f"  Log file: {log_file}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Timestamp: {timestamp}")
    logger.info("=" * 80)
    
    return logger

# ---------------------------
# Analysis Functions
# ---------------------------

def expand(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))

def load_results(jsonl_path: str, logger: logging.Logger):
    """Load results from output.jsonl file."""
    logger.info(f"Loading results from {jsonl_path}")
    results = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))
    logger.info(f"Loaded {len(results)} examples")
    return results

def extract_probabilities(scores, logger: logging.Logger):
    """
    Extract self-preference probabilities for legitimate (LSP), 
    illegitimate (ILSP), and all examples.
    
    Returns:
        dict with keys 'lsp', 'ilsp', 'all' containing lists of probabilities
        float: proportion of legitimate examples
    """
    examples = {
        "lsp": [],
        "ilsp": [],
        "all": []
    }
    
    skipped = 0
    for s in scores:
        # Skip examples with zero probabilities (failed to extract)
        if s['probabilities']['RJ_order'] == {'A': 0.0, 'B': 0.0, 'T': 0.0}:
            skipped += 1
            continue
        
        # Calculate P(self) = P(B) / (P(A) + P(B))
        # In RJ order: A=ref, B=judge, so P(B) is self-preference
        AB_normalized = s['probabilities']['RJ_order']["B"] / (
            s['probabilities']['RJ_order']["A"] + s['probabilities']['RJ_order']["B"]
        )
        
        examples['all'].append(AB_normalized)
        
        # Categorize as legitimate or illegitimate based on judge correctness
        if s['judge_correct'] == 1:
            examples['lsp'].append(AB_normalized)
        else:
            examples['ilsp'].append(AB_normalized)
    
    lsp_proportion = len(examples['lsp']) / (len(examples['lsp']) + len(examples['ilsp']))
    
    logger.info(f"Extracted probabilities:")
    logger.info(f"  Total valid examples: {len(examples['all'])}")
    logger.info(f"  Legitimate (LSP): {len(examples['lsp'])}")
    logger.info(f"  Illegitimate (ILSP): {len(examples['ilsp'])}")
    logger.info(f"  Skipped (zero probs): {skipped}")
    logger.info(f"  LSP proportion: {lsp_proportion:.3f}")
    
    return examples, lsp_proportion

def plot_histogram_with_stats(ax, data, objective, title=None, xlabel=None, 
                               color='cornflowerblue', no_spine=False, **hist_kwargs):
    """
    Plot histogram with mean, confidence interval, and objective score.
    """
    if len(data) == 0:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        return
    
    # Calculate mean and bootstrap CI
    mean = np.mean(data)
    res = bootstrap((np.array(data),), np.mean, random_state=42)
    CI = res.confidence_interval
    
    # Plotting
    ax.hist(data, color=color, edgecolor='white', linewidth=0.5, **hist_kwargs)
    ax.axvline(mean, color='black', linestyle='--', linewidth=2, label='μ')
    ax.axvline(objective, color='red', linestyle='--', linewidth=2, label='Judge Score')
    
    # Plot confidence interval
    ax.axvspan(CI.low, CI.high, alpha=0.2, color='orange', label='95% CI')
    
    if title:
        ax.set_title(title, fontweight='bold', fontsize=12, pad=18)
        
        # Subtitle with statistics
        subtitle = f"μ: {mean:.3f} | Judge Score: {objective:.3f} | n={len(data)}"
        ax.text(0.5, 1.0, subtitle,
                ha='center',
                va='bottom',
                transform=ax.transAxes,
                fontsize=9,
                color='gray')
    
    if xlabel:
        ax.set_xlabel(xlabel)
    
    if no_spine:
        ax.spines['left'].set_visible(False)
    
    ax.set_ylabel('Count')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def create_visualization(prob_data, lsp_proportion, output_path: Path, logger: logging.Logger):
    """Create and save visualization with three histograms."""
    logger.info("Creating visualization...")
    
    # Set up matplotlib style
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ['Liberation Sans'],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    })
    
    # Create figure with three subplots
    fig, axes = plt.subplots(ncols=3, figsize=(12, 4), sharey=True)
    
    # Plot legitimate examples (LSP)
    plot_histogram_with_stats(
        ax=axes[0],
        data=prob_data['lsp'],
        objective=lsp_proportion,
        title="Legitimate Self-Preference (LSP)",
        xlabel="P(self)",
        color='green',
        bins=20,
        alpha=0.7
    )
    
    # Plot illegitimate examples (ILSP)
    plot_histogram_with_stats(
        ax=axes[1],
        data=prob_data['ilsp'],
        objective=1 - lsp_proportion,
        title="Illegitimate Self-Preference (ILSP)",
        xlabel="P(self)",
        color='red',
        bins=20,
        alpha=0.7,
        no_spine=True
    )
    
    # Plot all examples
    plot_histogram_with_stats(
        ax=axes[2],
        data=prob_data['all'],
        objective=lsp_proportion,
        title="All Examples",
        xlabel="P(self)",
        color='cornflowerblue',
        bins=20,
        alpha=0.7,
        no_spine=True
    )
    
    # Add legend to last plot
    axes[2].legend(loc='upper left')
    
    plt.tight_layout()
    
    # Save figure
    pdf_path = output_path / "self_preference_analysis.pdf"
    plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
    logger.info(f"Saved visualization to {pdf_path}")
    
    # Also save as PNG for easy viewing
    png_path = output_path / "self_preference_analysis.png"
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    logger.info(f"Saved visualization to {png_path}")
    
    plt.close()

def save_statistics(prob_data, lsp_proportion, output_path: Path, logger: logging.Logger):
    """Save detailed statistics to JSON file."""
    logger.info("Computing and saving statistics...")
    
    stats = {
        "timestamp": str(datetime.now()),
        "n_total": len(prob_data['all']),
        "n_lsp": len(prob_data['lsp']),
        "n_ilsp": len(prob_data['ilsp']),
        "lsp_proportion": float(lsp_proportion),
        "statistics": {}
    }
    
    for category in ['lsp', 'ilsp', 'all']:
        data = prob_data[category]
        if len(data) > 0:
            mean = np.mean(data)
            res = bootstrap((np.array(data),), np.mean, random_state=42)
            CI = res.confidence_interval
            
            stats['statistics'][category] = {
                "mean": float(mean),
                "median": float(np.median(data)),
                "std": float(np.std(data)),
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "ci_low": float(CI.low),
                "ci_high": float(CI.high),
                "n": len(data)
            }
            
            logger.info(f"{category.upper()} statistics:")
            logger.info(f"  Mean: {mean:.3f}")
            logger.info(f"  95% CI: [{CI.low:.3f}, {CI.high:.3f}]")
            logger.info(f"  Median: {np.median(data):.3f}")
            logger.info(f"  Std: {np.std(data):.3f}")
    
    stats_path = output_path / "statistics.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved statistics to {stats_path}")

# ---------------------------
# Main
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Analyze and visualize self-preference probabilities.")
    p.add_argument("--input_dir", type=str, required=True, 
                   help="Directory containing output.jsonl and metadata.json from vllm_self_preference.py")
    return p.parse_args()

def main():
    args = parse_args()
    
    # Set up input/output paths
    input_dir = Path(expand(args.input_dir))
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    input_jsonl = input_dir / "output.jsonl"
    if not input_jsonl.exists():
        print(f"Error: output.jsonl not found in {input_dir}")
        sys.exit(1)
    
    # Archive existing analysis files if they exist
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for filename in ["self_preference_analysis.pdf", "self_preference_analysis.png", "statistics.json"]:
        file_path = input_dir / filename
        if file_path.exists():
            old_dir = input_dir / "old"
            old_dir.mkdir(parents=True, exist_ok=True)
            archived_path = old_dir / f"{file_path.stem}_{timestamp}{file_path.suffix}"
            file_path.rename(archived_path)
    
    # Set up logging
    logger = setup_logging(input_dir, args)
    
    # Load results
    results = load_results(str(input_jsonl), logger)
    
    # Extract probabilities
    prob_data, lsp_proportion = extract_probabilities(results, logger)
    
    # Create visualization
    create_visualization(prob_data, lsp_proportion, input_dir, logger)
    
    # Save statistics
    save_statistics(prob_data, lsp_proportion, input_dir, logger)
    
    logger.info("=" * 80)
    logger.info("ANALYSIS COMPLETED SUCCESSFULLY")
    logger.info(f"Results saved in: {input_dir}")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
