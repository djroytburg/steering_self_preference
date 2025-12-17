#!/usr/bin/env python3
"""
Quick analysis and monitoring tool for steering results.

This script provides real-time summary statistics and progress monitoring
for ongoing or completed steering evaluations.

Usage:
    # Quick summary of results
    python analyze_steering_results.py --results_dir steering_analysis/Qwen_QwQ-32B-Preview/evaluation

    # Watch progress in real-time
    python analyze_steering_results.py --results_dir ... --watch

    # Compare multiple models
    python analyze_steering_results.py --compare steering_analysis/*/evaluation
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

import numpy as np


def load_results(results_file: Path) -> List[Dict[str, Any]]:
    """Load results from JSONL file."""
    results = []
    if results_file.exists():
        with open(results_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    results.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    pass
    return results


def analyze_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze steering results and compute summary statistics."""
    if not results:
        return {
            'n_results': 0,
            'status': 'No results yet'
        }

    # Organize by configuration
    by_config = defaultdict(list)

    for result in results:
        layer = result['layer']
        mult = result['multiplier']
        p_self = result['p_self_ilsp']

        key = (layer, mult)
        by_config[key].append(p_self)

    # Find baseline
    baseline_configs = [k for k in by_config.keys() if k[1] == 0.0]

    if not baseline_configs:
        baseline_mean = None
    else:
        baseline_layer = baseline_configs[0][0]
        baseline_p_self = by_config[(baseline_layer, 0.0)]
        baseline_mean = np.mean(baseline_p_self)

    # Compute statistics for each config
    config_stats = {}

    for (layer, mult), p_self_list in by_config.items():
        mean_p_self = np.mean(p_self_list)

        if baseline_mean is not None:
            shift = mean_p_self - baseline_mean
        else:
            shift = None

        config_stats[(layer, mult)] = {
            'mean': mean_p_self,
            'median': np.median(p_self_list),
            'std': np.std(p_self_list),
            'n': len(p_self_list),
            'shift_from_baseline': shift
        }

    # Find best steering configurations
    if baseline_mean is not None:
        # Most reduction in self-preference
        steered_configs = [(k, v) for k, v in config_stats.items() if k[1] != 0.0]

        if steered_configs:
            best_reduction = min(steered_configs, key=lambda x: x[1]['mean'])
            best_increase = max(steered_configs, key=lambda x: x[1]['mean'])
        else:
            best_reduction = None
            best_increase = None
    else:
        best_reduction = None
        best_increase = None

    return {
        'n_results': len(results),
        'n_configs': len(by_config),
        'baseline_mean': baseline_mean,
        'config_stats': config_stats,
        'best_reduction': best_reduction,
        'best_increase': best_increase,
        'layers': sorted(set(k[0] for k in by_config.keys())),
        'multipliers': sorted(set(k[1] for k in by_config.keys()))
    }


def print_summary(analysis: Dict[str, Any], model_name: str = None):
    """Print formatted summary of analysis."""
    print("=" * 80)
    if model_name:
        print(f"STEERING RESULTS SUMMARY: {model_name}")
    else:
        print("STEERING RESULTS SUMMARY")
    print("=" * 80)
    print()

    print(f"Total results: {analysis['n_results']}")
    print(f"Configurations: {analysis['n_configs']}")

    if analysis['baseline_mean'] is not None:
        print(f"Baseline P(self): {analysis['baseline_mean']:.4f}")
        print()

        print("Results by configuration:")
        print("-" * 80)
        print(f"{'Layer':<8} {'Mult':<8} {'Mean P(self)':<15} {'Shift':<10} {'N':<6}")
        print("-" * 80)

        # Sort by layer, then multiplier
        sorted_configs = sorted(analysis['config_stats'].items(),
                               key=lambda x: (x[0][0], x[0][1]))

        for (layer, mult), stats in sorted_configs:
            shift_str = f"{stats['shift_from_baseline']:+.4f}" if stats['shift_from_baseline'] is not None else "N/A"
            print(f"{layer:<8} {mult:<8.2f} {stats['mean']:<15.4f} {shift_str:<10} {stats['n']:<6}")

        print("-" * 80)
        print()

        if analysis['best_reduction']:
            (layer, mult), stats = analysis['best_reduction']
            print(f"Best reduction in self-preference:")
            print(f"  Layer {layer}, Mult {mult:.2f}")
            print(f"  Mean P(self): {stats['mean']:.4f} (shift: {stats['shift_from_baseline']:+.4f})")
            print()

        if analysis['best_increase']:
            (layer, mult), stats = analysis['best_increase']
            print(f"Best increase in self-preference:")
            print(f"  Layer {layer}, Mult {mult:.2f}")
            print(f"  Mean P(self): {stats['mean']:.4f} (shift: {stats['shift_from_baseline']:+.4f})")
            print()

    print("=" * 80)


def watch_progress(results_dir: Path, interval: int = 30):
    """Watch results directory and print updates."""
    results_file = results_dir / "evaluation_results.jsonl"

    print(f"Watching: {results_file}")
    print(f"Update interval: {interval} seconds")
    print("Press Ctrl+C to stop")
    print()

    last_n = 0

    try:
        while True:
            results = load_results(results_file)
            n = len(results)

            if n != last_n:
                print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] New results: {n - last_n} (total: {n})")

                if n > 0:
                    analysis = analyze_results(results)
                    print_summary(analysis)

                last_n = n

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nStopped watching.")


def compare_models(results_dirs: List[Path]):
    """Compare steering results across multiple models."""
    print("=" * 80)
    print("COMPARING STEERING RESULTS ACROSS MODELS")
    print("=" * 80)
    print()

    model_analyses = {}

    for results_dir in results_dirs:
        model_name = results_dir.parent.name
        results_file = results_dir / "evaluation_results.jsonl"

        results = load_results(results_file)
        analysis = analyze_results(results)

        model_analyses[model_name] = analysis

    # Print comparison table
    print(f"{'Model':<30} {'N Results':<12} {'Baseline':<12} {'Best Reduction':<20}")
    print("-" * 80)

    for model_name, analysis in model_analyses.items():
        n_results = analysis['n_results']
        baseline = f"{analysis['baseline_mean']:.4f}" if analysis['baseline_mean'] is not None else "N/A"

        if analysis['best_reduction']:
            (layer, mult), stats = analysis['best_reduction']
            best_red = f"L{layer} M{mult:.1f} ({stats['shift_from_baseline']:+.4f})"
        else:
            best_red = "N/A"

        print(f"{model_name:<30} {n_results:<12} {baseline:<12} {best_red:<20}")

    print("-" * 80)
    print()

    # Detailed breakdown for each model
    for model_name, analysis in model_analyses.items():
        print_summary(analysis, model_name)
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and monitor steering results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--results_dir', type=str,
                        help='Results directory (e.g., steering_analysis/model/evaluation)')
    parser.add_argument('--watch', action='store_true',
                        help='Watch for new results in real-time')
    parser.add_argument('--interval', type=int, default=30,
                        help='Update interval in seconds for watch mode (default: 30)')
    parser.add_argument('--compare', type=str, nargs='+',
                        help='Compare multiple result directories')

    args = parser.parse_args()

    if args.compare:
        # Compare mode
        results_dirs = [Path(d) for d in args.compare]
        compare_models(results_dirs)

    elif args.results_dir:
        results_dir = Path(args.results_dir)

        if not results_dir.exists():
            print(f"Error: Results directory not found: {results_dir}")
            return 1

        if args.watch:
            # Watch mode
            watch_progress(results_dir, args.interval)
        else:
            # One-time analysis
            results_file = results_dir / "evaluation_results.jsonl"
            results = load_results(results_file)
            analysis = analyze_results(results)
            print_summary(analysis)

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
