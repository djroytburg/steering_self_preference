#!/usr/bin/env python3
"""
Analyze correlation between self-recognition accuracy and (il)legitimate self-preference.

This script reads the self-recognition results and computes statistics showing
whether models that correctly identify their own responses also show legitimate
self-preference (i.e., prefer their own response when it's actually better).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from scipy import stats
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

def expand(path: str) -> str:
    import os
    return os.path.abspath(os.path.expanduser(path))

def visualize_results(results: List[Dict], output_dir: Path, model_name: str):
    """Generate visualizations for self-recognition analysis."""
    print(f"Generating visualizations in {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set dark mode style
    plt.style.use('dark_background')
    sns.set_style("darkgrid", {"axes.facecolor": ".15", "figure.facecolor": ".1"})
    
    # Extract data
    categories = []
    recog_probs = []
    pref_probs = []
    
    for rec in results:
        cat = rec.get('self_pref_category', 'unknown')
        recog = rec.get('normalized', {}).get('correct_recognition', 0.5)
        pref = rec.get('self_pref_probabilities', {}).get('A', 0.5)
        
        categories.append(cat)
        recog_probs.append(recog)
        pref_probs.append(pref)
    
    # 1. Histogram of Self-Recognition Probabilities
    plt.figure(figsize=(10, 6))
    sns.histplot(recog_probs, bins=20, kde=True, color='cyan')
    plt.title(f'Distribution of Self-Recognition Probabilities\n{model_name}', color='white')
    plt.xlabel('Probability of Correct Recognition', color='white')
    plt.ylabel('Count', color='white')
    plt.axvline(x=0.5, color='red', linestyle='--', label='Random Chance')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'self_recognition_hist.png', dpi=300)
    plt.close()
    
    # 2. Boxplot by Category
    plt.figure(figsize=(12, 6))
    data = {'Category': categories, 'Recognition Probability': recog_probs}
    sns.boxplot(x='Category', y='Recognition Probability', data=data, palette='viridis')
    plt.title(f'Self-Recognition by Preference Category\n{model_name}', color='white')
    plt.xlabel('Category', color='white')
    plt.ylabel('Probability of Correct Recognition', color='white')
    plt.axhline(y=0.5, color='red', linestyle='--', label='Random Chance')
    plt.tight_layout()
    plt.savefig(output_dir / 'self_recognition_boxplot.png', dpi=300)
    plt.close()
    
    # 3. Scatter Plot: Recognition vs Preference
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=recog_probs, y=pref_probs, hue=categories, palette='bright', alpha=0.7)
    plt.title(f'Self-Recognition vs Self-Preference Strength\n{model_name}', color='white')
    plt.xlabel('Self-Recognition Probability', color='white')
    plt.ylabel('Self-Preference Probability (Prefers Own)', color='white')
    plt.axhline(y=0.5, color='gray', linestyle='--')
    plt.axvline(x=0.5, color='gray', linestyle='--')
    plt.legend(title='Category')
    plt.tight_layout()
    plt.savefig(output_dir / 'recognition_vs_preference_scatter.png', dpi=300)
    plt.close()
    
    print("Visualizations saved.")

def load_results(results_file: str) -> List[Dict]:
    """Load self-recognition results from JSONL file."""
    results = []
    with open(expand(results_file), 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line))
    return results

def analyze_by_category(results: List[Dict]) -> Dict:
    """
    Analyze self-recognition accuracy by self-preference category.
    
    Returns statistics for each category (LSP, ILSP, no_self_pref, tie).
    """
    categories = defaultdict(lambda: {
        'count': 0,
        'correct_recognition_probs': [],
        'self_pref_probs': [],
        'original_indices': [],
    })
    
    for rec in results:
        category = rec.get('self_pref_category', 'unknown')
        
        # Self-recognition accuracy
        correct_recog = rec.get('normalized', {}).get('correct_recognition', 0.5)
        
        # Self-preference strength (probability of preferring own response)
        self_pref_probs = rec.get('self_pref_probabilities', {})
        # In combined probs, A = judge (own response), B = ref (other response)
        pref_own = self_pref_probs.get('A', 0.5)
        
        categories[category]['count'] += 1
        categories[category]['correct_recognition_probs'].append(correct_recog)
        categories[category]['self_pref_probs'].append(pref_own)
        categories[category]['original_indices'].append(rec.get('original_index'))
    
    # Compute statistics for each category
    stats_by_category = {}
    for category, data in categories.items():
        if data['count'] == 0:
            continue
            
        recog_probs = np.array(data['correct_recognition_probs'])
        pref_probs = np.array(data['self_pref_probs'])
        
        stats_by_category[category] = {
            'count': data['count'],
            'mean_correct_recognition': float(np.mean(recog_probs)),
            'std_correct_recognition': float(np.std(recog_probs)),
            'median_correct_recognition': float(np.median(recog_probs)),
            'mean_self_pref': float(np.mean(pref_probs)),
            'std_self_pref': float(np.std(pref_probs)),
            'median_self_pref': float(np.median(pref_probs)),
        }
        
        # Compute correlation between recognition and preference within category
        if len(recog_probs) > 2:
            try:
                pearson_corr, pearson_p = stats.pearsonr(recog_probs, pref_probs)
                spearman_corr, spearman_p = stats.spearmanr(recog_probs, pref_probs)
                stats_by_category[category]['pearson_correlation'] = float(pearson_corr)
                stats_by_category[category]['pearson_p_value'] = float(pearson_p)
                stats_by_category[category]['spearman_correlation'] = float(spearman_corr)
                stats_by_category[category]['spearman_p_value'] = float(spearman_p)
            except Exception as e:
                print(f"Warning: Could not compute correlation for {category}: {e}")
    
    return stats_by_category

def compute_overall_correlation(results: List[Dict]) -> Dict:
    """
    Compute overall correlation between self-recognition and self-preference.
    """
    recog_probs = []
    pref_probs = []
    categories = []
    
    for rec in results:
        correct_recog = rec.get('normalized', {}).get('correct_recognition', 0.5)
        self_pref_probs = rec.get('self_pref_probabilities', {})
        pref_own = self_pref_probs.get('A', 0.5)
        category = rec.get('self_pref_category', 'unknown')
        
        recog_probs.append(correct_recog)
        pref_probs.append(pref_own)
        categories.append(category)
    
    recog_probs = np.array(recog_probs)
    pref_probs = np.array(pref_probs)
    
    overall_stats = {
        'total_count': len(results),
        'mean_correct_recognition': float(np.mean(recog_probs)),
        'std_correct_recognition': float(np.std(recog_probs)),
        'mean_self_pref': float(np.mean(pref_probs)),
        'std_self_pref': float(np.std(pref_probs)),
    }
    
    if len(recog_probs) > 2:
        try:
            pearson_corr, pearson_p = stats.pearsonr(recog_probs, pref_probs)
            spearman_corr, spearman_p = stats.spearmanr(recog_probs, pref_probs)
            overall_stats['pearson_correlation'] = float(pearson_corr)
            overall_stats['pearson_p_value'] = float(pearson_p)
            overall_stats['spearman_correlation'] = float(spearman_corr)
            overall_stats['spearman_p_value'] = float(spearman_p)
        except Exception as e:
            print(f"Warning: Could not compute overall correlation: {e}")
    
    return overall_stats

def compare_lsp_vs_ilsp(results: List[Dict]) -> Dict:
    """
    Compare self-recognition accuracy between LSP and ILSP cases.
    
    Key question: Do models recognize their own responses better when they
    legitimately prefer them (LSP) vs illegitimately prefer them (ILSP)?
    """
    lsp_recog = []
    ilsp_recog = []
    
    for rec in results:
        category = rec.get('self_pref_category')
        correct_recog = rec.get('normalized', {}).get('correct_recognition', 0.5)
        
        if category == 'LSP':
            lsp_recog.append(correct_recog)
        elif category == 'ILSP':
            ilsp_recog.append(correct_recog)
    
    comparison = {
        'LSP_count': len(lsp_recog),
        'ILSP_count': len(ilsp_recog),
    }
    
    if lsp_recog:
        lsp_arr = np.array(lsp_recog)
        comparison['LSP_mean_recognition'] = float(np.mean(lsp_arr))
        comparison['LSP_std_recognition'] = float(np.std(lsp_arr))
        comparison['LSP_median_recognition'] = float(np.median(lsp_arr))
    
    if ilsp_recog:
        ilsp_arr = np.array(ilsp_recog)
        comparison['ILSP_mean_recognition'] = float(np.mean(ilsp_arr))
        comparison['ILSP_std_recognition'] = float(np.std(ilsp_arr))
        comparison['ILSP_median_recognition'] = float(np.median(ilsp_arr))
    
    # Statistical test for difference
    if lsp_recog and ilsp_recog:
        try:
            # Mann-Whitney U test (non-parametric)
            u_stat, u_p = stats.mannwhitneyu(lsp_recog, ilsp_recog, alternative='two-sided')
            comparison['mannwhitney_u_statistic'] = float(u_stat)
            comparison['mannwhitney_p_value'] = float(u_p)
            
            # t-test (parametric)
            t_stat, t_p = stats.ttest_ind(lsp_recog, ilsp_recog)
            comparison['ttest_t_statistic'] = float(t_stat)
            comparison['ttest_p_value'] = float(t_p)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(lsp_recog) + np.var(ilsp_recog)) / 2)
            if pooled_std > 0:
                cohens_d = (np.mean(lsp_recog) - np.mean(ilsp_recog)) / pooled_std
                comparison['cohens_d'] = float(cohens_d)
        except Exception as e:
            print(f"Warning: Could not compute statistical tests: {e}")
    
    return comparison

def print_report(stats_by_category: Dict, overall_stats: Dict, lsp_vs_ilsp: Dict, model_name: str):
    """Print a formatted report of the analysis."""
    print("=" * 80)
    print(f"SELF-RECOGNITION ANALYSIS REPORT")
    print(f"Model: {model_name}")
    print("=" * 80)
    print()
    
    print("OVERALL STATISTICS")
    print("-" * 80)
    print(f"Total examples: {overall_stats['total_count']}")
    print(f"Mean self-recognition accuracy: {overall_stats['mean_correct_recognition']:.4f} ± {overall_stats['std_correct_recognition']:.4f}")
    print(f"Mean self-preference strength: {overall_stats['mean_self_pref']:.4f} ± {overall_stats['std_self_pref']:.4f}")
    if 'pearson_correlation' in overall_stats:
        print(f"\nOverall Correlation (Recognition vs Preference):")
        print(f"  Pearson r = {overall_stats['pearson_correlation']:.4f} (p = {overall_stats['pearson_p_value']:.4e})")
        print(f"  Spearman ρ = {overall_stats['spearman_correlation']:.4f} (p = {overall_stats['spearman_p_value']:.4e})")
    print()
    
    print("STATISTICS BY CATEGORY")
    print("-" * 80)
    for category in ['LSP', 'ILSP', 'no_self_pref', 'tie', 'unknown']:
        if category not in stats_by_category:
            continue
        stats = stats_by_category[category]
        print(f"\n{category}:")
        print(f"  Count: {stats['count']}")
        print(f"  Mean recognition: {stats['mean_correct_recognition']:.4f} ± {stats['std_correct_recognition']:.4f}")
        print(f"  Median recognition: {stats['median_correct_recognition']:.4f}")
        print(f"  Mean self-pref: {stats['mean_self_pref']:.4f} ± {stats['std_self_pref']:.4f}")
        print(f"  Median self-pref: {stats['median_self_pref']:.4f}")
        if 'pearson_correlation' in stats:
            print(f"  Pearson r = {stats['pearson_correlation']:.4f} (p = {stats['pearson_p_value']:.4e})")
            print(f"  Spearman ρ = {stats['spearman_correlation']:.4f} (p = {stats['spearman_p_value']:.4e})")
    print()
    
    print("LSP vs ILSP COMPARISON")
    print("-" * 80)
    print(f"LSP cases: {lsp_vs_ilsp.get('LSP_count', 0)}")
    if 'LSP_mean_recognition' in lsp_vs_ilsp:
        print(f"  Mean recognition: {lsp_vs_ilsp['LSP_mean_recognition']:.4f} ± {lsp_vs_ilsp['LSP_std_recognition']:.4f}")
        print(f"  Median recognition: {lsp_vs_ilsp['LSP_median_recognition']:.4f}")
    
    print(f"\nILSP cases: {lsp_vs_ilsp.get('ILSP_count', 0)}")
    if 'ILSP_mean_recognition' in lsp_vs_ilsp:
        print(f"  Mean recognition: {lsp_vs_ilsp['ILSP_mean_recognition']:.4f} ± {lsp_vs_ilsp['ILSP_std_recognition']:.4f}")
        print(f"  Median recognition: {lsp_vs_ilsp['ILSP_median_recognition']:.4f}")
    
    if 'mannwhitney_p_value' in lsp_vs_ilsp:
        print(f"\nStatistical Tests:")
        print(f"  Mann-Whitney U: U = {lsp_vs_ilsp['mannwhitney_u_statistic']:.2f}, p = {lsp_vs_ilsp['mannwhitney_p_value']:.4e}")
        print(f"  t-test: t = {lsp_vs_ilsp['ttest_t_statistic']:.4f}, p = {lsp_vs_ilsp['ttest_p_value']:.4e}")
        if 'cohens_d' in lsp_vs_ilsp:
            print(f"  Cohen's d = {lsp_vs_ilsp['cohens_d']:.4f}")
            # Interpret effect size
            d = abs(lsp_vs_ilsp['cohens_d'])
            if d < 0.2:
                effect = "negligible"
            elif d < 0.5:
                effect = "small"
            elif d < 0.8:
                effect = "medium"
            else:
                effect = "large"
            print(f"  Effect size: {effect}")
    print()
    
    print("INTERPRETATION")
    print("-" * 80)
    
    # Check if LSP has higher recognition than ILSP
    if 'LSP_mean_recognition' in lsp_vs_ilsp and 'ILSP_mean_recognition' in lsp_vs_ilsp:
        lsp_better = lsp_vs_ilsp['LSP_mean_recognition'] > lsp_vs_ilsp['ILSP_mean_recognition']
        diff = lsp_vs_ilsp['LSP_mean_recognition'] - lsp_vs_ilsp['ILSP_mean_recognition']
        
        if lsp_better:
            print(f"✓ Models show HIGHER self-recognition for LSP cases (+{diff:.4f})")
            print("  → This suggests models are better at recognizing their own responses")
            print("     when those responses are actually better (legitimate self-preference)")
        else:
            print(f"✗ Models show LOWER self-recognition for LSP cases ({diff:.4f})")
            print("  → This is unexpected and may indicate issues with self-awareness")
        
        if 'mannwhitney_p_value' in lsp_vs_ilsp:
            if lsp_vs_ilsp['mannwhitney_p_value'] < 0.05:
                print(f"  → Difference is statistically significant (p < 0.05)")
            else:
                print(f"  → Difference is NOT statistically significant (p = {lsp_vs_ilsp['mannwhitney_p_value']:.4f})")
    
    # Check overall correlation
    if 'pearson_correlation' in overall_stats:
        r = overall_stats['pearson_correlation']
        p = overall_stats['pearson_p_value']
        
        print(f"\nOverall correlation between recognition and preference: r = {r:.4f}")
        if p < 0.05:
            if r > 0:
                print("  → Positive correlation (significant): Models that recognize their own")
                print("     responses better also tend to prefer them more")
            else:
                print("  → Negative correlation (significant): Models that recognize their own")
                print("     responses better tend to prefer them less")
        else:
            print("  → Correlation is NOT statistically significant")
    
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Analyze self-recognition vs self-preference correlation")
    parser.add_argument("--results_file", type=str, required=True,
                       help="Path to self-recognition results JSONL file")
    parser.add_argument("--output_json", type=str, default=None,
                       help="Optional: Save analysis results to JSON file")
    parser.add_argument("--model_name", type=str, default="Unknown",
                       help="Model name for display in report")
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.results_file}...")
    results = load_results(args.results_file)
    print(f"Loaded {len(results)} results")
    
    # Run analyses
    print("Running analyses...")
    stats_by_category = analyze_by_category(results)
    overall_stats = compute_overall_correlation(results)
    lsp_vs_ilsp = compare_lsp_vs_ilsp(results)
    
    # Print report
    print_report(stats_by_category, overall_stats, lsp_vs_ilsp, args.model_name)
    
    # Generate visualizations
    output_dir = Path(expand(args.results_file)).parent / "plots"
    try:
        visualize_results(results, output_dir, args.model_name)
    except Exception as e:
        print(f"Error generating visualizations: {e}")
    
    # Optionally save to JSON
    if args.output_json:
        output_data = {
            'model_name': args.model_name,
            'overall_statistics': overall_stats,
            'statistics_by_category': stats_by_category,
            'lsp_vs_ilsp_comparison': lsp_vs_ilsp,
        }
        
        output_path = Path(expand(args.output_json))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nAnalysis results saved to: {output_path}")

if __name__ == "__main__":
    main()
