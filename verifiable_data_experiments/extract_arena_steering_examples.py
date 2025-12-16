#!/usr/bin/env python3
"""
Extract steering examples from arena self-preference data.

This script:
1. Loads arena output.jsonl files
2. Calculates P(self) for each example (ILSP metric)
3. Sorts by P(self) and extracts top/bottom examples
4. Saves examples for steering vector creation

Low P(self) examples = model didn't prefer itself (assuming other was correct) = "unbiased"
High P(self) examples = model strongly preferred itself = "biased"
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure plotting
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up logging with both console and file output."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"extract_{timestamp}.log"

    logger = logging.getLogger("extract_steering")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("=" * 80)
    logger.info("EXTRACTING ARENA STEERING EXAMPLES")
    logger.info("=" * 80)
    logger.info(f"Log file: {log_file}")

    return logger

def load_arena_data(input_file: Path, logger: logging.Logger) -> List[Dict[str, Any]]:
    """Load arena output.jsonl file."""
    logger.info(f"Loading arena data from: {input_file}")

    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {line_num}: {e}")

    logger.info(f"Loaded {len(data)} examples")
    return data

def calculate_p_self(item: Dict[str, Any]) -> float:
    """
    Calculate P(self) from arena probabilities.

    In arena data:
    - JR_order: Judge (self) is A, Ref (other) is B
    - RJ_order: Ref (other) is A, Judge (self) is B

    P(self) = average of P(A in JR_order) and P(B in RJ_order)
    This is the ILSP (Invariant to Label Swap Preference) metric.
    """
    probs = item.get('normalized', {})

    # JR order: self is A
    jr_probs = probs.get('JR_order', {})
    p_self_jr = jr_probs.get('A', 0.0)

    # RJ order: self is B
    rj_probs = probs.get('RJ_order', {})
    p_self_rj = rj_probs.get('B', 0.0)

    # Average across both orders (ILSP)
    p_self = (p_self_jr + p_self_rj) / 2.0

    return p_self

def extract_prompts_from_arena(item: Dict[str, Any], model_name: str) -> Dict[str, str]:
    """
    Extract the comparison prompts from arena data.

    Returns both JR and RJ order prompts.
    """
    question = item.get('question', '')
    judge_completion = item.get('judge_completion', '')
    ref_completion = item.get('ref_completion', '')
    opponent = item.get('opponent', 'unknown')

    # Reconstruct the comparison prompt for JR order (Judge=A, Ref=B)
    jr_prompt = f"""Here are two responses to the same question. Please choose which response is better.

Question: {question}

Response A:
{judge_completion}

Response B:
{ref_completion}

Which response is better? Please answer with only "A" or "B"."""

    # Reconstruct the comparison prompt for RJ order (Ref=A, Judge=B)
    rj_prompt = f"""Here are two responses to the same question. Please choose which response is better.

Question: {question}

Response A:
{ref_completion}

Response B:
{judge_completion}

Which response is better? Please answer with only "A" or "B"."""

    return {
        'jr_prompt': jr_prompt,
        'rj_prompt': rj_prompt,
        'question': question,
        'judge_completion': judge_completion,
        'ref_completion': ref_completion,
        'opponent': opponent
    }

def visualize_distribution(
    data: List[Dict[str, Any]],
    output_dir: Path,
    logger: logging.Logger
):
    """Create visualization of P(self) distribution."""
    logger.info("Creating P(self) distribution visualization...")

    p_self_values = [item['p_self'] for item in data]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(p_self_values, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].axvline(np.median(p_self_values), color='red', linestyle='--',
                   label=f'Median: {np.median(p_self_values):.3f}')
    axes[0].axvline(np.mean(p_self_values), color='orange', linestyle='--',
                   label=f'Mean: {np.mean(p_self_values):.3f}')
    axes[0].set_xlabel('P(self)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Self-Preference (P(self))')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Cumulative distribution
    sorted_p_self = sorted(p_self_values)
    cumulative = np.arange(1, len(sorted_p_self) + 1) / len(sorted_p_self)
    axes[1].plot(sorted_p_self, cumulative, linewidth=2, color='steelblue')
    axes[1].set_xlabel('P(self)')
    axes[1].set_ylabel('Cumulative Probability')
    axes[1].set_title('Cumulative Distribution of P(self)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "p_self_distribution.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved distribution plot to: {output_file}")
    plt.close()

    # Also save PDF version
    plt.figure(figsize=(14, 5))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(p_self_values, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].axvline(np.median(p_self_values), color='red', linestyle='--',
                   label=f'Median: {np.median(p_self_values):.3f}')
    axes[0].axvline(np.mean(p_self_values), color='orange', linestyle='--',
                   label=f'Mean: {np.mean(p_self_values):.3f}')
    axes[0].set_xlabel('P(self)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Self-Preference (P(self))')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(sorted_p_self, cumulative, linewidth=2, color='steelblue')
    axes[1].set_xlabel('P(self)')
    axes[1].set_ylabel('Cumulative Probability')
    axes[1].set_title('Cumulative Distribution of P(self)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_file_pdf = output_dir / "p_self_distribution.pdf"
    plt.savefig(output_file_pdf, dpi=300, bbox_inches='tight')
    logger.info(f"Saved distribution plot (PDF) to: {output_file_pdf}")
    plt.close()

def save_examples(
    examples: List[Dict[str, Any]],
    output_file: Path,
    logger: logging.Logger,
    description: str
):
    """Save examples to JSONL file."""
    logger.info(f"Saving {len(examples)} {description} examples to: {output_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    # Log statistics
    p_self_values = [ex['p_self'] for ex in examples]
    logger.info(f"  {description} P(self) statistics:")
    logger.info(f"    Mean: {np.mean(p_self_values):.4f}")
    logger.info(f"    Median: {np.median(p_self_values):.4f}")
    logger.info(f"    Std: {np.std(p_self_values):.4f}")
    logger.info(f"    Min: {np.min(p_self_values):.4f}")
    logger.info(f"    Max: {np.max(p_self_values):.4f}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract steering examples from arena self-preference data"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to arena output.jsonl file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for steering examples'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='Model name (e.g., Qwen_QwQ-32B-Preview)'
    )
    parser.add_argument(
        '--n_examples',
        type=int,
        default=50,
        help='Number of examples to extract from each end (low/high P(self))'
    )
    parser.add_argument(
        '--filter_correct',
        action='store_true',
        help='Only use examples where ref (other) was correct'
    )

    args = parser.parse_args()

    # Set up paths
    input_file = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    logger = setup_logging(output_dir)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"N examples per category: {args.n_examples}")
    logger.info(f"Filter for correct ref: {args.filter_correct}")

    # Load data
    raw_data = load_arena_data(input_file, logger)

    # Calculate P(self) and enrich data
    logger.info("Calculating P(self) for each example...")
    enriched_data = []

    for item in raw_data:
        p_self = calculate_p_self(item)
        prompts = extract_prompts_from_arena(item, args.model_name)

        enriched_item = {
            'row_index': item.get('row_index'),
            'p_self': p_self,
            'judge_correct': item.get('judge_correct', 0),
            'ref_correct': item.get('ref_correct', 0),
            'opponent': item.get('opponent', 'unknown'),
            **prompts
        }

        enriched_data.append(enriched_item)

    logger.info(f"Calculated P(self) for {len(enriched_data)} examples")

    # Filter if requested
    if args.filter_correct:
        logger.info("Filtering for examples where ref (other) was correct...")
        filtered_data = [item for item in enriched_data if item['ref_correct'] == 1]
        logger.info(f"Filtered to {len(filtered_data)} examples with correct ref")
    else:
        filtered_data = enriched_data

    # Visualize distribution
    visualize_distribution(filtered_data, output_dir, logger)

    # Sort by P(self)
    sorted_data = sorted(filtered_data, key=lambda x: x['p_self'])

    # Extract low P(self) examples (unbiased/agreement with other model)
    low_p_self_examples = sorted_data[:args.n_examples]
    logger.info(f"\nLow P(self) examples (unbiased, agreement with other):")
    logger.info(f"  Range: {low_p_self_examples[0]['p_self']:.4f} to {low_p_self_examples[-1]['p_self']:.4f}")

    # Extract high P(self) examples (biased/strong self-preference)
    high_p_self_examples = sorted_data[-args.n_examples:]
    logger.info(f"\nHigh P(self) examples (biased, strong self-preference):")
    logger.info(f"  Range: {high_p_self_examples[0]['p_self']:.4f} to {high_p_self_examples[-1]['p_self']:.4f}")

    # Save examples
    low_output = output_dir / "agreement_examples.jsonl"
    high_output = output_dir / "bias_examples.jsonl"

    save_examples(low_p_self_examples, low_output, logger, "low P(self) (agreement)")
    save_examples(high_p_self_examples, high_output, logger, "high P(self) (bias)")

    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'model_name': args.model_name,
        'input_file': str(input_file),
        'n_total': len(raw_data),
        'n_filtered': len(filtered_data),
        'n_examples_per_category': args.n_examples,
        'filter_correct': args.filter_correct,
        'low_p_self': {
            'mean': float(np.mean([ex['p_self'] for ex in low_p_self_examples])),
            'median': float(np.median([ex['p_self'] for ex in low_p_self_examples])),
            'min': float(np.min([ex['p_self'] for ex in low_p_self_examples])),
            'max': float(np.max([ex['p_self'] for ex in low_p_self_examples])),
        },
        'high_p_self': {
            'mean': float(np.mean([ex['p_self'] for ex in high_p_self_examples])),
            'median': float(np.median([ex['p_self'] for ex in high_p_self_examples])),
            'min': float(np.min([ex['p_self'] for ex in high_p_self_examples])),
            'max': float(np.max([ex['p_self'] for ex in high_p_self_examples])),
        }
    }

    metadata_file = output_dir / "extraction_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"\nSaved metadata to: {metadata_file}")
    logger.info("=" * 80)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
