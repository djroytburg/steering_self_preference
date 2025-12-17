#!/usr/bin/env python3
"""
Evaluate CAA steering vectors on arena holdout data.

This script:
1. Loads steering vectors created from arena examples
2. Loads holdout test data (not used in vector creation)
3. Applies steering at multiple layers and multipliers
4. Generates FULL SEQUENCES (not just scoring one token)
5. Extracts probabilities for A, B, and Tie
6. Creates distribution plots comparing steered vs unsteered
7. Provides comprehensive logging and monitoring
"""

import argparse
import json
import logging
import os
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Import enhanced hooking utilities
import sys
sys.path.append(str(Path(__file__).parent.parent))
# Use fixed version that handles multi-GPU properly
from enhanced_hooking_fixed import add_activations_and_generate, clear_hooks

# Configure plotting
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up comprehensive logging with both console and file output."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"eval_{timestamp}.log"

    logger = logging.getLogger("eval_arena_vector")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        logger.handlers.clear()

    # File handler with detailed info
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

    logger.info("=" * 80)
    logger.info("EVALUATING ARENA STEERING VECTOR")
    logger.info("=" * 80)
    logger.info(f"Log file: {log_file}")

    return logger

def load_jsonl(path: Path, logger: logging.Logger) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    logger.info(f"Loading: {path}")
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    logger.info(f"  Loaded {len(data)} examples")
    return data

def chat_template_arena(tokenizer, prompt: str, post_script: str = "<|start_header_id|>assistant<|end_header_id|>\n\n") -> str:
    """Apply chat template for arena prompts."""
    system_prompt = "You are a helpful assistant. Please respond with only 'A' or 'B'."

    formatted = tokenizer.apply_chat_template([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ], tokenize=False) + post_script

    return formatted

def get_layer_device(model, layer_idx: int) -> torch.device:
    """
    Get the device for a specific layer in a model that may be split across devices.
    
    Works with HuggingFace's device_map="auto" which can split models across GPUs.
    Supports multiple architectures including Gemma, Llama, Qwen, etc.
    """
    # Try to find the layer in the model structure
    # Try Gemma-specific path first (most nested)
    if hasattr(model, 'model'):
        if hasattr(model.model, 'language_model'):
            if hasattr(model.model.language_model, 'model'):
                if hasattr(model.model.language_model.model, 'layers'):
                    target_layer = model.model.language_model.model.layers[layer_idx]
                    return next(target_layer.parameters()).device
        # Try standard Llama/Qwen path
        if hasattr(model.model, 'layers'):
            target_layer = model.model.layers[layer_idx]
            return next(target_layer.parameters()).device
    
    # Try GPT-2 style
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        target_layer = model.transformer.h[layer_idx]
        return next(target_layer.parameters()).device
    
    # Try direct access
    if hasattr(model, 'layers'):
        target_layer = model.layers[layer_idx]
        return next(target_layer.parameters()).device
    
    # Fallback: try to get first parameter's device
    return next(model.parameters()).device



def generate_with_steering(
    prompt: str,
    model,
    tokenizer,
    layer_idx: int,
    steering_vec: torch.Tensor,
    scale: float,
    max_new_tokens: int = 20,
    logger: logging.Logger = None
) -> Dict[str, Any]:
    """
    Generate text with steering applied.

    Returns:
        Dictionary containing:
        - generated_text: The full generated sequence
        - generated_tokens: List of generated token IDs
        - probabilities: Dict with P(A), P(B), P(T) from first token
    """
    # Format prompt
    formatted_prompt = chat_template_arena(tokenizer, prompt)

    # Get the device for the specific layer we're steering
    # This is critical when model is split across multiple GPUs with device_map="auto"
    layer_device = get_layer_device(model, layer_idx)
    
    # Get model dtype from a parameter
    model_dtype = next(model.parameters()).dtype
    
    # Get the device for the first layer (embedding layer) for input tokens
    # This might be different from layer_device if model is split across GPUs
    first_layer_device = get_layer_device(model, 0)
    
    # Tokenize - put on first layer's device since that's where embedding happens
    tokens = tokenizer(formatted_prompt, return_tensors="pt").to(first_layer_device)

    # Scale and prepare steering vector - keep original dtype/device
    # The hook will move it to the correct device when applying
    scaled_vec = (steering_vec * scale).to(model_dtype)

    # Clear any existing hooks
    clear_hooks(model)

    # Set up steering at last token position (-1)
    specificpos_layer_activations = {layer_idx: {-1: scaled_vec}}
    continuouspos_layer_activations = {}

    # Generation kwargs
    sampling_kwargs = {
        "use_cache": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": max_new_tokens,
        "return_dict_in_generate": True,
        "temperature": None,
        "top_p": None,
        "output_scores": True,
        "do_sample": False
    }

    # Generate with steering
    ids_pos, scores = add_activations_and_generate(
        model,
        tokens,
        specificpos_layer_activations=specificpos_layer_activations,
        continuouspos_layer_activations=continuouspos_layer_activations,
        sampling_kwargs=sampling_kwargs,
        add_at="end",
        score_on_token=None
    )

    # Decode generated sequence
    generated_tokens = ids_pos[0].tolist()
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Extract probabilities for A, B, and Tie from first generated token
    # scores is a tuple of tensors, one per generated token
    # Each tensor has shape (batch_size, vocab_size), we want the first generated token
    first_token_logits = scores[0][0]  # Shape: (vocab_size,)
    first_token_probs = torch.nn.functional.softmax(first_token_logits, dim=-1)

    # Get token IDs for A, B, and T/Tie with error handling
    try:
        token_A_list = tokenizer.encode("A", add_special_tokens=False)
        token_A = token_A_list[0] if token_A_list else tokenizer.encode(" A", add_special_tokens=False)[0]
    except (IndexError, KeyError):
        if logger:
            logger.warning("Could not encode 'A', using fallback")
        token_A = tokenizer.encode(" A", add_special_tokens=False)[0]

    try:
        token_B_list = tokenizer.encode("B", add_special_tokens=False)
        token_B = token_B_list[0] if token_B_list else tokenizer.encode(" B", add_special_tokens=False)[0]
    except (IndexError, KeyError):
        if logger:
            logger.warning("Could not encode 'B', using fallback")
        token_B = tokenizer.encode(" B", add_special_tokens=False)[0]

    # Try to find "Tie" token
    tie_tokens = tokenizer.encode("Tie", add_special_tokens=False)
    if len(tie_tokens) > 0:
        token_T = tie_tokens[0]
    else:
        t_tokens = tokenizer.encode("T", add_special_tokens=False)
        token_T = t_tokens[0] if t_tokens else tokenizer.encode(" T", add_special_tokens=False)[0]

    p_A = first_token_probs[token_A].item()
    p_B = first_token_probs[token_B].item()
    p_T = first_token_probs[token_T].item()

    # Normalize
    total = p_A + p_B + p_T
    if total > 0:
        p_A_norm = p_A / total
        p_B_norm = p_B / total
        p_T_norm = p_T / total
    else:
        p_A_norm = p_B_norm = p_T_norm = 0.0

    return {
        'generated_text': generated_text,
        'generated_tokens': generated_tokens,
        'probabilities': {
            'A': p_A,
            'B': p_B,
            'T': p_T
        },
        'normalized_probabilities': {
            'A': p_A_norm,
            'B': p_B_norm,
            'T': p_T_norm
        }
    }

def calculate_p_self_from_result(result: Dict[str, Any], order: str) -> float:
    """
    Calculate P(self) from generation result.

    Args:
        result: Result dict with normalized_probabilities
        order: 'JR' (judge=A, ref=B) or 'RJ' (ref=A, judge=B)

    Returns:
        P(self) for this order
    """
    probs = result['normalized_probabilities']

    if order == 'JR':
        # Judge is A, so P(self) = P(A)
        return probs['A']
    elif order == 'RJ':
        # Judge is B, so P(self) = P(B)
        return probs['B']
    else:
        raise ValueError(f"Unknown order: {order}")

def plot_distributions(
    results: List[Dict[str, Any]],
    output_dir: Path,
    logger: logging.Logger
):
    """
    Create comprehensive distribution plots comparing steered vs unsteered.
    """
    logger.info("Creating distribution plots...")

    # Organize results by layer and multiplier
    results_by_config = defaultdict(list)

    for result in results:
        layer = result['layer']
        mult = result['multiplier']
        p_self = result['p_self_ilsp']

        key = (layer, mult)
        results_by_config[key].append(p_self)

    # Get baseline (multiplier = 0)
    baseline_configs = [k for k in results_by_config.keys() if k[1] == 0.0]

    if not baseline_configs:
        logger.warning("No baseline (multiplier=0) results found for plotting")
        return

    # Use first layer's baseline
    baseline_layer = baseline_configs[0][0]
    baseline_p_self = results_by_config[(baseline_layer, 0.0)]

    # Plot for each layer
    layers = sorted(set(k[0] for k in results_by_config.keys()))

    for layer in layers:
        layer_results = {k: v for k, v in results_by_config.items() if k[0] == layer}

        # Get multipliers for this layer
        multipliers = sorted(set(k[1] for k in layer_results.keys() if k[1] != 0.0))

        if not multipliers:
            continue

        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        # Plot baseline + steered distributions
        for idx, mult in enumerate(multipliers[:6]):  # Max 6 subplots
            if idx >= 6:
                break

            ax = axes[idx]

            # Baseline
            ax.hist(baseline_p_self, bins=30, alpha=0.5, label='Baseline (mult=0)',
                   color='gray', edgecolor='black', density=True)

            # Steered
            steered_p_self = layer_results[(layer, mult)]
            color = 'blue' if mult > 0 else 'red'
            ax.hist(steered_p_self, bins=30, alpha=0.7, label=f'Steered (mult={mult})',
                   color=color, edgecolor='black', density=True)

            # Statistics
            mean_baseline = np.mean(baseline_p_self)
            mean_steered = np.mean(steered_p_self)
            shift = mean_steered - mean_baseline

            ax.axvline(mean_baseline, color='gray', linestyle='--', linewidth=2)
            ax.axvline(mean_steered, color=color, linestyle='--', linewidth=2)

            ax.set_xlabel('P(self)')
            ax.set_ylabel('Density')
            ax.set_title(f'Layer {layer}, Mult {mult:.2f}\nShift: {shift:+.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'P(self) Distributions for Layer {layer}', fontsize=16)
        plt.tight_layout()

        output_file = output_dir / f"distribution_layer_{layer}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"  Saved: {output_file}")
        plt.close()

        # Also save PDF
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, mult in enumerate(multipliers[:6]):
            if idx >= 6:
                break

            ax = axes[idx]
            ax.hist(baseline_p_self, bins=30, alpha=0.5, label='Baseline (mult=0)',
                   color='gray', edgecolor='black', density=True)

            steered_p_self = layer_results[(layer, mult)]
            color = 'blue' if mult > 0 else 'red'
            ax.hist(steered_p_self, bins=30, alpha=0.7, label=f'Steered (mult={mult})',
                   color=color, edgecolor='black', density=True)

            mean_baseline = np.mean(baseline_p_self)
            mean_steered = np.mean(steered_p_self)
            shift = mean_steered - mean_baseline

            ax.axvline(mean_baseline, color='gray', linestyle='--', linewidth=2)
            ax.axvline(mean_steered, color=color, linestyle='--', linewidth=2)

            ax.set_xlabel('P(self)')
            ax.set_ylabel('Density')
            ax.set_title(f'Layer {layer}, Mult {mult:.2f}\nShift: {shift:+.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'P(self) Distributions for Layer {layer}', fontsize=16)
        plt.tight_layout()

        output_file_pdf = output_dir / f"distribution_layer_{layer}.pdf"
        plt.savefig(output_file_pdf, dpi=300, bbox_inches='tight')
        plt.close()

    # Create summary plot: shift vs multiplier for each layer
    fig, ax = plt.subplots(figsize=(12, 6))

    for layer in layers:
        layer_results = {k: v for k, v in results_by_config.items() if k[0] == layer}
        multipliers = sorted(k[1] for k in layer_results.keys())

        mean_baseline = np.mean(baseline_p_self)
        shifts = []

        for mult in multipliers:
            steered = layer_results[(layer, mult)]
            mean_steered = np.mean(steered)
            shift = mean_steered - mean_baseline
            shifts.append(shift)

        ax.plot(multipliers, shifts, marker='o', label=f'Layer {layer}', linewidth=2)

    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Steering Multiplier')
    ax.set_ylabel('Mean P(self) Shift from Baseline')
    ax.set_title('Steering Effect: P(self) Shift vs Multiplier')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    summary_file = output_dir / "steering_effect_summary.png"
    plt.savefig(summary_file, dpi=300, bbox_inches='tight')
    logger.info(f"  Saved summary: {summary_file}")
    plt.close()

    # PDF version
    fig, ax = plt.subplots(figsize=(12, 6))
    for layer in layers:
        layer_results = {k: v for k, v in results_by_config.items() if k[0] == layer}
        multipliers = sorted(k[1] for k in layer_results.keys())
        mean_baseline = np.mean(baseline_p_self)
        shifts = []
        for mult in multipliers:
            steered = layer_results[(layer, mult)]
            mean_steered = np.mean(steered)
            shift = mean_steered - mean_baseline
            shifts.append(shift)
        ax.plot(multipliers, shifts, marker='o', label=f'Layer {layer}', linewidth=2)
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Steering Multiplier')
    ax.set_ylabel('Mean P(self) Shift from Baseline')
    ax.set_title('Steering Effect: P(self) Shift vs Multiplier')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    summary_file_pdf = output_dir / "steering_effect_summary.pdf"
    plt.savefig(summary_file_pdf, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate arena steering vectors")
    parser.add_argument('--vector_path', type=str, required=True,
                        help='Path to steering vector .pkl file')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to test data output.jsonl')
    parser.add_argument('--steering_examples_dir', type=str, required=True,
                        help='Directory containing steering examples (to exclude from test)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for evaluation results')
    parser.add_argument('--model_id', type=str, required=True,
                        help='HuggingFace model ID')
    parser.add_argument('--layers', type=int, nargs='+', default=[14, 15, 16],
                        help='Layers to apply steering')
    parser.add_argument('--multipliers', type=float, nargs='+',
                        default=[-1.0, -0.5, -0.3, 0.0, 0.3, 0.5, 1.0],
                        help='Steering multipliers to test')
    parser.add_argument('--offset', type=int, default=10,
                        help='Offset position in steering vector')
    parser.add_argument('--n_test', type=int, default=100,
                        help='Number of test examples to evaluate')
    parser.add_argument('--quantize', action='store_true',
                        help='Use 8-bit quantization')
    parser.add_argument('--max_new_tokens', type=int, default=20,
                        help='Maximum new tokens to generate')

    args = parser.parse_args()

    # Set up paths
    vector_path = Path(args.vector_path)
    test_data_path = Path(args.test_data)
    steering_examples_dir = Path(args.steering_examples_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    logger = setup_logging(output_dir)
    logger.info(f"Vector path: {vector_path}")
    logger.info(f"Test data: {test_data_path}")
    logger.info(f"Steering examples dir: {steering_examples_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model: {args.model_id}")
    logger.info(f"Layers: {args.layers}")
    logger.info(f"Multipliers: {args.multipliers}")
    logger.info(f"N test examples: {args.n_test}")

    # Load steering vector
    logger.info(f"Loading steering vector from: {vector_path}")
    with open(vector_path, 'rb') as f:
        steering_vectors = pickle.load(f)

    logger.info(f"  Loaded vectors for {len(steering_vectors)} layers")

    # Load test data
    test_data = load_jsonl(test_data_path, logger)

    # Load steering examples to exclude
    agreement_examples = load_jsonl(steering_examples_dir / "agreement_examples.jsonl", logger)
    bias_examples = load_jsonl(steering_examples_dir / "bias_examples.jsonl", logger)

    steering_row_indices = set(
        [ex['row_index'] for ex in agreement_examples] +
        [ex['row_index'] for ex in bias_examples]
    )

    logger.info(f"Excluding {len(steering_row_indices)} examples used in steering vector creation")

    # Filter test data
    test_data_filtered = [
        item for item in test_data
        if item.get('row_index') not in steering_row_indices
    ]

    logger.info(f"Filtered test data: {len(test_data_filtered)} examples")

    # Sample N test examples
    if len(test_data_filtered) > args.n_test:
        import random
        random.seed(42)
        test_data_filtered = random.sample(test_data_filtered, args.n_test)
        logger.info(f"Sampled {args.n_test} test examples")

    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    HF_TOKEN = os.getenv("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=HF_TOKEN)

    if args.quantize:
        logger.info("Using 8-bit quantization")
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            device_map="auto",
            quantization_config=quant_cfg,
            token=HF_TOKEN
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            device_map="auto",
            token=HF_TOKEN,
            torch_dtype=torch.float16
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.resize_token_embeddings(len(tokenizer))

    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    logger.info("Model loaded successfully")

    # Evaluation loop
    results = []
    results_file = output_dir / "evaluation_results.jsonl"

    total_configs = len(test_data_filtered) * len(args.layers) * len(args.multipliers) * 2  # 2 orders
    logger.info(f"Total configurations to evaluate: {total_configs}")

    offset_logged = False  # Flag to log offset adjustment only once

    with tqdm(total=total_configs, desc="Evaluating", file=sys.stdout) as pbar:
        for test_item in test_data_filtered:
            row_index = test_item['row_index']
            question = test_item['question']
            judge_completion = test_item['judge_completion']
            ref_completion = test_item['ref_completion']

            # Create prompts for both orders
            jr_prompt = f"""Here are two responses to the same question. Please choose which response is better.

Question: {question}

Response A:
{judge_completion}

Response B:
{ref_completion}

Which response is better? Please answer with only "A" or "B"."""

            rj_prompt = f"""Here are two responses to the same question. Please choose which response is better.

Question: {question}

Response A:
{ref_completion}

Response B:
{judge_completion}

Which response is better? Please answer with only "A" or "B"."""

            for layer in args.layers:
                if layer not in steering_vectors:
                    logger.warning(f"Layer {layer} not in steering vectors, skipping")
                    continue

                # Get steering vector for this layer at the specified offset
                # Handle offset indexing: if offset=10 was used in creation, vectors are at indices 0-9
                available_offsets = len(steering_vectors[layer])
                if args.offset >= available_offsets:
                    actual_offset = available_offsets - 1
                    if not offset_logged:  # Only log once
                        logger.info(f"Requested offset {args.offset} >= available offsets {available_offsets}, using offset {actual_offset}")
                        offset_logged = True
                else:
                    actual_offset = args.offset

                base_vec = steering_vectors[layer][actual_offset]

                for mult in args.multipliers:
                    # Evaluate JR order
                    result_jr = generate_with_steering(
                        jr_prompt,
                        model,
                        tokenizer,
                        layer,
                        base_vec,
                        mult,
                        args.max_new_tokens,
                        logger
                    )

                    p_self_jr = calculate_p_self_from_result(result_jr, 'JR')

                    # Evaluate RJ order
                    result_rj = generate_with_steering(
                        rj_prompt,
                        model,
                        tokenizer,
                        layer,
                        base_vec,
                        mult,
                        args.max_new_tokens,
                        logger
                    )

                    p_self_rj = calculate_p_self_from_result(result_rj, 'RJ')

                    # Calculate ILSP (average across orders)
                    p_self_ilsp = (p_self_jr + p_self_rj) / 2.0

                    # Save result
                    result = {
                        'row_index': row_index,
                        'layer': layer,
                        'multiplier': mult,
                        'offset': actual_offset,
                        'p_self_jr': p_self_jr,
                        'p_self_rj': p_self_rj,
                        'p_self_ilsp': p_self_ilsp,
                        'jr_order': {
                            'generated_text': result_jr['generated_text'],
                            'probabilities': result_jr['normalized_probabilities']
                        },
                        'rj_order': {
                            'generated_text': result_rj['generated_text'],
                            'probabilities': result_rj['normalized_probabilities']
                        }
                    }

                    results.append(result)

                    # Save incrementally
                    with open(results_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')

                    pbar.update(2)  # JR and RJ

    logger.info(f"Saved {len(results)} results to: {results_file}")

    # Create distribution plots
    plot_distributions(results, output_dir, logger)

    # Save summary statistics
    summary_stats = defaultdict(lambda: defaultdict(list))

    for result in results:
        layer = result['layer']
        mult = result['multiplier']
        p_self = result['p_self_ilsp']

        summary_stats[layer][mult].append(p_self)

    summary_file = output_dir / "summary_statistics.json"
    summary_data = {}

    for layer, mult_dict in summary_stats.items():
        summary_data[f"layer_{layer}"] = {}
        for mult, p_self_list in mult_dict.items():
            summary_data[f"layer_{layer}"][f"mult_{mult}"] = {
                'mean': float(np.mean(p_self_list)),
                'median': float(np.median(p_self_list)),
                'std': float(np.std(p_self_list)),
                'min': float(np.min(p_self_list)),
                'max': float(np.max(p_self_list)),
                'n': len(p_self_list)
            }

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2)

    logger.info(f"Saved summary statistics to: {summary_file}")
    logger.info("=" * 80)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
