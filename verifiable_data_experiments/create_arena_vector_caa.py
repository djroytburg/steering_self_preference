#!/usr/bin/env python3
"""
Create CAA (Contrastive Activation Addition) steering vectors from arena examples.

Based on create_vector_caa.py but adapted for arena self-preference data.

Key steps:
1. Load low P(self) (agreement) and high P(self) (bias) examples
2. Extract activations at multiple layers and token positions
3. Compute mean difference vectors
4. Create nuisance vectors from simple prompt pairs
5. Project out nuisance direction from mean diff vectors
6. Save steering vectors and visualizations
"""

from dotenv import load_dotenv
import os
import json
import pickle
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from collections import defaultdict
import argparse
import numpy as np
import seaborn as sns
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import logging
import sys
from pathlib import Path
from datetime import datetime

load_dotenv()

# Font configuration for plots
font_path = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
if os.path.exists(font_path):
    prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()

def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up comprehensive logging."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"create_vector_{timestamp}.log"

    logger = logging.getLogger("create_arena_vector")
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

    # Console handler with progress info
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("=" * 80)
    logger.info("CREATING ARENA CAA STEERING VECTOR")
    logger.info("=" * 80)
    logger.info(f"Log file: {log_file}")

    return logger

def load_jsonl(path: Path, logger: logging.Logger) -> list:
    """Load JSONL file."""
    logger.info(f"Loading: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    logger.info(f"  Loaded {len(data)} examples")
    return data

def get_num_layers(model_config) -> int:
    """
    Get number of layers from model config, handling different attribute names.
    Different model families use different attribute names.
    """
    # Try different possible attribute names
    for attr in ['num_hidden_layers', 'num_layers', 'n_layer', 'n_layers']:
        if hasattr(model_config, attr):
            return getattr(model_config, attr)
    if hasattr(model_config, 'text_config'):
        text_config = model_config.text_config
        for attr in ['num_hidden_layers', 'num_layers', 'n_layer', 'n_layers']:
            if hasattr(text_config, attr):
                return getattr(text_config, attr)
    elif hasattr(model_config, 'model_config'):
        nested_config = model_config.model_config
        for attr in ['num_hidden_layers', 'num_layers', 'n_layer', 'n_layers']:
            if hasattr(nested_config, attr):
                return getattr(nested_config, attr)
    
    # Fallback: raise error with helpful message
    raise AttributeError(
        f"Could not find number of layers in model config. "
        f"Available attributes: {[k for k in dir(model_config) if not k.startswith('_')]}"
    )


def get_hidden_size(model_config) -> int:
    """
    Get number of layers from model config, handling different attribute names.
    Different model families use different attribute names.
    """
    # Try different possible attribute names
    for attr in ['hidden_size', 'd_model', 'n_embd', 'embed_dim', 'dim']:
        if hasattr(model_config, attr):
            return getattr(model_config, attr)
    if hasattr(model_config, 'text_config'):
        text_config = model_config.text_config
        for attr in ['hidden_size', 'd_model', 'n_embd', 'embed_dim', 'dim']:
            if hasattr(text_config, attr):
                return getattr(text_config, attr)
    elif hasattr(model_config, 'model_config'):
        nested_config = model_config.model_config
        for attr in ['hidden_size', 'd_model', 'n_embd', 'embed_dim', 'dim']:
            if hasattr(nested_config, attr):
                return getattr(nested_config, attr)
    
    # Fallback: raise error with helpful message
    raise AttributeError(
        f"Could not find number of layers in model config. "
        f"Available attributes: {[k for k in dir(model_config) if not k.startswith('_')]}"
    )

def chat_template_arena(tokenizer, prompt: str, post_script: str = "<|start_header_id|>assistant<|end_header_id|>") -> str:
    """
    Apply chat template for arena prompts.
    Uses a simple system prompt for comparison.
    """
    system_prompt = "You are a helpful assistant. Please respond with only 'A' or 'B'."

    formatted = tokenizer.apply_chat_template([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ], tokenize=False) + post_script

    return formatted

def accumulate_activations(
    prompts: list,
    sum_accumulators: dict,
    model,
    tokenizer,
    num_layers: int,
    max_tokens: int,
    logger: logging.Logger,
    description: str
):
    """
    Accumulate activations for a set of prompts.

    For each prompt:
    - Extract hidden states at each layer
    - Sum activations at the last N token positions
    """
    logger.info(f"Accumulating activations for {len(prompts)} {description} prompts...")

    # Get device for first layer (embedding layer) - critical for device_map="auto"
    first_layer_device = None
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        first_layer_device = next(model.model.layers[0].parameters()).device
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        first_layer_device = next(model.transformer.h[0].parameters()).device
    elif hasattr(model, 'layers'):
        first_layer_device = next(model.layers[0].parameters()).device
    else:
        # Fallback to first parameter device
        first_layer_device = next(model.parameters()).device

    for prompt in tqdm(prompts, desc=f"Processing {description}", file=sys.stdout):
        # For arena, we'll use the JR_order prompt (first order)
        prompt_text = prompt['jr_prompt']

        # Apply chat template
        formatted_prompt = chat_template_arena(tokenizer, prompt_text)

        # Tokenize
        token_ids = tokenizer(formatted_prompt, add_special_tokens=True)["input_ids"]
        tokens_to_process = min(max_tokens, len(token_ids))

        # Get activations - use first_layer_device for input
        with torch.no_grad():
            outputs = model(
                **tokenizer(formatted_prompt, return_tensors="pt").to(first_layer_device),
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states

        # Accumulate at each offset position
        for offset in range(tokens_to_process):
            for layer_idx in range(num_layers):
                # hidden_states[layer_idx + 1] because layer 0 is embedding
                vec = hidden_states[layer_idx + 1][0, -(offset + 1), :].cpu()
                sum_accumulators[layer_idx][offset] += vec

        # Periodic memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info(f"  Completed accumulation for {description}")

def show_top_token_heatmap_all_layers_offsets(
    layer_proj: dict,
    model,
    tokenizer,
    output_path: Path,
    K: int = 10,
    negative: bool = False,
    prompt_tokens: list = None,
    logger: logging.Logger = None
):
    """
    Create heatmap showing top predicted tokens for each layer and offset.

    Args:
        layer_proj: Dict mapping layer index to list of steering vectors
        model: The model
        tokenizer: The tokenizer
        output_path: Where to save the plot
        K: Number of offsets to visualize
        negative: Whether to negate vectors (for opposite direction)
        prompt_tokens: Optional list of token IDs for x-axis labels
    """
    if logger:
        logger.info(f"Creating heatmap ({'negative' if negative else 'positive'} direction)...")

    model_dtype = next(model.parameters()).dtype
    num_layers = max(layer_proj.keys()) + 1
    
    # Get devices for layer norm and LM head (typically on the last GPU)
    lm_head_device = next(model.lm_head.parameters()).device
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        norm_device = next(model.model.norm.parameters()).device
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'ln_f'):
        norm_device = next(model.transformer.ln_f.parameters()).device
    else:
        norm_device = lm_head_device

    token_matrix = []
    prob_matrix = []

    for layer in range(num_layers):
        layer_tokens = []
        layer_probs = []

        for offset in range(1, K + 1):
            vec = layer_proj[layer][K - offset]
            # Move vector to norm device and correct dtype
            vec = vec.to(norm_device).to(model_dtype)
            vec *= -1 if negative else 1

            # Pass through layer norm - handle different model architectures
            if hasattr(model, 'model') and hasattr(model.model, 'norm'):
                normed = model.model.norm(vec)
            elif hasattr(model, 'transformer') and hasattr(model.transformer, 'ln_f'):
                normed = model.transformer.ln_f(vec)
            elif hasattr(model, 'norm'):
                normed = model.norm(vec)
            else:
                # No norm layer found, use vector as-is
                normed = vec
            
            # Move to LM head device if different
            if norm_device != lm_head_device:
                normed = normed.to(lm_head_device)
            logits = model.lm_head(normed)
            probs = torch.softmax(logits, dim=-1)

            top_idx = torch.argmax(probs).item()
            top_token = tokenizer.decode([top_idx])

            # Filter non-printable
            if not top_token or len(top_token.strip()) == 0 or not all(32 <= ord(c) < 127 for c in top_token):
                top_token = "<unk>"

            top_prob = probs[top_idx].item()
            layer_tokens.append(top_token)
            layer_probs.append(top_prob)

        token_matrix.append(layer_tokens)
        prob_matrix.append(layer_probs)

    token_matrix = np.array(token_matrix)
    prob_matrix = np.array(prob_matrix)

    # Create plot
    plt.figure(figsize=(K + 4, num_layers / 2 + 2))

    if prompt_tokens is not None:
        xticklabels = [tokenizer.decode([t]) for t in prompt_tokens]
    else:
        xticklabels = [f"-{K - i}" for i in range(K)]

    ax = sns.heatmap(
        prob_matrix,
        annot=token_matrix,
        fmt='',
        cmap="Reds" if negative else "Blues",
        xticklabels=xticklabels,
        yticklabels=[f"Layer {i}" for i in range(num_layers)],
        cbar_kws={'label': 'Probability'}
    )

    plt.title(f"Top Token per Layer & Offset ({'Negative' if negative else 'Positive'} Direction)")
    plt.xlabel("Offset (from last token)")
    plt.ylabel("Layer")
    plt.tight_layout()

    # Save both PNG and PDF
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')

    if logger:
        logger.info(f"  Saved heatmap to: {output_path}")

    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Create CAA steering vector from arena examples")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing agreement_examples.jsonl and bias_examples.jsonl')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for steering vectors')
    parser.add_argument('--model_id', type=str, required=True,
                        help='HuggingFace model ID')
    parser.add_argument('--offset', type=int, default=10,
                        help='Number of token positions to extract activations from')
    parser.add_argument('--quantize', action='store_true',
                        help='Use 8-bit quantization')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device map (default: auto)')

    args = parser.parse_args()

    # Set up paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    logger = setup_logging(output_dir)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model: {args.model_id}")
    logger.info(f"Offset: {args.offset}")
    logger.info(f"Quantize: {args.quantize}")

    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    HF_TOKEN = os.getenv("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=HF_TOKEN)

    if args.quantize:
        logger.info("Using 8-bit quantization")
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            device_map=args.device,
            quantization_config=quant_cfg,
            token=HF_TOKEN
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            device_map=args.device,
            token=HF_TOKEN,
            torch_dtype=torch.float16
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.resize_token_embeddings(len(tokenizer))

    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    num_layers = get_num_layers(model.config)
    hidden_size = get_hidden_size(model.config)

    logger.info(f"Model loaded: {num_layers} layers, {hidden_size} hidden size")

    # Load examples
    agreement_examples = load_jsonl(input_dir / "agreement_examples.jsonl", logger)
    bias_examples = load_jsonl(input_dir / "bias_examples.jsonl", logger)

    num_positive = len(agreement_examples)
    num_negative = len(bias_examples)

    # Initialize accumulators for mean difference vectors
    logger.info("Initializing activation accumulators...")
    positive_sums_by_layer = {
        layer: [torch.zeros(hidden_size) for _ in range(args.offset)]
        for layer in range(num_layers)
    }
    negative_sums_by_layer = {
        layer: [torch.zeros(hidden_size) for _ in range(args.offset)]
        for layer in range(num_layers)
    }

    # Accumulate activations for agreement examples (positive)
    accumulate_activations(
        agreement_examples,
        positive_sums_by_layer,
        model,
        tokenizer,
        num_layers,
        args.offset,
        logger,
        "agreement (low P(self))"
    )

    # Accumulate activations for bias examples (negative)
    accumulate_activations(
        bias_examples,
        negative_sums_by_layer,
        model,
        tokenizer,
        num_layers,
        args.offset,
        logger,
        "bias (high P(self))"
    )

    # Compute mean difference vectors
    logger.info("Computing mean difference vectors...")
    layer_mean_diff_vectors = defaultdict(list)

    for layer_idx in range(num_layers):
        for offset in range(args.offset):
            avg_pos = positive_sums_by_layer[layer_idx][offset] / num_positive
            avg_neg = negative_sums_by_layer[layer_idx][offset] / num_negative
            diff = avg_pos - avg_neg
            layer_mean_diff_vectors[layer_idx].append(diff)

    logger.info(f"  Created {sum(len(v) for v in layer_mean_diff_vectors.values())} mean diff vectors")

    # Create nuisance vectors
    logger.info("Creating nuisance vectors...")

    # Define nuisance pairs (simple binary choices)
    yes_no_pairs = [
        ("Say Yes", "Say No"),
        ("Say 1", "Say 2"),
        ("Say first option", "Say second option"),
        ("Say A", "Say B"),
        ("Say Agree", "Say Disagree"),
    ]

    self_pairs = [
        ("Say I", "Say Someone"),
        ("Say I", "Say He"),
        ("Say I", "Say She"),
        ("Say Me", "Say Him"),
        ("Say Me", "Say Her"),
        ("Say My", "Say His"),
        ("Say My", "Say Her"),
        ("Say Myself", "Say Himself"),
        ("Say Myself", "Say Herself"),
        ("Say Mine", "Say His"),
        ("Say Mine", "Say Hers"),
    ]

    bias_pairs = [
        ("Say Better", "Say Worse"),
        ("Say Clear", "Say Confusing"),
        ("Say Helpful", "Say Useless"),
    ]

    nuisance_pairs = yes_no_pairs + self_pairs + bias_pairs
    num_nuisance_pairs = len(nuisance_pairs)

    logger.info(f"  Using {num_nuisance_pairs} nuisance pairs")

    # Create simple prompt dicts for nuisance
    positive_nuisance_prompts = [{'jr_prompt': pos} for pos, _ in nuisance_pairs]
    negative_nuisance_prompts = [{'jr_prompt': neg} for _, neg in nuisance_pairs]

    nuisance_positive_sums = {
        layer: [torch.zeros(hidden_size)]
        for layer in range(num_layers)
    }
    nuisance_negative_sums = {
        layer: [torch.zeros(hidden_size)]
        for layer in range(num_layers)
    }

    accumulate_activations(
        positive_nuisance_prompts,
        nuisance_positive_sums,
        model,
        tokenizer,
        num_layers,
        max_tokens=1,
        logger=logger,
        description="positive nuisance"
    )

    accumulate_activations(
        negative_nuisance_prompts,
        nuisance_negative_sums,
        model,
        tokenizer,
        num_layers,
        max_tokens=1,
        logger=logger,
        description="negative nuisance"
    )

    # Compute nuisance direction per layer
    logger.info("Computing nuisance directions...")
    pairwise_nuisance = {}

    for layer_idx in range(num_layers):
        mean_pos = nuisance_positive_sums[layer_idx][0] / num_nuisance_pairs
        mean_neg = nuisance_negative_sums[layer_idx][0] / num_nuisance_pairs
        diff = mean_pos - mean_neg
        pairwise_nuisance[layer_idx] = diff / diff.norm()

    # Project out nuisance direction
    logger.info("Projecting out nuisance direction from mean diff vectors...")
    projected_vectors_by_layer = defaultdict(list)

    for layer_idx, mean_diff_list in layer_mean_diff_vectors.items():
        nuisance_vec = pairwise_nuisance[layer_idx]
        nuisance_unit = nuisance_vec / nuisance_vec.norm()

        for mean_diff in mean_diff_list:
            residual = mean_diff.clone()
            proj_coef = (residual @ nuisance_unit) / (nuisance_unit.norm() ** 2)
            residual = residual - proj_coef * nuisance_unit
            residual = residual / residual.norm()
            projected_vectors_by_layer[layer_idx].append(residual)

    total_projected = sum(len(v) for v in projected_vectors_by_layer.values())
    total_original = sum(len(v) for v in layer_mean_diff_vectors.values())

    logger.info(f"  Projected {total_projected} vectors out of {total_original} mean-diff vectors")

    # Save steering vectors
    vector_file = output_dir / "steering_vector_arena.pkl"
    logger.info(f"Saving steering vectors to: {vector_file}")

    with open(vector_file, "wb") as f:
        pickle.dump(projected_vectors_by_layer, f)

    logger.info("  Saved!")

    # Create visualizations
    logger.info("Creating visualizations...")

    # Get sample prompt tokens for x-axis labels
    sample_prompt = chat_template_arena(tokenizer, bias_examples[0]['jr_prompt'])
    prompt_tokens = tokenizer(sample_prompt, add_special_tokens=True)['input_ids'][-args.offset:]

    # Stack vectors for visualization
    layer_proj = {k: torch.stack(v) for k, v in projected_vectors_by_layer.items()}

    # Positive direction heatmap
    show_top_token_heatmap_all_layers_offsets(
        layer_proj=layer_proj,
        model=model,
        tokenizer=tokenizer,
        output_path=output_dir / "steering_vector_heatmap_pos",
        K=args.offset,
        negative=False,
        prompt_tokens=prompt_tokens,
        logger=logger
    )

    # Negative direction heatmap
    show_top_token_heatmap_all_layers_offsets(
        layer_proj=layer_proj,
        model=model,
        tokenizer=tokenizer,
        output_path=output_dir / "steering_vector_heatmap_neg",
        K=args.offset,
        negative=True,
        prompt_tokens=prompt_tokens,
        logger=logger
    )

    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'model_id': args.model_id,
        'num_layers': num_layers,
        'hidden_size': hidden_size,
        'offset': args.offset,
        'num_agreement_examples': num_positive,
        'num_bias_examples': num_negative,
        'num_nuisance_pairs': num_nuisance_pairs,
        'quantized': args.quantize
    }

    metadata_file = output_dir / "steering_vector_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata to: {metadata_file}")
    logger.info("=" * 80)
    logger.info("VECTOR CREATION COMPLETE")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
