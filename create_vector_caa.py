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

font_path = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = prop.get_name()
from prompts import COMPARISON_SYSTEM_PROMPT

load_dotenv()

parser = argparse.ArgumentParser(description="Create steering vector with awareness setting.")
parser.add_argument('--aware', action='store_true', help='Use aware setting if specified.')
parser.add_argument('--offset', type=int, default=10)
args = parser.parse_args()

## What goes after the chat template
post_script="<|begin_header_id|>assistant<|end_header_id|>"

HF_TOKEN = os.getenv("HF_TOKEN")
quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

tok   = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            quantization_config=quant_cfg,
            token=HF_TOKEN
        )

if tok.pad_token is None:
    tok.pad_token = tok.eos_token
    model.resize_token_embeddings(len(tok))
model.config.pad_token_id = tok.pad_token_id
model.eval()

num_layers = model.config.num_hidden_layers
hidden_size = model.config.hidden_size

def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]
setting = "aware" if args.aware else "unaware"
agreement_examples = load_jsonl(f'steering_inputs/{setting}/agreement_examples.jsonl')
bias_examples = load_jsonl(f'steering_inputs/{setting}/bias_examples.jsonl')

# ─── DEFINE PAIRS CONCISELY AS LIST LITERALS ─────────────────────────────────────
yes_no_pairs = [
    ("Say Yes",           "Say No"),
    ("Say 1",             "Say 2"),
    ("Say first option",  "Say second option"),
    ("Say A",             "Say B"),
    ("Say Agree",         "Say Disagree"),
]

self_pairs = [
    ("Say I",      "Say Someone"),
    ("Say I",      "Say He"),
    ("Say I",      "Say She"),
    ("Say Me",     "Say Him"),
    ("Say Me",     "Say Her"),
    ("Say My",     "Say His"),
    ("Say My",     "Say Her"),
    ("Say Myself", "Say Himself"),
    ("Say Myself", "Say Herself"),
    ("Say Mine",   "Say His"),
    ("Say Mine",   "Say Hers"),
]

bias_pairs = [
    ("Say Better", "Say Worse"),
    ("Say Clear",  "Say Confusing"),
    ("Say Helpful","Say Useless"),
]

nuisance_pairs = yes_no_pairs + self_pairs + bias_pairs

positive_nuisance_prompts = [positive for positive, _ in nuisance_pairs]
negative_nuisance_prompts = [negative for _, negative in nuisance_pairs]

print(f"Positive nuisance prompts: {len(positive_nuisance_prompts)}")
print(f"Negative nuisance prompts: {len(negative_nuisance_prompts)}")

# MEAN DIFF VECTOR VARIABLES
num_positive = len(agreement_examples) 
num_negative = len(bias_examples)

positive_sums_by_layer = {
    layer: [torch.zeros(hidden_size) for _ in range(args.offset)]
    for layer in range(num_layers)
}
negative_sums_by_layer = {
    layer: [torch.zeros(hidden_size) for _ in range(args.offset)]
    for layer in range(num_layers)
}

# NUISANCE VECTOR VARIABLES
num_nuisance_pairs = len(nuisance_pairs)
nuisance_positive_sums = {
    layer: [torch.zeros(hidden_size)]
    for layer in range(num_layers)
}
nuisance_negative_sums = {
    layer: [torch.zeros(hidden_size)]
    for layer in range(num_layers)
}

def accumulate_activations(prompts, sum_accumulators, num_layers, max_tokens):
    for prompt in tqdm(prompts, desc="Accumulating activations"):
        prompt = tok.apply_chat_template([
            {
                "role": "system", 
                "content": COMPARISON_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }],tokenize=False) + "<|start_header_id|>assistant<|end_header_id|>"
        token_ids = tok(prompt, add_special_tokens=True)["input_ids"]
        tokens_to_process = min(max_tokens, len(token_ids))
        with torch.no_grad():
            outputs = model(
                **tok(prompt, return_tensors="pt").to(model.device),
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states
        for offset in range(tokens_to_process):
            for layer_idx in range(num_layers):
                vec = hidden_states[layer_idx + 1][0, -(offset + 1), :].cpu()
                sum_accumulators[layer_idx][offset] += vec

# ─── COMPUTE LAYER‐MEAN‐DIFFERENCE VECTORS ─────────────────────────────────────────
agreement_prompts = [example['prompt'] for example in agreement_examples]
bias_prompts = [example['prompt'] for example in bias_examples]

accumulate_activations(agreement_prompts,   positive_sums_by_layer, num_layers, args.offset)
accumulate_activations(bias_prompts,  negative_sums_by_layer, num_layers, args.offset)

layer_mean_diff_vectors = defaultdict(list)
for layer_idx in range(num_layers):
    for offset in range(args.offset):
        avg_pos = positive_sums_by_layer[layer_idx][offset] / num_positive
        avg_neg = negative_sums_by_layer[layer_idx][offset] / num_negative
        diff    = avg_pos - avg_neg
        normalized = diff / diff.norm()
        layer_mean_diff_vectors[layer_idx].append(diff)

# ─── COMPUTE ONE “NUISANCE” VECTOR PER LAYER ───────────────────────────────────────
accumulate_activations(positive_nuisance_prompts, nuisance_positive_sums, num_layers, max_tokens=1)
accumulate_activations(negative_nuisance_prompts, nuisance_negative_sums, num_layers, max_tokens=1)

pairwise_nuisance = {}
for layer_idx in range(num_layers):
    mean_pos = nuisance_positive_sums[layer_idx][0] / num_nuisance_pairs
    mean_neg = nuisance_negative_sums[layer_idx][0] / num_nuisance_pairs
    diff     = mean_pos - mean_neg
    pairwise_nuisance[layer_idx] = diff / diff.norm()

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
total_original  = sum(len(v) for v in layer_mean_diff_vectors.values())

print(f"Projected {total_projected} vectors out of {total_original} mean-diff vectors")

with open(f"vectors/caa/steering_vector_{setting}_final.pkl", "wb") as f:
    pickle.dump(projected_vectors_by_layer, f)


print(f"Saved vector to vectors/caa/steering_vector_{setting}_final.pkl")

def show_top_token_heatmap_all_layers_offsets(layer_proj, model, tokenizer, K=10, negative = False, prompt_tokens=None):
    """
    Shows a heatmap of the top token (decoded) for each layer and offset.
    Rows: layers (0-based)
    Columns: offsets (Kth-to-last to last token)
    """
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    num_layers = max(layer_proj.keys()) + 1  # layers are 0-indexed

    # Prepare arrays for tokens and probabilities
    token_matrix = []
    prob_matrix = []

    for layer in range(num_layers):
        layer_tokens = []
        layer_probs = []
        for offset in range(1, K + 1):
            vec = layer_proj[layer][K-offset]
            vec = vec.to(device).to(model_dtype)
            vec *= -1 if negative else 1
            normed = model.model.norm(vec)
            logits = model.lm_head(normed)
            probs = torch.softmax(logits, dim=-1)
            top_idx = torch.argmax(probs).item()
            top_token = tokenizer.decode([top_idx])
            if not top_token or len(top_token.strip()) == 0 or not all(32 <= ord(c) < 127 for c in top_token):
                top_token = "<unk>"
            top_prob = probs[top_idx].item()
            layer_tokens.append(top_token)
            layer_probs.append(top_prob)
        token_matrix.append(layer_tokens)
        prob_matrix.append(layer_probs)

    token_matrix = np.array(token_matrix)
    prob_matrix = np.array(prob_matrix)

    plt.figure(figsize=(K+4, num_layers/2+2))
    if prompt_tokens is not None:
        assert len(prompt_tokens) == token_matrix.shape[1], (len(prompt_tokens), token_matrix.shape)
        xticklabels = [tokenizer.decode([t]) for t in prompt_tokens]
    else:   xticklabels=[f"-{K-i}" for i in range(K)]

    ax = sns.heatmap(prob_matrix, annot=token_matrix, fmt='', cmap="Reds" if negative else "Blues",
                     xticklabels=xticklabels,
                     yticklabels=[f"Layer {i}" for i in range(num_layers)])
    plt.title(f"Top Token per Layer & Offset")
    plt.xlabel("Offset (from last token)")
    plt.ylabel("Layer")
    plt.savefig(f"vectors/caa/steering_vector_{setting}_heatmap{'_neg' if negative else ''}.pdf", dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

def chat_template(prompt, post_script="<|start_header_id|>assistant<|end_header_id|>"):
    prompt = tok.apply_chat_template([
            {
                "role": "system", 
                "content": COMPARISON_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }],tokenize=False) + post_script
    return prompt

prompt_tokens = []
for decoded_prompt_token in tok(chat_template(bias_prompts[0], post_script=post_script), add_special_tokens=True)['input_ids'][-args.offset:]:
    prompt_tokens.append(decoded_prompt_token)

# Example usage:
layer_proj = {k: torch.stack(v) for k, v in projected_vectors_by_layer.items()}

show_top_token_heatmap_all_layers_offsets(
    layer_proj=layer_proj,
    model=model,
    tokenizer=tok,
    negative=False,
    K=args.offset,
    prompt_tokens=prompt_tokens
)

show_top_token_heatmap_all_layers_offsets(
    layer_proj=layer_proj,
    model=model,
    tokenizer=tok,
    negative=True,
    K=args.offset,
    prompt_tokens=prompt_tokens
)