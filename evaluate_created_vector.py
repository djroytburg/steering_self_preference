import os
import json
import random
import pickle
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
from prompts import (
    COMPARISON_PROMPT_TEMPLATE_AB_FULL_AWARE,
    COMPARISON_PROMPT_TEMPLATE,
    COMPARISON_PROMPT_TEMPLATE_LABELS,
    COMPARISON_PROMPT_TEMPLATE_SELF_OTHER_AWARE,
    LLAMA_PROMPT_TEMPLATE,
    COMPARISON_SYSTEM_PROMPT,
    COMPARISON_SYSTEM_PROMPT_AWARE
)


from enhanced_hooking import (
    get_activations,
    add_activations_and_generate,
    zeroout_projections_and_generate,
    clear_hooks,
)

from data import load_data

# --- Argparse ---
parser = argparse.ArgumentParser(description="Evaluate created steering vector.")
parser.add_argument('--vector_path', type=str, default="vectors/caa/steering_vector_unaware_final.pkl")
parser.add_argument('--layers', type=int, nargs='+', default=[14,15,16])
parser.add_argument('--multipliers', type=float, nargs='+', default=[-0.5,-0.3,-0.1,0.1,0.3,0.5])
parser.add_argument('--num_samples', type=int, default=30)
parser.add_argument('--offset', type=int, default=0)
parser.add_argument('--results_path', type=str, default="steering_evals/caa/unaware/classic/results.jsonl")
args = parser.parse_args()

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN")

if torch.cuda.is_available():
    print("CUDA is available!")
else:
    print("CUDA is NOT available.")
    
# --- Load model and tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map='auto',
    token=HF_TOKEN
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
model.eval()
transformers.logging.set_verbosity_info()
sampling_kwargs={"use_cache": True, "pad_token_id": tokenizer.eos_token_id, "max_new_tokens": 10,"return_dict_in_generate": True, "output_scores": True, "do_sample": False}
tokenizer.padding_side = "left"

# --- Load steering vector ---
with open(args.vector_path, "rb") as f:
    steering_vector = pickle.load(f)

# --- Load datasets ---
def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

positives = load_jsonl("preference_extraction/unaware/xsum_llama3.1-8b-instruct_agreement_examples.jsonl")
negatives = load_jsonl("preference_extraction/unaware/xsum_llama3.1-8b-instruct_bias_examples.jsonl")
test_lsp = load_jsonl("preference_extraction/unaware/xsum_llama3.1-8b-instruct_legit_self_pref_examples.jsonl")

positives_steering = load_jsonl("steering_inputs/unaware/agreement_examples.jsonl")
negatives_steering = load_jsonl("steering_inputs/unaware/bias_examples.jsonl")

responses, articles, keys = load_data("xsum", sources= ['gpt35','llama3.1-8b-instruct'],target_model='llama3.1-8b-instruct',num_samples=1000, extras=False)

vector_positive_ids = [ex['id'] for ex in positives_steering]
test_positives = [ex for ex in positives if ex not in vector_positive_ids]

vector_negative_ids = [ex['id'] for ex in negatives_steering]
test_negatives = [ex for ex in negatives if ex not in vector_negative_ids]


def reconstruct(result, responses, articles, source='llama3.1-8b-instruct', set_aware = False, self_other_labels = False, source_summary_first = True, invert_option_order = False):
    """
    Four-way reconstruction with aware and unaware prompting mechanisms.
    
    Arguments:
        - result (dict): result object according to DBG score output.
        - responses (dict): dictionary of responses created by load_data from preference_experiments/load_data.py
        - articles (dict): dictionary of articles created by load_data from preference_experiments/load_data.py
        - source (str): the source model to use for the first summary. Default 'llama3.1-8b-instruct'.
        - set_aware (bool): whether to use the aware or unaware prompt template. Defaults to use_aware, set above.
        - source_summary_first (bool): whether the first summary is from the source model or the other model. Defaults to True.
        - invert_option_order (bool): whether the first summary is the first or second choice. Defaults to False.
    
    Returns the correct prompt string for options, and the expected output letter that indicates self-preference.
    """
    article = articles[result['id']]
    source_summary = responses[source][result['id']]
    other_summary = responses['gpt35'][result['id']]
    if not set_aware: # Standard condition
        if not self_other_labels:
            prompt_template = COMPARISON_PROMPT_TEMPLATE
            summary1 = source_summary if source_summary_first else other_summary
            summary2 = other_summary if source_summary_first else source_summary
            
            unbiased_output = "2" if source_summary_first else "1"
            if result['gold_judges_consensus']['prefers_target']:
                unbiased_output = "1" if unbiased_output == "2" else "2"

            return prompt_template.format(
                article = article,
                summary1 = summary1,
                summary2 = summary2
            ), unbiased_output
        elif self_other_labels:
            prompt_template = COMPARISON_PROMPT_TEMPLATE_LABELS
            summary1 = source_summary if source_summary_first else other_summary
            summary2 = other_summary if source_summary_first else source_summary
            
            unbiased_output = "[]" if source_summary_first else "[]"
            if result['gold_judges_consensus']['prefers_target']:
                unbiased_output = "This" if unbiased_output == "That" else "That"

            return prompt_template.format(
                article = article,
                summary1 = summary1,
                summary2 = summary2,
            ), unbiased_output
    else:
        if not self_other_labels:
            prompt_template = COMPARISON_PROMPT_TEMPLATE_AB_FULL_AWARE
            summary1 = source_summary if source_summary_first else other_summary
            summary2 = other_summary if source_summary_first else source_summary
            who1 = "your" if source_summary_first else "other model's"
            who2 = "other model's" if source_summary_first else "your"
            first_summary = "Summary1" if not invert_option_order else "Summary2"
            second_summary = "Summary2" if not invert_option_order else "Summary1"

            unbiased_output = "2" if source_summary_first != invert_option_order else "1"
            if result['gold_judges_consensus']['prefers_target']:
                unbiased_output = "1" if unbiased_output == "2" else "2"
            return prompt_template.format(
                article = article,
                summary1 = summary1,
                summary2 = summary2,
                first_summary = first_summary,
                second_summary = second_summary,
                who1 = who1,
                who2 = who2
            ), unbiased_output
random.seed(42)

def subsample(data, n):
    return random.sample(data, min(n, len(data)))

positives = subsample(test_positives, args.num_samples)
negatives = subsample(test_negatives, args.num_samples)
lsp = subsample(test_lsp, args.num_samples)

print(positives[0].items())
print(negatives[0].items())
print(lsp[0].items())
# --- Steering layer initialization ---
for layer_idx, vec_list in steering_vector.items():
    
    vec = vec_list[0]
    # Make sure dtype/device matches the model you’ll run on
    vec = vec.to(dtype=model.dtype, device=next(model.parameters()).device)

# --- Generate Function ---

def generate_with_vec(prompt_str, layer_idx, base_vec, scale):
    clear_hooks(model)
    tokens = tokenizer(prompt_str, return_tensors="pt").to(model.device)
    vec = (base_vec * scale).to(model.dtype)
    ids_pos, scores = add_activations_and_generate(
        model,
        tokens,
        specificpos_layer_activations={layer_idx: {-1: vec}},
        continuouspos_layer_activations={},
        sampling_kwargs=sampling_kwargs,
        add_at="end"
    )
    
    txt_list = [(tokenizer.decode(token), prob.item()) for token, prob in zip(ids_pos[0], scores[0])]
    return txt_list

# --- Main evaluation loop ---
run_id = 0
results = []
for dataset_name, dataset in zip(["positives", "negatives"], [positives, negatives]):
    for example in tqdm(dataset, desc=dataset_name):
        for source_summary_first in [True, False]:
            for self_other_labels in [True]:
                prompt_text, unbiased_output = reconstruct(
                    example,
                    responses,
                    articles,
                    source='llama3.1-8b-instruct',
                    set_aware=False,
                    source_summary_first=source_summary_first,
                    self_other_labels=self_other_labels
                )
                prompt_text += " \n\n"    
                for layer in args.layers:
                    if layer not in steering_vector:
                        print(f"No vector for layer {layer}; skipping")
                        continue
                    base_vec = steering_vector[layer][args.offset].to(
                        dtype=model.dtype, device=model.device
                    )
                    for m in args.multipliers:
                        comp = generate_with_vec(
                            prompt_text,
                            layer_idx=layer,
                            base_vec=base_vec,
                            scale=m
                        )
                        result = {
                            "id": example['id'],
                            "source_summary_first": source_summary_first,
                            "self_other_labels": self_other_labels,
                            "dataset": dataset_name,
                            "layer": layer,
                            "mult": m,
                            "unbiased_output": unbiased_output,
                            "output": comp,
                            "prompt": prompt_text,
                        }
                        results.append(result)
                        with open(args.results_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        run_id += 1
print(f"Saved {len(results)} generations to {args.results_path}")
