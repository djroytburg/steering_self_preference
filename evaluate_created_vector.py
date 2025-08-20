import os
import json
import random
import pickle
import argparse
from tqdm import tqdm
import torch
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
import steering_opt

# --- Argparse ---
parser = argparse.ArgumentParser(description="Evaluate created steering vector.")
parser.add_argument('--steering_type', type=str, default="caa", choices = ["caa", "optimization"])
parser.add_argument('--setting', type=str, default="unaware", choices=["aware", "unaware"])
parser.add_argument('--vector_path', type=str, default="vectors/")
parser.add_argument('--prompt_template', type=str, default='classic', choices = ['classic', 'self-other'])
parser.add_argument('--layers', type=int, nargs='+', default=[14,15,16])
parser.add_argument('--cases', type=str, default="bias,agreement,lsp")
parser.add_argument('--multipliers', type=float, nargs='+', default=[-0.5,-0.3,-0.1,0.1,0.3,0.5])
parser.add_argument('--num_samples', type=int, default=30)
parser.add_argument('--offset', type=int, default=10)
parser.add_argument('--results_path', type=str, default="steering_evals")
parser.add_argument('--seed', type=int, default=42)

args = parser.parse_args()

if args.steering_type == "optimization" and len(args.layers) > 1:
    raise ValueError("Optimization steering only supports a single layer. Please specify one layer.")

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
sampling_kwargs={"use_cache": True, "pad_token_id": tokenizer.eos_token_id, "max_new_tokens": 10,"return_dict_in_generate": True, "temperature": None, "top_p": None, "output_scores": True, "do_sample": False}
tokenizer.padding_side = "left"

# --- Load steering vector ---
vector_path = os.path.join(args.vector_path, args.steering_type, f"steering_vector_{args.setting}_final.pkl")
with open(vector_path, "rb") as f:
    steering_vector = pickle.load(f)

# --- Load datasets ---
def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

if any([case not in ["bias", "agreement", "lsp"] for case in args.cases.split(",")]):
    raise ValueError("Invalid case specified. Choose from 'bias', 'agreement', 'lsp'.")


def subsample(data, n, seed = args.seed):
    random.seed(seed)
    return random.sample(data, min(n, len(data)))

cases = args.cases.split(",")
case_to_dataset = {}
if "agreement" in cases:
    positives = load_jsonl(f"preference_extraction/{args.setting}/xsum_llama3.1-8b-instruct_agreement_examples.jsonl")
    positives_steering = load_jsonl(f"steering_inputs/{args.setting}/agreement_examples.jsonl")
    vector_positive_ids = [ex['id'] for ex in positives_steering]
    test_positives = [ex for ex in positives if ex['id'] not in vector_positive_ids]
    case_to_dataset['agreement'] = subsample(test_positives, args.num_samples)

if "bias" in cases:
    negatives = load_jsonl(f"preference_extraction/{args.setting}/xsum_llama3.1-8b-instruct_bias_examples.jsonl")
    negatives_steering = load_jsonl(f"steering_inputs/{args.setting}/bias_examples.jsonl")
    vector_negative_ids = [ex['id'] for ex in negatives_steering]
    test_negatives = [ex for ex in negatives if ex['id'] not in vector_negative_ids]
    case_to_dataset['bias'] = subsample(test_negatives, args.num_samples)

if "lsp" in cases:
    test_lsp = load_jsonl(f"preference_extraction/{args.setting}/xsum_llama3.1-8b-instruct_legit_self_pref_examples.jsonl")
    case_to_dataset['lsp'] = subsample(test_lsp, args.num_samples)

responses, articles, keys = load_data("xsum", sources= ['gpt35','llama3.1-8b-instruct'],target_model='llama3.1-8b-instruct',num_samples=1000, extras=False)


def chat_template(prompt, post_script="<|start_header_id|>assistant<|end_header_id|>\n\n"):
    prompt = tokenizer.apply_chat_template([
            {
                "role": "system", 
                "content": COMPARISON_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }],tokenize=False) + post_script
    return prompt


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
            
            unbiased_output = "That" if source_summary_first else "This"
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



# --- Steering layer initialization ---
if args.steering_type == "caa":
    for layer_idx, vec_list in steering_vector.items():
        
        vec = vec_list[0]
        # Make sure dtype/device matches the model you’ll run on
        vec = vec.to(dtype=model.dtype, device=next(model.parameters()).device)

# --- Generate Function ---

def generate_with_vec(prompt_str, layer_idx, base_vec, scale, steering_type= args.steering_type, score_on_token=None):
    if steering_type == "caa":
        clear_hooks(model)
        tokens = tokenizer(prompt_str, return_tensors="pt").to(model.device)
        vec = (base_vec * scale).to(model.dtype)
        ids_pos, scores = add_activations_and_generate(
            model,
            tokens,
            specificpos_layer_activations={layer_idx: {-1: vec}},
            continuouspos_layer_activations={},
            sampling_kwargs=sampling_kwargs,
            add_at="end",
            score_on_token=score_on_token
        )
        
    elif steering_type == "optimization":
        with steering_opt.hf_hooks_contextmanager(model, [
            (layer_idx, steering_opt.make_steering_hook_hf(scale * base_vec, steering_opt.make_abl_mat(scale * base_vec)))]):
            # print("tokenizing")
            tokens = tokenizer(chat_template(prompt_str), return_tensors='pt').to(model.device)
            # print("generating")
            output = model.generate(**tokens, **sampling_kwargs)
            stacked_scores = torch.stack(output.scores, dim=0)
            stacked_scores = stacked_scores.permute(1, 0, 2)   # -> Tensor(batch_size, new_tokens, vocab_size)
            probabilities = stacked_scores.softmax(dim=-1)  # -> Softmax over vocab_size for probabilities
            if score_on_token is None:
                highest_probabilities, highest_score_indices = torch.max(probabilities, dim=-1) # -> Get probabilities of tokens
                generated_sequences = output.sequences
                start_pos = generated_sequences.shape[1] - highest_score_indices.shape[1] # New tokens only
                generated_tokens_ids = generated_sequences[:, start_pos:]
                
                assert torch.equal(generated_tokens_ids, highest_score_indices), (generated_tokens_ids, highest_score_indices)
                ids_pos, scores = generated_tokens_ids.detach().cpu(), highest_probabilities.detach().cpu()
            else:
                score_on_token = tokenizer.convert_tokens_to_ids(score_on_token)
                if score_on_token < 0 or score_on_token >= probabilities.shape[-1]:
                    raise ValueError(f"score_on_token {score_on_token} is out of bounds for the model's vocabulary size {probabilities.shape[-1]}.")
                scores = probabilities[:, :, score_on_token].detach().cpu()      
                ids_pos = torch.full(scores.shape, score_on_token)  
    txt_list = [(tokenizer.decode(token), prob.item()) for token, prob in zip(ids_pos[0], scores[0])]
    return txt_list

self_other_labels = args.prompt_template == 'self-other'
set_aware = args.setting == 'aware'

results_path = os.path.join(args.results_path, args.steering_type, args.setting, args.prompt_template, "results.jsonl")

# --- Main evaluation loop ---
run_id = 0
results = []
for dataset_name, dataset in case_to_dataset.items():
    for example in tqdm(dataset, desc=dataset_name):
        for source_summary_first in [True, False]:
            prompt_text, unbiased_output = reconstruct(
                example,
                responses,
                articles,
                source='llama3.1-8b-instruct',
                set_aware=set_aware,
                source_summary_first=source_summary_first,
                self_other_labels=self_other_labels
            )
            prompt_text += " \n\n"    
            for layer in args.layers:
                if args.steering_type == "caa":
                    if layer not in steering_vector:
                        print(f"No vector for layer {layer}; skipping")
                        continue
                    base_vec = steering_vector[layer][args.offset].to(
                        dtype=model.dtype, device=model.device
                    )
                else:
                    base_vec = steering_vector
                for m in args.multipliers:
                    comp = generate_with_vec(
                        prompt_text,
                        layer_idx=layer,
                        base_vec=base_vec,
                        scale=m,
                        score_on_token=unbiased_output
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
                    with open(results_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    run_id += 1
print(f"Saved {len(results)} generations to {results_path}")
