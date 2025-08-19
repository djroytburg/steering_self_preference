import os
import json
import re
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
from prompts import COMPARISON_SYSTEM_PROMPT
import steering_opt
import argparse

load_dotenv()

parser = argparse.ArgumentParser(description="Optimize steering vector with awareness setting.")
parser.add_argument('--setting', type=str, default="unaware", choices=["aware", "unaware"], help="Prompting setting: aware or unaware.")
parser.add_argument('--layer', type=str, default="14")
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--max_iters', type=int, default=20)
args = parser.parse_args()

TARGET = "llama3.1-8b-instruct"

def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
HF_TOKEN = os.getenv("HFTOKEN")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    token=HF_TOKEN
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
model.eval()
tokenizer.pad_token_id = tokenizer.eos_token_id

# Remove chat template logic, use plain prompts
flip_output = lambda a: "1" if a == "2" else "2"

# Use correct path for bias, lsp examples based on setting
bias_examples_path = os.path.join("steering_inputs", args.setting, "bias_examples.jsonl")
bias_examples = []
with open(bias_examples_path, "r") as f:
    for line in f:
        bias_examples.append(json.loads(line))


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

# Optimization process
print(f"Loaded {len(bias_examples)} bias examples for setting: {args.setting}")
datapoints = []
for bias_example in tqdm(bias_examples):
    text = chat_template(bias_example['prompt'])
    src_completion = flip_output(bias_example['unbiased_output'])
    dst_completion = bias_example['unbiased_output']
    datapoints.append(
        steering_opt.TrainingDatapoint(
            text,
            src_completions=[src_completion],
            dst_completions=[dst_completion]
        )
    )

layer = min(int(args.layer), model.config.num_hidden_layers)
vector, losses = steering_opt.optimize_completion(
    model, datapoints, layer, tokenizer=tokenizer,
    lr=args.lr, max_iters=args.max_iters, use_transformer_lens=False,
    do_target_loss_avg=False, return_loss=True,
    target_loss=None, target_loss_target_iters=5,
    debug=True
)

print("Steering vector optimized.")

# Save the vector to workspace-relative path
out_path = os.path.join("vectors", "optimization", f"steering_vector_{args.setting}_final.pkl")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
import pickle
with open(out_path, "wb") as f:
    pickle.dump(vector, f)
print(f"Optimized steering vector saved to {out_path}")