import os
import json
import re
import torch
import torch.nn as nn
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv
from prompts import COMPARISON_SYSTEM_PROMPT
import steering_opt_batched as steering_opt
import argparse

load_dotenv()

parser = argparse.ArgumentParser(description="Optimize steering vector with awareness setting.")
parser.add_argument('--setting', type=str, default="unaware", choices=["aware", "unaware"], help="Prompting setting: aware or unaware.")
parser.add_argument('--layer', type=str, default="14")
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--max_iters', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=16, help="Mini-batch size for processing (lower = less memory)")
args = parser.parse_args()

TARGET = "llama3.1-8b-instruct"

def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
HF_TOKEN = os.getenv("HFTOKEN")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# Use DataParallel for parallel processing
print(f"Detected {torch.cuda.device_count()} GPUs")

if torch.cuda.device_count() > 1:
    print(f"Using DataParallel with {torch.cuda.device_count()} GPUs (fp16 mode)")
    # DataParallel requires fp16
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map=None,  # Important
        torch_dtype=torch.float16,
        token=HF_TOKEN
    )
    print("Successfully loaded fp16 model")
    
    # Resize embeddings before wrapping
    if tokenizer.pad_token is not None:
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    
    # Wrap
    model = nn.DataParallel(model)
    model = model.cuda()
    print("Model wrapped with DataParallel")
else:
    print("Single GPU detected, using 8-bit quantization")
    quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        quantization_config=quant_cfg,
        token=HF_TOKEN
    )
    print("Successfully loaded 8-bit model")
    
    # Resize embeddings if needed
    if tokenizer.pad_token is not None:
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

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

# Handle DataParallel wrapped models
if hasattr(model, 'module'):
    num_layers = model.module.config.num_hidden_layers
else:
    num_layers = model.config.num_hidden_layers
layer = min(int(args.layer), num_layers)
print(f"Starting optimization (batched) with batch_size={args.batch_size}...")
start_time = time.time()

vector, losses = steering_opt.optimize_completion(
    model, datapoints, layer, tokenizer=tokenizer,
    lr=args.lr, max_iters=args.max_iters, use_transformer_lens=False,
    do_target_loss_avg=False, return_loss=True,
    target_loss=None, target_loss_target_iters=5,
    debug=True,
    batch_size=args.batch_size
)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"\n{'='*60}")
print(f"Steering vector optimized successfully!")
print(f"Time elapsed: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
print(f"{'='*60}\n")

# Save the vector to workspace-relative path
out_path = os.path.join("vectors", "optimization", f"steering_vector_{args.setting}_final.pkl")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
import pickle
with open(out_path, "wb") as f:
    pickle.dump(vector, f)
print(f"Optimized steering vector saved to {out_path}")