#!/usr/bin/env python3
from datetime import datetime
import os
import sys
import torch
import torch.nn.functional as F
import argparse
import json
import math
import numpy as np
import logging
import getpass
import socket
import platform
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

load_dotenv()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
# ---------------------------
# Logging Setup
# ---------------------------

def setup_logging(output_dir: Path, args) -> logging.Logger:
    """
    Set up logging with comprehensive metadata about the run.
    Creates both console and file handlers.
    Logs are saved in output_dir/logs/
    """
    # Create logs directory within the output directory
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"run_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger("vllm_self_preference")
    logger.setLevel(logging.DEBUG)
    
    # Prevent duplicate handlers if function is called multiple times
    if logger.handlers:
        logger.handlers.clear()
    
    # File handler with detailed formatting
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
    
    # Log comprehensive run metadata
    logger.info("=" * 80)
    logger.info("NEW RUN STARTED")
    logger.info("=" * 80)
    
    # System metadata
    logger.info("SYSTEM METADATA:")
    logger.info(f"  User: {getpass.getuser()}")
    logger.info(f"  Hostname: {socket.gethostname()}")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  Python version: {sys.version.split()[0]}")
    logger.info(f"  PyTorch version: {torch.__version__}")
    logger.info(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"  CUDA version: {torch.version.cuda}")
        logger.info(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            logger.info(f"    Total memory: {props.total_memory / 1e9:.2f} GB")
    
    # Run parameters
    logger.info("RUN PARAMETERS:")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"  {arg}: {value}")
    
    logger.info(f"  Log file: {log_file}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Timestamp: {timestamp}")
    logger.info("=" * 80)
    
    return logger

# ---------------------------
# Helpers
# ---------------------------

def expand(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))

def _truncate(s: Optional[str], n: int = 200) -> str:
    if s is None:
        return ""
    s = str(s)
    if len(s) <= n:
        return s
    return s[:n] + f"...<truncated {len(s)} chars>"

def build_messages(prompt: str, answer1: str, answer2: str) -> List[Dict[str, str]]:
    system_msg = "You are an impartial and unbiased judge."
    answer1 = answer1.replace("<think>", "").replace("</think>", "")
    answer2 = answer2.replace("<think>", "").replace("</think>", "")
    user_content = (
        "You will be given answers from two assistants (Assistant A and Assistant B) to a user prompt. "
        "Please act as an impartial judge and evaluate the quality of their answers. "
        "You must output only one of the following choices:\n"
        "'A' if Assistant A's answer is better.\n"
        "'T' if both answers are relatively the same in quality.\n"
        "'B' if Assistant B's answer is better.\n"
        "Do not think or reason. Just choose which answer you prefer: A or B or T.\n"
        "Output only one letter: A, B, or T. No spaces, no newlines, no explanation.\n"
        f"{prompt}\n"
        "<The Start of Assistant A's Answer>\n"
        f"{answer1}\n"
        "<The End of Assistant A's Answer>\n"
        "<The Start of Assistant B's Answer>\n"
        f"{answer2}\n"
        "<The End of Assistant B's Answer>\n"
        "Do not think or reason. Just choose which answer you prefer: A or B or T.\n"
    )
    return [{"role": "system", "content": system_msg},
            {"role": "user", "content": user_content}]

ABT_STRIPPED = {"A", "B", "T"}

def decode_token(tokenizer, tid: int) -> str:
    try:
        return tokenizer.decode([tid], skip_special_tokens=False)
    except Exception:
        return ""

def logits_to_prob_map(logits_tensor, tokenizer, topk: int = 200, logger: Optional[logging.Logger] = None) -> Dict[str, float]:
    """
    Convert logits tensor to {decoded_token_text -> probability}.
    Uses log-softmax and takes top-k tokens.
    """
    if logger is None:
        logger = logging.getLogger("vllm_self_preference")
    
    text_to_prob: Dict[str, float] = {}
    
    logger.debug(f"logits_to_prob_map: input shape={logits_tensor.shape}, dtype={logits_tensor.dtype}")
    
    # Compute log probabilities
    log_probs = F.log_softmax(logits_tensor, dim=-1)
    
    # Get top-k indices
    k = min(topk, logits_tensor.shape[-1])
    topk_logprobs, topk_indices = torch.topk(log_probs, k, dim=-1)
    
    # Convert to CPU and numpy for processing
    topk_logprobs_np = topk_logprobs.detach().cpu().numpy()
    topk_indices_np = topk_indices.detach().cpu().numpy()
    
    logger.debug(f"Top-k extraction: k={k}, top logprobs range=[{topk_logprobs_np.min():.4f}, {topk_logprobs_np.max():.4f}]")
    
    for i, (token_id, logprob) in enumerate(zip(topk_indices_np, topk_logprobs_np)):
        token_id = int(token_id)
        logprob = float(logprob)
        prob = math.exp(logprob)
        ttext = decode_token(tokenizer, token_id)
        text_to_prob[str(ttext)] = prob
        
        if i < 10:  # Log first 10 for debugging
            logger.debug(f"topk[{i}]: token_id={token_id}, text='{_truncate(ttext, 40)}', logprob={logprob:.4f}, prob={prob:.6f}")
    
    return text_to_prob

def only_abt_probs(text_prob_map: Dict[str, float], logger: Optional[logging.Logger] = None) -> Dict[str, float]:
    """Collect probabilities for A/B/T tokens (and common whitespace-prefixed variants)."""
    if logger is None:
        logger = logging.getLogger("vllm_self_preference")
    
    aliases = {
        "A": {"A", "▁A", " A", "\nA", "\tA"},
        "B": {"B", "▁B", " B", "\nB", "\tB"},
        "T": {"T", "▁T", " T", "\nT", "\tT"},
    }
    out = {"A": 0.0, "B": 0.0, "T": 0.0}
    for letter, alset in aliases.items():
        out[letter] = max((text_prob_map.get(a, 0.0) for a in alset), default=0.0)
    
    logger.debug(f"only_abt_probs result: {out}")
    return out

def normalize_probs_abc(prob_map: Dict[str, float]) -> Dict[str, float]:
    total = prob_map["A"] + prob_map["B"] + prob_map["T"]
    if total <= 0:
        # fall back to uniform so they sum to 1
        return {"A": 1/3, "B": 1/3, "T": 1/3}
    return {k: prob_map[k] / total for k in ["A", "B", "T"]}

# ---------------------------
# Main
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="A/B/T probabilities for two prompt orders.")
    p.add_argument("--output_path", type=str, required=True, help="Where to write JSONL output.")
    p.add_argument("--judge", type=str, default="google/gemma-3-12b-it")
    p.add_argument("--ref", type=str, default='deepseek-ai/DeepSeek-R1-Distill-Qwen-32B')
    p.add_argument("--data_jsonl", type=str,
                   default="diffs.jsonl",
                   help="JSONL fields: question, judge_completion, ref_completion; optional: judge_correct, ref_correct.")
    p.add_argument("--limit", type=int, default=0, help="Max rows; 0 = all.")
    # Model
    p.add_argument("--bf16", action="store_true", help="Use bfloat16 precision")
    p.add_argument("--trust_remote_code", action="store_true")
    # Decoding
    p.add_argument("--logprobs_k", type=int, default=50, help="Top-k for probability extraction (reduce to save memory)")
    p.add_argument("--max_tokens", type=int, default=2)
    return p.parse_args()

def swap_judge_ref(file):
    """
    Swap judge and ref completions and metadata in the given file dict.
    Used when the dataset entry has judge/ref reversed, as they are arbitrary in diff.json
    """

    # Swap judge and ref in metadata
    ref = file['metadata']['ref']
    judge = file['metadata']['judge']
    file['metadata']['ref'] = judge
    file['metadata']['judge'] = ref
    temp = file['metadata']['num_right']
    file['metadata']['num_right'] = file['metadata']['num_wrong']
    file['metadata']['num_wrong'] = temp

    # Swap illegit and legit idxs
    legit_idx = file['legit_idx']
    file['legit_idx'] = [1 - i for i in file['illegit_idx']]
    file['illegit_idx'] = [1 - i for i in legit_idx]

    for i in range(len(file['data'])):
        row = file['data'][i]
        temp_completion = row['judge_completion']
        row['judge_completion'] = row['ref_completion']
        row['ref_completion'] = temp_completion
        temp_extracted_answer = row['judge_extracted_answer']
        row['judge_extracted_answer'] = row['ref_extracted_answer']
        row['ref_extracted_answer'] = temp_extracted_answer
        temp_score = row['judge_score']
        row['judge_score'] = row['ref_score']
        row['ref_score'] = temp_score

    return file


def main():
    args = parse_args()
    torch.set_float32_matmul_precision('high')
    
    # Set up output directory first (needed for logging)
    this_dir = Path(expand(os.path.join(args.output_path, args.judge.replace("/", "_"), args.ref.replace("/", "_"))))
    this_dir.mkdir(parents=True, exist_ok=True)
    
    # Archive existing metadata.json and output.jsonl if they exist
    metadata_file = this_dir / "metadata.json"
    output_file = this_dir / "output.jsonl"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if metadata_file.exists() or output_file.exists():
        old_dir = this_dir / "old"
        old_dir.mkdir(parents=True, exist_ok=True)
        
        if metadata_file.exists():
            archived_metadata = old_dir / f"metadata_{timestamp}.json"
            metadata_file.rename(archived_metadata)
        
        if output_file.exists():
            archived_output = old_dir / f"output_{timestamp}.jsonl"
            output_file.rename(archived_output)
    
    # Set up logging with comprehensive metadata
    logger = setup_logging(this_dir, args)
    
    logger.info("Starting data loading...")
    # Load dataset
    data = None
    all_diffs = open(expand(args.data_jsonl), "r", encoding="utf-8").readlines()
    for line in all_diffs:
        item = json.loads(line)
        if (item['metadata']['judge'] == args.judge and item['metadata']['ref'] == args.ref):
            data = item
            break
        elif (item['metadata']['judge'] == args.ref and item['metadata']['ref'] == args.judge):
            data = swap_judge_ref(item)
            break
    assert data is not None, "No matching judge/ref pair found in dataset."
    del all_diffs
    logger.info(f"Dataset loaded: {len(data['data'])} examples found")
    
    # Load HF tokenizer and model (following model_manager.py pattern)
    logger.info(f'Loading tokenizer and model from {args.judge}')
    tokenizer = AutoTokenizer.from_pretrained(
        args.judge,
        trust_remote_code=args.trust_remote_code,
        padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.debug(f"Set pad_token to eos_token: {tokenizer.eos_token}")
    
    logger.info("Loading model (this may take a while)...")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.judge,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=args.trust_remote_code,
        device_map="auto",
    )
    
   
    model.eval()
    device = model.device
    
    logger.info(f"Model loaded successfully on device: {device}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"Initial GPU memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
    
    # Set memory efficient settings
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = True
    
    # Clear cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"Initial GPU memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")

    with open(metadata_file, "w", encoding="utf-8") as f_meta:
        metadata = {
            "judge_model": args.judge,
            "ref_model": args.ref,
            "logprobs_k": args.logprobs_k,
            "max_tokens": args.max_tokens,
            "n_limited_examples": args.limit if args.limit != 0 else "None",
            "data_json_path": args.data_jsonl,
            "output_path": str(this_dir),
            "date_collected": str(datetime.now()),
            "float": "bfloat16" if args.bf16 else "float32",
            "collected_by": getpass.getuser(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        json.dump(metadata, f_meta, ensure_ascii=False, indent=2)
    logger.info(f"Metadata saved to {metadata_file}")

    logger.info(f"Starting processing, output will be written to {output_file}")
    processed = 0
    if args.limit != 0:
        queues = min(args.limit, len(data['data']))
    else:
        queues = len(data['data'])
    with open(output_file, "w", encoding="utf-8") as f_out:
        for i in range(queues):
            row = data['data'][i]
            question = (row.get("question") or "").strip()
            judge_comp = (row.get("judge_completion") or "").strip()
            ref_comp = (row.get("ref_completion") or "").strip()
            if not question or not judge_comp or not ref_comp:
                continue
            judge_score = row.get("judge_score", None)
            ref_score = row.get("ref_score", None)

            # Two orders
            msgs_jr = build_messages(question, judge_comp, ref_comp)  # A=judge, B=ref
            msgs_rj = build_messages(question, ref_comp, judge_comp)  # A=ref, B=judge

            # Log progress
            if i % 10 == 0:
                logger.info(f"Processing row {i}/{len(data['data'])}")
            
            logger.debug(f"========== Row {i} ==========")
            logger.debug(f"Built msgs_jr with {len(msgs_jr)} messages")
            for mi, m in enumerate(msgs_jr):
                logger.debug(f"msgs_jr[{mi}] role={m.get('role')} content={_truncate(m.get('content'), 300)}")
            
            # Process each order separately to save memory
            jr_text, jr_probs = None, None
            rj_text, rj_probs = None, None
            
            for order_idx, (msg_list, order_name) in enumerate([(msgs_jr, "JR"), (msgs_rj, "RJ")]):
                # Apply chat template
                prompt = tokenizer.apply_chat_template(
                    msg_list,
                    tokenize=False,
                    add_generation_prompt=True #TODO below, need to consider thinking/reasoning tokens in models, as that cna be annoying.
                ) + "Ok, I am not supposed to reason about the problem itself; only compare the provided answers.\n I prefer Assistant  "
                
                logger.debug(f"{order_name} prompt (truncated): {_truncate(prompt, 500)}")
                
                inputs = tokenizer(
                    [prompt],
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False
                )


                ############################################################################################# 
                ### This stuff was for annoying truncation problems that we will not deal with right now. ###
                #############################################################################################


                # actual_len = inputs["attention_mask"].sum().item()
                # print(f"[DEBUG] {order_name} actual prompt length before truncation: {actual_len}")
                
                # # Now tokenize with truncation if needed
                # inputs = tokenizer(
                #     [prompt],
                #     return_tensors="pt",
                #     padding=True,
                #     truncation=True,
                #     add_special_tokens=False
                # )


                # Get prompt length
                prompt_len = inputs["attention_mask"].sum().item()
                logger.debug(f"{order_name} prompt length: {prompt_len}")
                
                # Move to device and generate
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                
                with torch.no_grad():
                    gen_out = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        do_sample=False,
                        max_new_tokens=args.max_tokens,
                        return_dict_in_generate=True,
                        output_scores=True,
                        pad_token_id=tokenizer.pad_token_id,
                        use_cache=False
                    )
                
                sequences = gen_out.sequences  # (1, total_seq_len)
                scores = gen_out.scores  # tuple of (1, vocab_size) tensors
                
                logger.debug(f"{order_name}: Generated {len(scores)} steps")
                
                # Process the single example
                gen_tokens = sequences[0, prompt_len:].tolist()
                gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=False)
                
                logger.debug(f"{order_name}: generated text='{gen_text}'")
                
                # Find first A/B/T token
                step_idx = None
                for j, tid in enumerate(gen_tokens):
                    ttext = decode_token(tokenizer, tid)
                    logger.debug(f"{order_name} step {j}: token_id={tid}, text='{ttext}'")
                    if ttext.strip() in ABT_STRIPPED:
                        step_idx = j
                        logger.debug(f"{order_name}: Found A/B/T at step {step_idx}")
                        break
                
                probs_map = {"A": 0.0, "B": 0.0, "T": 0.0}
                if step_idx is not None and step_idx < len(scores):
                    # Get logits for this step
                    logits = scores[step_idx][0].cpu()  # (vocab_size,)
                    logger.debug(f"{order_name} step {step_idx}: extracting probs from logits")
                    
                    # Convert logits to probability map
                    text_prob_map = logits_to_prob_map(logits, tokenizer, topk=args.logprobs_k, logger=logger)
                    probs_map = only_abt_probs(text_prob_map, logger=logger)

                    del logits, text_prob_map
                else:
                    logger.warning(f"{order_name}: No A/B/T token found, using zero probs")
                
                # Store results
                if order_name == "JR":
                    jr_text, jr_probs = gen_text, probs_map
                else:
                    rj_text, rj_probs = gen_text, probs_map
                            
            # Normalize probabilities
            jr_norm = normalize_probs_abc(jr_probs)
            rj_norm = normalize_probs_abc(rj_probs)
            
            # Sanity check: probabilities should be distinct
            for p_1, p_2 in [('A', 'B'), ('A', 'T'), ('B', 'T')]:
                if rj_probs != {"A": 0.0, "B": 0.0, "T": 0.0}:
                    if np.isclose(rj_probs[p_1], rj_probs[p_2]):
                        logger.warning(f"RJ probs {p_1} and {p_2} are TOO CLOSE: {rj_probs}")
                if jr_probs != {"A": 0.0, "B": 0.0, "T": 0.0}:
                    if np.isclose(jr_probs[p_1], jr_probs[p_2]):
                        logger.warning(f"JR probs {p_1} and {p_2} are TOO CLOSE: {jr_probs}")

            # Combined (sum probabilities from both orders, then normalize to sum=1)
            swap = lambda x: "B" if x == "A" else ("A" if x == "B" else "T")
            combined_probs = {k: jr_probs[k] + rj_probs[swap(k)] for k in ["A", "B", "T"]}
            combined_norm = normalize_probs_abc(combined_probs)

            rec = {
                "row_index": int(i),
                "question": question,
                "judge_completion": judge_comp,
                "ref_completion": ref_comp,
                "judge_correct": judge_score,
                "ref_correct": ref_score,
                "model": args.judge,
                "outputs": {
                    "JR_order_text": jr_text,
                    "RJ_order_text": rj_text,
                },
                "probabilities": {
                    "JR_order": jr_probs,         # probs for A/B/T at the letter step
                    "RJ_order": rj_probs,
                    "combined": combined_probs,   # JR+RJ (not normalized)
                },
                "normalized": {
                    "JR_order": jr_norm,          # sums to 1
                    "RJ_order": rj_norm,          # sums to 1
                    "combined": combined_norm,    # sums to 1
                },
            }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f_out.flush()
            processed += 1
            
            logger.debug(f"Row {i} complete: JR={jr_probs}, RJ={rj_probs}, combined_norm={combined_norm}")
            
            del gen_out, sequences, scores, input_ids, attention_mask, inputs
            del jr_text, jr_probs, rj_text, rj_probs
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()



    logger.info("=" * 80)
    logger.info(f"RUN COMPLETED SUCCESSFULLY")
    logger.info(f"Processed {processed} rows")
    logger.info(f"Output saved to: {output_file}")
    logger.info(f"All results in: {this_dir}")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
