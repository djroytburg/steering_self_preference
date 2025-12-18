#!/usr/bin/env python3
"""
Adapted version of run_self_preference.py for Arena dataset.

Key differences:
- Reads from arena_diffs/*.jsonl files
- No need for swap_judge_ref since Arena data is already properly formatted
- Works with single-model datasets where judge is the model being tested
"""
from datetime import datetime
import argparse
import json
import logging
import os
import sys
import getpass
import socket
import platform
import math
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch
import torch.nn.functional as F

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
    logger = logging.getLogger("arena_self_preference")
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
    logger.info("ARENA SELF-PREFERENCE RUN STARTED")
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
        logger = logging.getLogger("arena_self_preference")
    
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
        logger = logging.getLogger("arena_self_preference")
    
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
    p = argparse.ArgumentParser(description="A/B/T probabilities for Arena data (single model self-preference).")
    p.add_argument("--output_path", type=str, required=True, help="Where to write JSONL output.")
    p.add_argument("--judge", type=str, required=True, help="Model to test (e.g., gemma-3-27b-it)")
    p.add_argument("--data_jsonl", type=str, required=True,
                   help="Path to the Arena JSONL file for this model (from arena_diffs/)")
    p.add_argument("--limit", type=int, default=0, help="Max rows; 0 = all.")
    # Model
    p.add_argument("--bf16", action="store_true", help="Use bfloat16 precision")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--lora_adapter", type=str, default=None, help="Path to LoRA adapter to load on top of the judge model")
    # Decoding
    p.add_argument("--logprobs_k", type=int, default=50, help="Top-k for probability extraction")
    p.add_argument("--max_tokens", type=int, default=2)
    return p.parse_args()

def main():
    args = parse_args()
    torch.set_float32_matmul_precision('high')
    
    # Extract model name for output directory
    model_name = args.judge.replace("/", "_")
    
    # Set up output directory
    this_dir = Path(expand(os.path.join(args.output_path, model_name, "arena")))
    this_dir.mkdir(parents=True, exist_ok=True)
    
    # Archive existing files if they exist
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
    
    # Set up logging
    logger = setup_logging(this_dir, args)
    
    logger.info("Starting data loading...")
    # Load Arena dataset for this model
    data = []
    with open(expand(args.data_jsonl), "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    
    logger.info(f"Dataset loaded: {len(data)} examples found")
    
    # Load HF tokenizer and model
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

    # Load LoRA adapter if provided
    if args.lora_adapter is not None:
        logger.info(f"Loading LoRA adapter from {args.lora_adapter}...")
        model = PeftModel.from_pretrained(model, args.lora_adapter)
        logger.info("LoRA adapter loaded successfully")

    model.eval()
    device = model.device

    logger.info(f"Model loaded successfully on device: {device}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"Initial GPU memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
    
    # Set memory efficient settings
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = True
    
    # Write metadata
    with open(metadata_file, "w", encoding="utf-8") as f_meta:
        metadata = {
            "judge_model": args.judge,
            "lora_adapter": args.lora_adapter if args.lora_adapter is not None else "None",
            "data_source": "arena",
            "logprobs_k": args.logprobs_k,
            "max_tokens": args.max_tokens,
            "n_limited_examples": args.limit if args.limit != 0 else "None",
            "data_jsonl_path": args.data_jsonl,
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
    queues = min(args.limit, len(data)) if args.limit != 0 else len(data)
    
    with open(output_file, "w", encoding="utf-8") as f_out:
        for i in range(queues):
            row = data[i]
            question = (row.get("question") or "").strip()
            judge_comp = (row.get("judge_completion") or "").strip()
            ref_comp = (row.get("ref_completion") or "").strip()
            if not question or not judge_comp or not ref_comp:
                continue
            judge_score = row.get("judge_correct", None)
            ref_score = row.get("ref_correct", None)
            opponent = row.get("opponent", "unknown")

            # Two orders: JR = judge is A, ref is B; RJ = ref is A, judge is B
            msgs_jr = build_messages(question, judge_comp, ref_comp)
            msgs_rj = build_messages(question, ref_comp, judge_comp)

            # Log progress
            if i % 10 == 0:
                logger.info(f"Processing row {i}/{queues}")
            
            logger.debug(f"========== Row {i} ==========")
            logger.debug(f"Opponent: {opponent}, Judge won: {judge_score}")
            
            # Process each order separately to save memory
            jr_text, jr_probs = None, None
            rj_text, rj_probs = None, None
            
            for order_idx, (msg_list, order_name) in enumerate([(msgs_jr, "JR"), (msgs_rj, "RJ")]):
                # Apply chat template
                prompt = tokenizer.apply_chat_template(
                    msg_list,
                    tokenize=False,
                    add_generation_prompt=True
                ) + "Ok, I am not supposed to reason about the problem itself; only compare the provided answers.\n I prefer Assistant  "
                
                logger.debug(f"{order_name} prompt (truncated): {_truncate(prompt, 500)}")
                
                inputs = tokenizer(
                    [prompt],
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False
                )

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
                        max_new_tokens=args.max_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                
                sequences = gen_out.sequences
                scores = gen_out.scores
                
                logger.debug(f"{order_name}: Generated {len(scores)} steps")
                
                # Process the single example
                gen_tokens = sequences[0, prompt_len:].tolist()
                gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=False)
                
                logger.debug(f"{order_name}: generated text='{gen_text}'")
                
                # Find first A/B/T token
                step_idx = None
                for j, tid in enumerate(gen_tokens):
                    tok_str = decode_token(tokenizer, tid).strip()
                    if tok_str in ABT_STRIPPED:
                        step_idx = j
                        logger.debug(f"{order_name}: Found A/B/T at step {j}, token={tok_str}")
                        break
                
                probs_map = {"A": 0.0, "B": 0.0, "T": 0.0}
                if step_idx is not None and step_idx < len(scores):
                    logits_tensor = scores[step_idx][0]
                    text_probs = logits_to_prob_map(logits_tensor, tokenizer, topk=args.logprobs_k, logger=logger)
                    probs_map = only_abt_probs(text_probs, logger=logger)
                    logger.debug(f"{order_name}: step={step_idx}, probs={probs_map}")
                else:
                    logger.warning(f"{order_name}: No A/B/T found in first {args.max_tokens} tokens")
                
                # Store results
                if order_name == "JR":
                    jr_text = gen_text
                    jr_probs = probs_map
                else:
                    rj_text = gen_text
                    rj_probs = probs_map
                
                # Clean up to save memory
                del gen_out, sequences, scores, input_ids, attention_mask, inputs
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                            
            # Normalize probabilities
            jr_norm = normalize_probs_abc(jr_probs)
            rj_norm = normalize_probs_abc(rj_probs)
            
            # Combined (sum probabilities from both orders, then normalize)
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
                "opponent": opponent,
                "model": args.judge,
                "outputs": {
                    "JR_order_text": jr_text,
                    "RJ_order_text": rj_text,
                },
                "probabilities": {
                    "JR_order": jr_probs,
                    "RJ_order": rj_probs,
                    "combined": combined_probs,
                },
                "normalized": {
                    "JR_order": jr_norm,
                    "RJ_order": rj_norm,
                    "combined": combined_norm,
                },
            }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f_out.flush()
            processed += 1
            
            logger.debug(f"Row {i} complete: JR={jr_probs}, RJ={rj_probs}, combined_norm={combined_norm}")

    logger.info("=" * 80)
    logger.info(f"RUN COMPLETED SUCCESSFULLY")
    logger.info(f"Processed {processed} rows")
    logger.info(f"Output saved to: {output_file}")
    logger.info(f"All results in: {this_dir}")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
