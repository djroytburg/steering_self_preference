#!/usr/bin/env python3
import os
import torch
import torch.nn.functional as F
import argparse
import json
import math
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Tuple
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()

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
        "<The End of Assistant B's Answer>"
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

def logits_to_prob_map(logits_tensor, tokenizer, topk: int = 200) -> Dict[str, float]:
    """
    Convert logits tensor to {decoded_token_text -> probability}.
    Uses log-softmax and takes top-k tokens.
    """
    text_to_prob: Dict[str, float] = {}
    
    print(f"[DEBUG] logits_to_prob_map: input shape={logits_tensor.shape}, dtype={logits_tensor.dtype}")
    
    # Compute log probabilities
    log_probs = F.log_softmax(logits_tensor, dim=-1)
    
    # Get top-k indices
    k = min(topk, logits_tensor.shape[-1])
    topk_logprobs, topk_indices = torch.topk(log_probs, k, dim=-1)
    
    # Convert to CPU and numpy for processing
    topk_logprobs_np = topk_logprobs.detach().cpu().numpy()
    topk_indices_np = topk_indices.detach().cpu().numpy()
    
    print(f"[DEBUG] Top-k extraction: k={k}, top logprobs range=[{topk_logprobs_np.min():.4f}, {topk_logprobs_np.max():.4f}]")
    
    for i, (token_id, logprob) in enumerate(zip(topk_indices_np, topk_logprobs_np)):
        token_id = int(token_id)
        logprob = float(logprob)
        prob = math.exp(logprob)
        ttext = decode_token(tokenizer, token_id)
        text_to_prob[str(ttext)] = prob
        
        if i < 10:  # Print first 10 for debugging
            print(f"[DEBUG] topk[{i}]: token_id={token_id}, text='{_truncate(ttext, 40)}', logprob={logprob:.4f}, prob={prob:.6f}")
    
    return text_to_prob

def only_abt_probs(text_prob_map: Dict[str, float]) -> Dict[str, float]:
    """Collect probabilities for A/B/T tokens (and common whitespace-prefixed variants)."""
    aliases = {
        "A": {"A", "▁A", " A", "\nA", "\tA"},
        "B": {"B", "▁B", " B", "\nB", "\tB"},
        "T": {"T", "▁T", " T", "\nT", "\tT"},
    }
    out = {"A": 0.0, "B": 0.0, "T": 0.0}
    for letter, alset in aliases.items():
        out[letter] = max((text_prob_map.get(a, 0.0) for a in alset), default=0.0)
    
    print(f"[DEBUG] only_abt_probs result: {out}")
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
    p = argparse.ArgumentParser(description="A/B/T probabilities from Llama-3.1-8B-Instruct for two prompt orders.")
    p.add_argument("--output_path", type=str, required=True, help="Where to write JSONL output.")
    p.add_argument("--data_jsonl", type=str,
                   default="/home/ubuntu/dani/data_explorations/llama_gpt_math_comparison.jsonl",
                   help="JSONL fields: question, judge_completion, ref_completion; optional: judge_correct, ref_correct.")
    p.add_argument("--limit", type=int, default=0, help="Max rows; 0 = all.")
    # Model
    p.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--bf16", action="store_true", help="Use bfloat16 precision")
    p.add_argument("--trust_remote_code", action="store_true")
    # Decoding
    p.add_argument("--logprobs_k", type=int, default=50, help="Top-k for probability extraction (reduce to save memory)")
    p.add_argument("--max_tokens", type=int, default=2)
    return p.parse_args()

def main():
    args = parse_args()
    torch.set_float32_matmul_precision('high')
    
    # Load dataset
    ds = load_dataset("json", data_files=expand(args.data_jsonl))["train"]
    if args.limit and args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    # Load HF tokenizer and model (following model_manager.py pattern)
    print(f'Loading tokenizer and model from {args.base_model}')
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
        padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=args.trust_remote_code,
        device_map="auto",
    )
    
    # Enable gradient checkpointing to save memory
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    model.eval()
    device = model.device
    
    print(f"Model loaded on device: {device}")
    
    # Clear cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")

    out_path = Path(expand(args.output_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    with out_path.open("w", encoding="utf-8") as f_out:
        for i in range(len(ds)):
            row = ds[i]
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

            # Debug: print built messages with truncated contents
            print(f"\n[DEBUG] ========== Row {i} ==========")
            print(f"[DEBUG] Built msgs_jr with {len(msgs_jr)} messages")
            for mi, m in enumerate(msgs_jr):
                print(f"[DEBUG] msgs_jr[{mi}] role={m.get('role')} content={_truncate(m.get('content'), 300)}")
            
            # Process each order separately to save memory
            jr_text, jr_probs = None, None
            rj_text, rj_probs = None, None
            
            for order_idx, (msg_list, order_name) in enumerate([(msgs_jr, "JR"), (msgs_rj, "RJ")]):
                # Apply chat template
                prompt = tokenizer.apply_chat_template(
                    msg_list,
                    tokenize=False,
                    add_generation_prompt=True
                ) + "<think>\n Ok, I am not supposed to reason about the problem itself; only compare the provided answers.</think>\n I prefer Assistant  "
                
                print(f"[DEBUG] {order_name} prompt (truncated): {prompt}")
                
                # Tokenize to check length first (without truncation)
                temp_inputs = tokenizer(
                    [prompt],
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False
                )
                # actual_len = temp_inputs["attention_mask"].sum().item()
                # print(f"[DEBUG] {order_name} actual prompt length before truncation: {actual_len}")
                
                # # Now tokenize with truncation if needed
                # inputs = tokenizer(
                #     [prompt],
                #     return_tensors="pt",
                #     padding=True,
                #     truncation=True,
                #     add_special_tokens=False
                # )
                inputs = temp_inputs
                # Get prompt length
                prompt_len = inputs["attention_mask"].sum().item()
                print(f"[DEBUG] {order_name} prompt length: {prompt_len}")
                
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
                    )
                
                sequences = gen_out.sequences  # (1, total_seq_len)
                scores = gen_out.scores  # tuple of (1, vocab_size) tensors
                
                print(f"[DEBUG] {order_name}: Generated {len(scores)} steps")
                
                # Process the single example
                gen_tokens = sequences[0, prompt_len:].tolist()
                gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=False)
                
                print(f"[DEBUG] {order_name}: generated text='{gen_text}'")
                
                # Find first A/B/T token
                step_idx = None
                for j, tid in enumerate(gen_tokens):
                    ttext = decode_token(tokenizer, tid)
                    print(f"[DEBUG] {order_name} step {j}: token_id={tid}, text='{ttext}'")
                    if ttext.strip() in ABT_STRIPPED:
                        step_idx = j
                        print(f"[DEBUG] {order_name}: Found A/B/T at step {step_idx}")
                        break
                
                probs_map = {"A": 0.0, "B": 0.0, "T": 0.0}
                if step_idx is not None and step_idx < len(scores):
                    # Get logits for this step
                    logits = scores[step_idx][0]  # (vocab_size,)
                    print(f"[DEBUG] {order_name} step {step_idx}: extracting probs from logits")
                    
                    # Convert logits to probability map
                    text_prob_map = logits_to_prob_map(logits, tokenizer, topk=args.logprobs_k)
                    probs_map = only_abt_probs(text_prob_map)
                else:
                    print(f"[DEBUG] {order_name}: No A/B/T token found, using zero probs")
                
                # Store results
                if order_name == "JR":
                    jr_text, jr_probs = gen_text, probs_map
                else:
                    rj_text, rj_probs = gen_text, probs_map
                
                # Clean up to free memory
                del gen_out, sequences, scores, input_ids, attention_mask, inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Normalize probabilities
            # Normalize probabilities
            jr_norm = normalize_probs_abc(jr_probs)
            
            rj_norm = normalize_probs_abc(rj_probs)
            
            # Sanity check: probabilities should be distinct
            for p_1, p_2 in [('A', 'B'), ('A', 'T'), ('B', 'T')]:
                if rj_probs != {"A": 0.0, "B": 0.0, "T": 0.0}:
                    if np.isclose(rj_probs[p_1], rj_probs[p_2]):
                        print(f"[WARNING] RJ probs {p_1} and {p_2} are TOO CLOSE: {rj_probs}")
                if jr_probs != {"A": 0.0, "B": 0.0, "T": 0.0}:
                    if np.isclose(jr_probs[p_1], jr_probs[p_2]):
                        print(f"[WARNING] JR probs {p_1} and {p_2} are TOO CLOSE: {jr_probs}")

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
                "model": args.base_model,
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
            
            print(f"[DEBUG] Row {i} complete: JR={jr_probs}, RJ={rj_probs}, combined_norm={combined_norm}")

    print(f"\n✓ Processed {processed} rows → {out_path}")

if __name__ == "__main__":
    main()
