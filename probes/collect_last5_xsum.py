#!/usr/bin/env python3
"""
collect_last5_local_llama31_bf16safe.py
---------------------------------------
Local XSum-style loader + Llama 3.1 8B Instruct feature extractor
that ALWAYS outputs float32 arrays (no bfloat16 leaks).

What it does:
  • Reads a bias JSONL (must have: id, bias_type)
  • Loads article text from your local JSON store
  • Builds a prompt with {text}
  • Runs Llama 3.1 8B (float32) with output_hidden_states=True
  • Aggregates across layers (mean by default)
  • Takes the last 5 token positions (right-aligned; padded if <5)
  • Saves: x:(N,5,D) float32, y:(N,) int64, mask:(N,5) int64, ids:(N,)

Run:
  python collect_last5_local_llama31_bf16safe.py \
    --bias-jsonl path/to/bias.jsonl \
    --dataset xsum \
    --data-type sources \
    --prompt-template "Summarize the following article:\n{text}\nSummary:" \
    --out xsum_last5_llama31_local.npz
"""

import argparse
import json
from typing import List, Dict, Tuple
import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM

# ------------------------ Local data loaders ------------------------

def load_from_json(file_name: str) -> dict:
    with open(file_name, "r", encoding="utf-8") as f:
        return json.load(f)

def load_sources(dataset: str, extras: bool = False, data_type: str = "sources") -> Tuple[Dict[str, str], List[str]]:
    suffix = "_extra" if extras else ""
    path = f"{data_type}/{dataset}_train_{data_type}{suffix}.json"
    articles = load_from_json(path)
    keys = list(articles.keys())
    return articles, keys

def read_bias_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

# ------------------------ Prompt + encoding ------------------------

def make_prompt(template: str, text: str) -> str:
    return template.replace("{text}", text)

@torch.no_grad()
def encode_batch(
    model,
    tokenizer,
    prompts: List[str],
    device: torch.device,
    max_len: int,
    layer_agg: str = "mean",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      H: (B, T, D) float32 (forced)
      mask: (B, T) int64
    """
    tok = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    tok = {k: v.to(device) for k, v in tok.items()}
    out = model(**tok, output_hidden_states=True, use_cache=False)

    # hidden_states: tuple(L+1) of (B, T, D); idx 0 is embeddings
    hs = torch.stack(out.hidden_states[1:], dim=0)  # (L, B, T, D)

    if layer_agg == "mean":
        H = hs.mean(dim=0)      # (B, T, D)
    elif layer_agg == "last":
        H = hs[-1]              # (B, T, D)
    elif layer_agg == "concat":
        L, B, T, D = hs.shape
        H = hs.permute(1, 2, 3, 0).reshape(B, T, D * L)  # (B, T, L*D)
    else:
        raise ValueError("layer_agg must be mean|last|concat")

    # ---- HARD CAST to float32 here ----
    H = H.to(torch.float32)

    # Ensure mask is int64 (not bf16/bool)
    mask = tok["attention_mask"].to(torch.int64)
    return H, mask

def last_k(H: torch.Tensor, mask: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract last-k token positions (right-aligned).
    Returns:
      out: (B, k, D) float32
      out_mask: (B, k) int64
    """
    B, T, D = H.shape
    lengths = mask.sum(dim=1)  # (B,)
    out = torch.zeros(B, k, D, dtype=torch.float32, device=H.device)   # float32 sink
    out_mask = torch.zeros(B, k, dtype=torch.int64, device=mask.device)
    for i in range(B):
        L = int(lengths[i].item())
        take = min(k, L)
        if take > 0:
            h_i = H[i, L - take : L, :].to(torch.float32)  # enforce float32
            out[i, k - take :, :] = h_i                    # right-align
            out_mask[i, k - take :] = 1
    return out, out_mask

# ------------------------ Llama 3.1 loader (float32) ------------------------

def load_llama31_model(model_id: str, device: torch.device):
    """
    Load Meta-Llama-3.1 in float32 and set tokenizer pad token.
    Falls back to AutoModelForCausalLM if AutoModel isn't available.
    """
    trust = True
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust)
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=trust)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    try:
        mdl = AutoModel.from_pretrained(
            model_id,
            config=cfg,
            trust_remote_code=trust,
            dtype=torch.float32,   # use 'dtype', not deprecated 'torch_dtype'
        )
    except Exception:
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=cfg,
            trust_remote_code=trust,
            dtype=torch.float32,
        )

    mdl.to(device).eval()
    return tok, mdl

# ------------------------ Main ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bias-jsonl", required=True, help="JSONL with fields: id, bias_type")
    ap.add_argument("--dataset", required=True, help="Dataset name used in your local JSON files (e.g., xsum)")
    ap.add_argument("--data-type", default="sources", choices=["sources", "code"],
                    help="Top-level folder and filename suffix to load from (default: sources)")
    ap.add_argument("--extras", action="store_true", help="If set, reads *_extra.json variant")
    ap.add_argument("--out", required=True, help="Output .npz path")
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                    help="HF model id (default: Llama 3.1 8B Instruct)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--layer-agg", choices=["mean", "last", "concat"], default="mean")
    ap.add_argument("--prompt-template", required=True,
                    help="Template with {text} placeholder for the article text")
    ap.add_argument("--positive-bias-type", default="self_preference_bias",
                    help="Label=1 when bias_type equals this string")
    ap.add_argument("--limit", type=int, default=0, help="Process at most N rows (0 = all)")
    args = ap.parse_args()

    # harden defaults
    torch.set_float32_matmul_precision("high")  # optional; keeps fp32 compute
    device = torch.device(
        "cuda"
        if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()))
        else "cpu"
    )
    tokenizer, model = load_llama31_model(args.model, device)

    # Load local article dict
    articles, _ = load_sources(args.dataset, extras=args.extras, data_type=args.data_type)

    # Read bias JSONL
    rows = read_bias_jsonl(args.bias_jsonl)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    # Build prompts + labels
    prompts: List[str] = []
    labels: List[int] = []
    ids: List[str] = []
    missing = 0

    for r in rows:
        rid = str(r.get("id"))
        text = articles.get(rid) or articles.get(r.get("id"))
        if text is None:
            missing += 1
            continue
        prompts.append(make_prompt(args.prompt_template, text))
        labels.append(1 if r.get("bias_type") == args.positive_bias_type else 0)
        ids.append(rid)

    print(f"[INFO] Collected {len(prompts)} prompts from local store (missing: {missing})")

    # Encode in batches
    X_chunks, M_chunks = [], []
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i : i + args.batch_size]
        H, mask = encode_batch(model, tokenizer, batch, device, args.max_len, args.layer_agg)  # H is float32
        last5, last5_mask = last_k(H, mask, 5)  # last5 is float32
        X_chunks.append(last5.cpu())            # keep as float32 tensors on CPU
        M_chunks.append(last5_mask.cpu().to(torch.int64))

    if not X_chunks:
        raise SystemExit("[ERROR] No examples processed. Check paths, ids, and template.")

    # ---- FINAL hard cast during concatenation ----
    X = torch.cat([t.to(torch.float32) for t in X_chunks], dim=0).cpu().numpy()   # (N,5,D) float32
    M = torch.cat([m.to(torch.int64) for m in M_chunks], dim=0).cpu().numpy()     # (N,5) int64
    Y = np.array(labels, dtype="int64")
    np.savez(args.out, x=X, y=Y, mask=M, ids=np.array(ids, dtype=object))
    print(f"[OK] Saved {args.out} with x {X.shape}, y {Y.shape}, mask {M.shape}, ids {len(ids)}")


if __name__ == "__main__":
    main()
