
#!/usr/bin/env python3
"""
collect_last5_xsum_llama31.py
-----------------------------
Build (x, y, mask) arrays for probe training from XSum using IDs in a JSONL.

Requirements:
  pip install -U "transformers>=4.41" datasets torch accelerate safetensors sentencepiece

Example:
  python collect_last5_xsum_llama31.py \
    --bias-jsonl bias_data.jsonl \
    --split train \
    --prompt-template "Summarize the following article:\n{text}\nSummary:" \
    --out xsum_last5_llama31.npz
"""

import argparse, json
from typing import List, Tuple
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM


def read_bias_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_index_xsum(split: str):
    ds = load_dataset("xsum", split=split)
    return {ex["id"]: ex for ex in ds}


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
):
    """
    Returns:
      H: (B, T, D) aggregated across layers
      mask: (B, T) attention mask (1 real, 0 pad)
    """
    tok = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    # Move to device
    tok = {k: v.to(device) for k, v in tok.items()}
    out = model(**tok, output_hidden_states=True, use_cache=False)

    # hidden_states: tuple(L+1) of (B, T, D). index 0 is embeddings; drop it.
    hs = torch.stack(out.hidden_states[1:], dim=0)  # (L, B, T, D)

    if layer_agg == "mean":
        H = hs.mean(dim=0)       # (B, T, D)
    elif layer_agg == "last":
        H = hs[-1]               # (B, T, D)
    elif layer_agg == "concat":
        L, B, T, D = hs.shape
        H = hs.permute(1, 2, 3, 0).reshape(B, T, D * L)  # (B, T, L*D)
    else:
        raise ValueError("layer_agg must be one of: mean | last | concat")

    return H, tok["attention_mask"]


def last_k(H: torch.Tensor, mask: torch.Tensor, k: int):
    """
    Extract the last-k token positions (right-aligned).
    Returns:
      out: (B, k, D)
      out_mask: (B, k)  (1 where real, 0 if padded)
    """
    B, T, D = H.shape
    lengths = mask.sum(dim=1)  # (B,)
    out = torch.zeros(B, k, D, dtype=H.dtype, device=H.device)
    out_mask = torch.zeros(B, k, dtype=mask.dtype, device=mask.device)
    for i in range(B):
        L = int(lengths[i].item())
        take = min(k, L)
        if take > 0:
            h_i = H[i, L - take : L, :]        # (take, D)
            out[i, k - take :, :] = h_i        # right-align
            out_mask[i, k - take :] = 1
    return out, out_mask


def load_llama31_model(model_id: str, device: torch.device):
    """
    Robust loader for Llama 3.1 (Instruct). Falls back to AutoModelForCausalLM if
    the base backbone class isn't available in your transformers build.
    """
    trust = True
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust)
    tok = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, trust_remote_code=trust
    )
    # Llama models often lack an explicit pad; set pad to eos and use right padding.
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    try:
        mdl = AutoModel.from_pretrained(
            model_id,
            config=cfg,
            trust_remote_code=trust,
            torch_dtype="auto",
        )
    except Exception:
        # Some transformer versions don't expose the base class; use CausalLM head instead.
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=cfg,
            trust_remote_code=trust,
            torch_dtype="auto",
        )

    mdl.to(device).eval()
    return tok, mdl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bias-jsonl", required=True, help="JSONL with fields: id, bias_type")
    ap.add_argument(
        "--positive-bias-type",
        default="self_preference_bias",
        help="Label=1 when bias_type equals this string.",
    )
    ap.add_argument("--split", default="train", choices=["train", "validation", "test"])
    ap.add_argument("--out", required=True, help="Output .npz path")
    ap.add_argument(
        "--model",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="HF model id (default: Llama 3.1 8B Instruct).",
    )
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--layer-agg", choices=["mean", "last", "concat"], default="mean")
    ap.add_argument(
        "--prompt-template",
        required=True,
        help="Template with {text} placeholder for the XSum article.",
    )
    ap.add_argument("--limit", type=int, default=0, help="Process at most N rows (0 = all)")
    args = ap.parse_args()

    device = torch.device(
        "cuda"
        if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()))
        else "cpu"
    )

    tokenizer, model = load_llama31_model(args.model, device)

    index = build_index_xsum(args.split)
    rows = read_bias_jsonl(args.bias_jsonl)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    prompts, labels, ids = [], [], []
    missing = 0
    for r in rows:
        rid = str(r.get("id"))
        ex = index.get(rid)
        if ex is None:
            missing += 1
            continue
        prompt = make_prompt(args.prompt_template, ex["document"])
        prompts.append(prompt)
        labels.append(1 if r.get("bias_type") == args.positive_bias_type else 0)
        ids.append(rid)

    print(f"Collected {len(prompts)} prompts (missing in XSum: {missing})")

    X_chunks, M_chunks = [], []
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i : i + args.batch_size]
        H, mask = encode_batch(model, tokenizer, batch, device, args.max_len, args.layer_agg)
        last5, last5_mask = last_k(H, mask, 5)
        X_chunks.append(last5.cpu())
        M_chunks.append(last5_mask.cpu())

    if not X_chunks:
        raise SystemExit("No examples processed. Check inputs and filters.")

    X = torch.cat(X_chunks, dim=0).numpy().astype("float32")  # (N,5,D)
    M = torch.cat(M_chunks, dim=0).numpy().astype("int64")    # (N,5)
    Y = np.array(labels, dtype="int64")
    np.savez(args.out, x=X, y=Y, mask=M, ids=np.array(ids, dtype=object))
    print(f"Saved {args.out} with x {X.shape}, y {Y.shape}, mask {M.shape}, ids {len(ids)}")


if __name__ == "__main__":
    main()
