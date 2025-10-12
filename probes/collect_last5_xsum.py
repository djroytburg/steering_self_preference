#!/usr/bin/env python3
"""
collect_last5_local_llama31_multi.py
------------------------------------
Local XSum-style loader + Llama 3.1 8B Instruct feature extractor that
supports MULTIPLE input datasets:

  • --pos-jsonl  path  (repeatable): rows are treated as y=1 (self-preference)
  • --neg-jsonl  path  (repeatable): rows are treated as y=0 (non self-preference)
  • --bias-jsonl path  (optional): legacy file with a 'bias_type' field
      - rows are treated as positive iff bias_type == --positive-bias-type

All JSONLs must have an 'id' field corresponding to keys in your local store.

Dedup policy if the same id appears in multiple sources:
  --dedup pos_wins | neg_wins | drop_conflicts   (default: pos_wins)

Outputs:
  x: (N, 5, D) float32
  y: (N,)      int64
  mask: (N, 5) int64
  ids: (N,)    object

Example:
  python collect_last5_local_llama31_multi.py \
    --dataset xsum --data-type sources \
    --pos-jsonl sp_train.jsonl \
    --neg-jsonl nonsp_train.jsonl \
    --prompt-template "Summarize the following article:\\n{text}\\nSummary:" \
    --out xsum_last5_llama31_multi.npz

Requirements:
  pip install -U "transformers>=4.41" torch accelerate safetensors sentencepiece
"""

import argparse
import json
from typing import List, Dict, Tuple, Iterable
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


# ------------------------ JSONL readers ------------------------

def read_ids_from_jsonl(path: str, id_field: str = "id") -> List[str]:
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if id_field not in obj:
                continue
            ids.append(str(obj[id_field]))
    return ids

def read_pos_from_bias_jsonl(path: str, positive_bias_type: str = "self_preference_bias",
                             id_field: str = "id", bias_field: str = "bias_type") -> List[str]:
    pos_ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if id_field in obj and obj.get(bias_field) == positive_bias_type:
                pos_ids.append(str(obj[id_field]))
    return pos_ids


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

    # ---- HARD CAST to float32 ----
    H = H.to(torch.float32)

    # Ensure mask is int64
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
    out = torch.zeros(B, k, D, dtype=torch.float32, device=H.device)
    out_mask = torch.zeros(B, k, dtype=torch.int64, device=mask.device)
    for i in range(B):
        L = int(lengths[i].item())
        take = min(k, L)
        if take > 0:
            h_i = H[i, L - take : L, :].to(torch.float32)
            out[i, k - take :, :] = h_i
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
            dtype=torch.float32,   # use 'dtype' (not deprecated 'torch_dtype')
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


# ------------------------ Utilities ------------------------

def _add_with_label(id_list: Iterable[str], label: int, dest: Dict[str, int], dedup: str):
    """
    Merge ids into dest dict with a label, respecting dedup policy:
      - pos_wins:  if conflicts, label 1 overrides 0
      - neg_wins:  if conflicts, label 0 overrides 1
      - drop_conflicts: remove ids that appear with opposite labels
    """
    for _id in id_list:
        if _id not in dest:
            dest[_id] = label
        else:
            if dest[_id] == label:
                continue
            # conflict
            if dedup == "pos_wins":
                dest[_id] = max(dest[_id], label)
            elif dedup == "neg_wins":
                dest[_id] = min(dest[_id], label)
            elif dedup == "drop_conflicts":
                dest.pop(_id, None)
            else:
                raise ValueError(f"Unknown dedup policy: {dedup}")


# ------------------------ Main ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Dataset name in your local store (e.g., xsum)")
    ap.add_argument("--data-type", default="sources", choices=["sources", "code"],
                    help="Folder/suffix used by your local JSON files (default: sources)")
    ap.add_argument("--extras", action="store_true", help="If set, reads *_extra.json variant")

    # Sources: multiple JSONLs for positives / negatives
    ap.add_argument("--pos-jsonl", action="append", default=[],
                    help="JSONL of positive (self-preference) examples. Repeatable.")
    ap.add_argument("--neg-jsonl", action="append", default=[],
                    help="JSONL of negative (non self-preference) examples. Repeatable.")

    # Legacy bias-jsonl (optional)
    ap.add_argument("--bias-jsonl", default=None,
                    help="Legacy bias JSONL with 'bias_type' field; positives picked via --positive-bias-type")
    ap.add_argument("--positive-bias-type", default="self_preference_bias",
                    help="Label=1 when bias_type equals this string (for --bias-jsonl)")

    ap.add_argument("--out", required=True, help="Output .npz path")
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                    help="HF model id (default: Llama 3.1 8B Instruct)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--layer-agg", choices=["mean", "last", "concat"], default="mean")
    ap.add_argument("--prompt-template", required=True,
                    help="Template with {text} placeholder for the article text")
    ap.add_argument("--dedup", choices=["pos_wins", "neg_wins", "drop_conflicts"], default="pos_wins",
                    help="How to resolve id conflicts across sources (default: pos_wins)")
    ap.add_argument("--limit", type=int, default=0, help="Process at most N rows after merging (0 = all)")
    args = ap.parse_args()

    # Device
    torch.set_float32_matmul_precision("high")
    device = torch.device(
        "cuda"
        if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()))
        else "cpu"
    )
    tokenizer, model = load_llama31_model(args.model, device)

    # Load local article dict
    articles, _ = load_sources(args.dataset, extras=args.extras, data_type=args.data_type)

    # Build merged id->label map
    id2label: Dict[str, int] = {}

    # Positives from explicit files
    for p in args.pos_jsonl:
        pos_ids = read_ids_from_jsonl(p, id_field="id")
        _add_with_label(pos_ids, 1, id2label, dedup=args.dedup)

    # Negatives from explicit files
    for n in args.neg_jsonl:
        neg_ids = read_ids_from_jsonl(n, id_field="id")
        _add_with_label(neg_ids, 0, id2label, dedup=args.dedup)

    # Legacy bias-jsonl (optional)
    if args.bias_jsonl:
        pos_from_bias = read_pos_from_bias_jsonl(args.bias_jsonl, positive_bias_type=args.positive_bias_type)
        _add_with_label(pos_from_bias, 1, id2label, dedup=args.dedup)

    # Turn map into ordered lists (stable order by insertion)
    merged_items = list(id2label.items())
    if args.limit and args.limit > 0:
        merged_items = merged_items[: args.limit]

    # Build prompts + labels
    prompts: List[str] = []
    labels: List[int] = []
    ids: List[str] = []
    missing = 0

    for rid, lab in merged_items:
        text = articles.get(rid)
        if text is None:
            # try non-str key just in case
            text = articles.get(rid if not rid.isdigit() else int(rid))
        if text is None:
            missing += 1
            continue
        prompts.append(make_prompt(args.prompt_template, text))
        labels.append(int(lab))
        ids.append(str(rid))

    print(f"[INFO] Prepared {len(prompts)} prompts from local store (missing articles: {missing})")

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

    # Final concatenation and save
    X = torch.cat([t.to(torch.float32) for t in X_chunks], dim=0).cpu().numpy()   # (N,5,D)
    M = torch.cat([m.to(torch.int64) for m in M_chunks], dim=0).cpu().numpy()     # (N,5)
    Y = np.array(labels, dtype="int64")
    np.savez(args.out, x=X, y=Y, mask=M, ids=np.array(ids, dtype=object))
    print(f"[OK] Saved {args.out} with x {X.shape}, y {Y.shape}, mask {M.shape}, ids {len(ids)}")


if __name__ == "__main__":
    main()
