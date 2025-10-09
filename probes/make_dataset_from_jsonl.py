
#!/usr/bin/env python3
"""
make_dataset_from_jsonl.py
--------------------------
Build (x, y, mask) arrays for probe training from a JSONL file with fields:
  {"prompt": "...", "label": 0 or 1}

It encodes prompts with a Hugging Face Transformer, collects hidden states,
aggregates across layers/tokens, and saves an NPZ: x, y, (optional) mask.

Examples
--------
python make_dataset_from_jsonl.py \

    --in data.jsonl --out latents.npz \

    --model gpt2 --layer-agg mean --token-agg last --max-len 512

Requirements
------------
pip install transformers torch
"""
import argparse, json, os, sys
from typing import List, Tuple, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

def load_jsonl(path: str) -> Tuple[List[str], np.ndarray]:
    prompts, labels = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            prompts.append(obj['prompt'])
            labels.append(int(obj['label']))
    return prompts, np.asarray(labels, dtype=np.int64)

@torch.no_grad()
def encode(
    model,
    tokenizer,
    prompts: List[str],
    device: torch.device,
    max_len: int,
    layer_agg: str = "mean",
    token_agg: str = "last",
    use_all_tokens: bool = False,
    batch_size: int = 16,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Returns:
      x: (N, D) if not use_all_tokens else (N, T, D) padded to max_T in batch with mask
      mask: (N, T) with 1 for real tokens, 0 for pad (only if use_all_tokens)
    """
    all_x, all_mask = [], []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i+batch_size]
        tok = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)
        out = model(**tok, output_hidden_states=True, use_cache=False)
        # hidden_states: tuple(L+1) of (B, T, D) including embeddings at 0
        hs = torch.stack(out.hidden_states[1:], dim=0)  # (L, B, T, D) exclude embeddings
        if layer_agg == "mean":
            h = hs.mean(dim=0)  # (B, T, D)
        elif layer_agg == "last":
            h = hs[-1]         # (B, T, D)
        elif layer_agg == "concat":
            B, T, D = hs.shape[1:]
            h = hs.permute(1,2,3,0).reshape(B, T, D*hs.shape[0])  # (B, T, L*D)
        else:
            raise ValueError("layer_agg must be mean|last|concat")

        attn_mask = tok['attention_mask']  # (B, T)
        lengths = attn_mask.sum(dim=1)

        if use_all_tokens:
            # keep all tokens, return mask too
            all_x.append(h.cpu())
            all_mask.append(attn_mask.cpu())
        else:
            if token_agg == "last":
                idx = (lengths - 1).clamp(min=0)  # position of last non-pad token
                gather = idx.view(-1,1,1).expand(-1,1,h.size(-1))
                pooled = h.gather(1, gather).squeeze(1)  # (B, D)
            elif token_agg == "mean":
                # mean over real tokens
                summed = (h * attn_mask.unsqueeze(-1)).sum(dim=1)
                pooled = summed / lengths.unsqueeze(-1)
            elif token_agg == "max":
                h_masked = h.masked_fill(attn_mask.unsqueeze(-1)==0, float('-inf'))
                pooled = h_masked.max(dim=1).values
            else:
                raise ValueError("token_agg must be last|mean|max")
            all_x.append(pooled.cpu())

    if use_all_tokens:
        # Pad sequences to the same T across the whole dataset
        xs = torch.nn.utils.rnn.pad_sequence([t for t in all_x], batch_first=True)  # stacks along batch already
        ms = torch.nn.utils.rnn.pad_sequence([m for m in all_mask], batch_first=True)
        x_np = xs.numpy()
        m_np = ms.numpy().astype(np.int64)
        return x_np, m_np
    else:
        x_np = torch.cat(all_x, dim=0).numpy()
        return x_np, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='in_path', required=True, help='Input JSONL with prompt,label')
    ap.add_argument('--out', dest='out_path', required=True, help='Output .npz path')
    ap.add_argument('--model', default='gpt2', help='HF model id (encoder or decoder)')
    ap.add_argument('--device', default='auto', choices=['auto','cpu','cuda'])
    ap.add_argument('--max-len', type=int, default=512)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--layer-agg', choices=['mean','last','concat'], default='mean',
                    help='How to aggregate across layers (paper used mean across all layers)')
    ap.add_argument('--token-agg', choices=['last','mean','max'], default='last',
                    help='How to pool tokens if not using all tokens (paper used last token of the prompt)')
    ap.add_argument('--use-all-tokens', action='store_true',
                    help='If set, returns (N,T,D) with a mask; otherwise (N,D) pooled')
    args = ap.parse_args()

    dev = torch.device('cuda' if (args.device == 'cuda' or (args.device=='auto' and torch.cuda.is_available())) else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(dev)
    prompts, labels = load_jsonl(args.in_path)

    x, mask = encode(model, tokenizer, prompts, dev, args.max_len, args.layer_agg, args.token_agg, args.use_all_tokens, args.batch_size)
    if mask is None:
        np.savez(args.out_path, x=x.astype('float32'), y=labels.astype('int64'))
    else:
        np.savez(args.out_path, x=x.astype('float32'), y=labels.astype('int64'), mask=mask.astype('int64'))
    print(f"Saved {args.out_path} with x shape {x.shape} and y shape {labels.shape}")

if __name__ == '__main__':
    main()
