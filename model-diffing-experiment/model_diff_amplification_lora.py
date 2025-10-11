#!/usr/bin/env python3
"""
Model Diff Amplification (MDA) sampler — with optional LoRA adapter for the post-trained model.

Core idea:
    logits_amplified = logits_after + alpha * (logits_after - logits_before)

New options:
- Provide a LoRA adapter path/ID for the AFTER model via --after_lora_adapter.
- Optionally merge the LoRA weights into the base model for faster inference via --merge_lora.

Requires:
    pip install torch transformers peft
"""

from dataclasses import dataclass
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import random


@dataclass
class MDAConfig:
    model_before: str
    model_after_base: str
    after_lora_adapter: Optional[str] = None
    merge_lora: bool = False
    alpha: float = 3.0
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 0
    max_new_tokens: int = 128
    device: Optional[str] = None
    seed: Optional[int] = 0
    eos_token_id: Optional[int] = None
    fp16: bool = True


class MDASampler:
    def __init__(self, cfg: MDAConfig):
        self.cfg = cfg

        if cfg.seed is not None:
            random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)
            torch.cuda.manual_seed_all(cfg.seed)

        if cfg.device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        else:
            device = cfg.device
        self.device = torch.device(device)

        dtype = torch.float16 if (cfg.fp16 and self.device.type == "cuda") else torch.float32

        self.tok_after = AutoTokenizer.from_pretrained(cfg.model_after_base, use_fast=True)
        self.tok_before = AutoTokenizer.from_pretrained(cfg.model_before, use_fast=True)

        self.model_before = AutoModelForCausalLM.from_pretrained(cfg.model_before, torch_dtype=dtype).to(self.device)
        self.model_before.eval()

        after_base = AutoModelForCausalLM.from_pretrained(cfg.model_after_base, torch_dtype=dtype).to(self.device)

        if cfg.after_lora_adapter:
            after_with_lora = PeftModel.from_pretrained(after_base, cfg.after_lora_adapter)
            if cfg.merge_lora:
                after_with_lora = after_with_lora.merge_and_unload()
            self.model_after = after_with_lora.to(self.device)
        else:
            self.model_after = after_base

        self.model_after.eval()

        self.eos_id = (
            cfg.eos_token_id if cfg.eos_token_id is not None
            else (self.tok_after.eos_token_id if self.tok_after.eos_token_id is not None else None)
        )

    @torch.no_grad()
    def _forward_logits(self, model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits[:, -1, :]

    @staticmethod
    def _top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
        scores = logits.clone()
        if top_k and top_k > 0:
            kth_vals = torch.topk(scores, k=min(top_k, scores.size(-1)))[0][..., -1, None]
            scores = torch.where(scores < kth_vals, torch.full_like(scores, float("-inf")), scores)
        if top_p < 1.0:
            sorted_scores, sorted_indices = torch.sort(scores, descending=True)
            probs = torch.softmax(sorted_scores, dim=-1)
            cumulative_probs = torch.cumsum(probs, dim=-1)
            cutoff = cumulative_probs > top_p
            cutoff[..., 1:] = cutoff[..., :-1].clone()
            cutoff[..., 0] = False
            sorted_scores = torch.where(cutoff, torch.full_like(sorted_scores, float("-inf")), sorted_scores)
            scores = torch.full_like(scores, float("-inf"))
            scores.scatter_(dim=-1, index=sorted_indices, src=sorted_scores)
        return scores

    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        cfg = self.cfg
        enc_after = self.tok_after(prompt, return_tensors="pt")
        input_ids = enc_after["input_ids"].to(self.device)
        attn = enc_after["attention_mask"].to(self.device)

        for _ in range(cfg.max_new_tokens):
            logits_after = self._forward_logits(self.model_after, input_ids, attn)

            current_text = self.tok_after.decode(input_ids[0], skip_special_tokens=False)
            enc_before = self.tok_before(current_text, return_tensors="pt")
            logits_before = self._forward_logits(
                self.model_before,
                enc_before["input_ids"].to(self.device),
                enc_before["attention_mask"].to(self.device),
            )

            logits_amplified = logits_after + cfg.alpha * (logits_after - logits_before)

            logits_scaled = logits_amplified / max(cfg.temperature, 1e-6)
            logits_filtered = self._top_k_top_p_filtering(logits_scaled, cfg.top_k, cfg.top_p)
            probs = torch.softmax(logits_filtered, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_id], dim=-1)
            attn = torch.ones_like(input_ids, device=self.device)

            if self.eos_id is not None and next_id.item() == self.eos_id:
                break

        return self.tok_after.decode(input_ids[0], skip_special_tokens=False)


def main():
    import argparse
    p = argparse.ArgumentParser(description="MDA with optional LoRA on AFTER model")
    p.add_argument("--model_before", type=str, required=True, help="HF id/path for BEFORE model (no LoRA)")
    p.add_argument("--model_after_base", type=str, required=True, help="HF id/path for AFTER base model")
    p.add_argument("--after_lora_adapter", type=str, default=None, help="HF id/path to AFTER LoRA adapter")
    p.add_argument("--merge_lora", action="store_true", help="Merge LoRA into base for faster inference")
    p.add_argument("--alpha", type=float, default=3.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-fp16", action="store_true", help="Disable float16")
    p.add_argument("--eos", type=int, default=None)
    p.add_argument("--prompt", type=str, required=True)
    args = p.parse_args()

    cfg = MDAConfig(
        model_before=args.model_before,
        model_after_base=args.model_after_base,
        after_lora_adapter=args.after_lora_adapter,
        merge_lora=args.merge_lora,
        alpha=args.alpha,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        seed=None if args.seed == -1 else args.seed,
        eos_token_id=args.eos,
        fp16=not args.no_fp16,
    )

    sampler = MDASampler(cfg)
    out = sampler.generate(args.prompt)
    print(out)


if __name__ == "__main__":
    main()
