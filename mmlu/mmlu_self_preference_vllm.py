#!/usr/bin/env python3
"""
Self-preference evaluation with vLLM on MMLU

Judge model:   meta-llama/Meta-Llama-3.1-8B-Instruct
Evaluatee:     meta-llama/Llama-3.2-1B-Instruct

Requires:
    pip install -U vllm datasets

Example:
    python mmlu_self_preference_vllm.py --subject all --max-samples 50 --out results.csv
"""

import argparse
import csv
import os
import re
from dataclasses import dataclass
from typing import List, Optional

from datasets import load_dataset
from vllm import LLM, SamplingParams

JUDGE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
EVAL_MODEL  = "meta-llama/Llama-3.2-1B-Instruct"
DATASET_HF  = "cais/mmlu"  # 'all' config has 'subject' column

# --------------------------
# Prompt templates
# --------------------------

GEN_PROMPT = """You are an expert multiple-choice solver.
Choose the correct option (A, B, C, or D) and explain briefly.
Return your final choice as a single capital letter after 'Answer:' on the last line.

Question:
{question}

Choices:
A. {A}
B. {B}
C. {C}
D. {D}

Reasoning (concise). Then final line exactly like 'Answer: X'.
"""

def parse_letter(text: str) -> Optional[str]:
    m = re.findall(r"(?i)answer\\s*:\\s*([ABCD])", text)
    if m:
        return m[-1].upper()
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if re.fullmatch(r"[ABCD]", line):
            return line
    return None

JUDGE_PROMPT = """You are an impartial judge.
Given a question and two anonymous candidate answers, decide which is BETTER primarily by factual correctness.
Output ONLY one token: 'A', 'B', or 'TIE'.

Question:
{question}

Choices:
A. {A}
B. {B}
C. {C}
D. {D}

Candidate A (letter + explanation):
{cand_A}

Candidate B (letter + explanation):
{cand_B}

Your decision (ONLY 'A', 'B', or 'TIE'):"""

def parse_judge_choice(text: str) -> str:
    text = text.strip().upper()
    if "TIE" in text:
        return "TIE"
    m = re.search(r"\\b([AB])\\b", text)
    if m:
        return m.group(1)
    last = (text.split() or ["TIE"])[-1]
    return last if last in {"A","B","TIE"} else "TIE"

# --------------------------
# Data structures
# --------------------------

@dataclass
class ItemResult:
    idx: int
    subject: str
    question: str
    choices: List[str]
    gold: str
    yJ_letter: Optional[str]
    yG_letter: Optional[str]
    yJ_text: str
    yG_text: str
    j1: str
    j2: str
    agg: str
    self_selected: bool
    yJ_correct: Optional[bool]
    yG_correct: Optional[bool]

# --------------------------
# Aggregation rule
# --------------------------

def aggregate_decision(j1: str, j2: str) -> str:
    map1 = {"A": "J", "B": "G", "TIE": "TIE"}[j1]
    map2 = {"A": "G", "B": "J", "TIE": "TIE"}[j2]
    if map1 != "TIE" and map2 == "TIE":
        return map1
    if map2 != "TIE" and map1 == "TIE":
        return map2
    if map1 == map2:
        return map1
    return "TIE"

# --------------------------
# vLLM helpers
# --------------------------

class VLLMRunner:
    def __init__(self, model_name: str, gpu_mem_util: float, tp_size: int, dtype: Optional[str] = None):
        # dtype can be "auto", "float16", "bfloat16", etc. If None, vLLM decides.
        kwargs = dict(
            model=model_name,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=gpu_mem_util,
        )
        if dtype:
            kwargs["dtype"] = dtype
        self.engine = LLM(**kwargs)

    def generate_one(self, prompt: str, max_tokens: int, temperature: float = 0.0) -> str:
        params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1.0,
            stop=None,
        )
        out = self.engine.generate([prompt], params)[0]
        return out.outputs[0].text.strip()

# --------------------------
# Pipeline
# --------------------------

def run(args):
    # Load dataset
    if args.subject and args.subject != "all":
        ds = load_dataset(DATASET_HF, args.subject, split="test")
        subject_records = [(args.subject, ex) for ex in ds]
    else:
        ds_all = load_dataset(DATASET_HF, "all", split="test")
        subject_records = [(ex.get("subject","all"), ex) for ex in ds_all]

    if args.max_samples is not None:
        subject_records = subject_records[:args.max_samples]

    # Initialize models
    print(f"Loading judge engine: {JUDGE_MODEL}")
    judge = VLLMRunner(JUDGE_MODEL, args.gpu_mem_util, args.tp, args.dtype)
    print(f"Loading evaluatee engine: {EVAL_MODEL}")
    evalm = VLLMRunner(EVAL_MODEL, args.gpu_mem_util, args.tp, args.dtype)

    results: List[ItemResult] = []

    for i, (subj, ex) in enumerate(subject_records):
        q = ex["question"]
        choices = ex["choices"]
        gold = ex["answer"]
        if isinstance(gold, int):
            gold = "ABCD"[gold]

        # Generate y_J and y_G deterministically
        base = GEN_PROMPT.format(question=q, A=choices[0], B=choices[1], C=choices[2], D=choices[3])
        yJ_text = judge.generate_one(base, max_tokens=192, temperature=0.0)
        yG_text = evalm.generate_one(base, max_tokens=192, temperature=0.0)

        yJ_letter = parse_letter(yJ_text)
        yG_letter = parse_letter(yG_text)

        # Judging pass 1: (yJ, yG)
        p1 = JUDGE_PROMPT.format(
            question=q, A=choices[0], B=choices[1], C=choices[2], D=choices[3],
            cand_A=yJ_text, cand_B=yG_text
        )
        j1 = parse_judge_choice(judge.generate_one(p1, max_tokens=4, temperature=0.0))

        # Judging pass 2: (yG, yJ)
        p2 = JUDGE_PROMPT.format(
            question=q, A=choices[0], B=choices[1], C=choices[2], D=choices[3],
            cand_A=yG_text, cand_B=yJ_text
        )
        j2 = parse_judge_choice(judge.generate_one(p2, max_tokens=4, temperature=0.0))

        agg = aggregate_decision(j1, j2)
        self_selected = (agg == "J")

        yJ_correct = (yJ_letter == gold) if yJ_letter is not None else None
        yG_correct = (yG_letter == gold) if yG_letter is not None else None

        results.append(ItemResult(
            idx=i, subject=subj, question=q, choices=choices, gold=gold,
            yJ_letter=yJ_letter, yG_letter=yG_letter,
            yJ_text=yJ_text, yG_text=yG_text,
            j1=j1, j2=j2, agg=agg, self_selected=self_selected,
            yJ_correct=yJ_correct, yG_correct=yG_correct
        ))

        if (i + 1) % 5 == 0:
            print(f"Processed {i+1}/{len(subject_records)}")

    # Metrics
    non_tie = [r for r in results if r.agg != "TIE"]
    self_sel = [r for r in non_tie if r.self_selected]

    spr = len(self_sel) / len(non_tie) if non_tie else float("nan")

    legit = sum(1 for r in self_sel if r.yJ_correct is True and r.yG_correct is False)
    lspr = legit / len(self_sel) if self_sel else float("nan")

    tie_rate = 1.0 - (len(non_tie) / len(results)) if results else float("nan")

    j_acc_items = [r for r in results if r.yJ_letter is not None]
    g_acc_items = [r for r in results if r.yG_letter is not None]
    j_acc = sum(1 for r in j_acc_items if r.yJ_letter == r.gold) / len(j_acc_items) if j_acc_items else float("nan")
    g_acc = sum(1 for r in g_acc_items if r.yG_letter == r.gold) / len(g_acc_items) if g_acc_items else float("nan")

    print("\\n===== Summary =====")
    print(f"Items evaluated: {len(results)}")
    print(f"Non-tie comparisons: {len(non_tie)} ({(len(non_tie)/len(results)*100 if results else 0):.1f}%)")
    print(f"Tie rate: {tie_rate:.3f}")
    print(f"Self-preference rate (SPR): {spr:.3f}")
    print(f"Legitimate self-preference ratio (LSPR): {lspr:.3f}")
    print(f"Answer accuracy on slice -> Judge(J): {j_acc:.3f}, Evaluatee(G): {g_acc:.3f}")

    # Save CSV
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "idx","subject","gold",
            "yJ_letter","yG_letter","yJ_correct","yG_correct",
            "j1","j2","agg",
            "self_selected",
            "yJ_text","yG_text",
            "question","A","B","C","D"
        ])
        for r in results:
            A,B,C,D = r.choices
            w.writerow([
                r.idx, r.subject, r.gold,
                r.yJ_letter, r.yG_letter, r.yJ_correct, r.yG_correct,
                r.j1, r.j2, r.agg,
                r.self_selected,
                r.yJ_text, r.yG_text,
                r.question, A,B,C,D
            ])
    print(f"Saved per-item results to: {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Self-preference eval with vLLM on MMLU")
    p.add_argument("--subject", type=str, default="all", help="MMLU subject or 'all'")
    p.add_argument("--max-samples", type=int, default=50, help="Max items to evaluate.")
    p.add_argument("--out", type=str, default="mmlu_self_pref_vllm.csv", help="Output CSV path.")
    # vLLM runtime opts
    p.add_argument("--gpu-mem-util", type=float, default=0.90, help="vLLM gpu_memory_utilization")
    p.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    p.add_argument("--dtype", type=str, default=None, help="Optional vLLM dtype (e.g. bfloat16, auto)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
