#!/usr/bin/env python3
"""
Query Llama-3.1-70B-Instruct on all Hendrycks MATH test questions via OpenRouter,
using the OpenAI AsyncOpenAI client with bounded concurrency and resume support.

- Dataset: EleutherAI/hendrycks_math (all subjects, test split)
- Model:  meta-llama/llama-3.1-70b-instruct  (OpenRouter)
- Output: JSONL, one record per problem

Resume behavior:
- If --resume is set and the output file exists, previously completed
  (subject, index) pairs are loaded from the JSONL.
- Those jobs are skipped; new results are appended to the same file.
"""

import argparse
import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Set

from datasets import load_dataset
from openai import AsyncOpenAI

# All Hendrycks MATH subjects in the HF dataset
DEFAULT_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]
# Single model: Llama 3.1 70B Instruct on OpenRouter
MODEL_ID = "qwen/qwen-2.5-coder-32b-instruct"


def collect_hendrycks_math_test(
    subjects: List[str],
    max_per_subject: Optional[int] = None,
) -> List[Tuple[str, int, Dict[str, Any]]]:
    """
    Collect (subject, index, example_dict) for all requested subjects.
    example_dict has keys: 'problem', 'solution', 'level', 'type', ...
    """
    jobs: List[Tuple[str, int, Dict[str, Any]]] = []

    for subject in subjects:
        print(f"Loading subject '{subject}' (test split)...", flush=True)
        ds = load_dataset("EleutherAI/hendrycks_math", subject, split="test")

        for i, ex in enumerate(ds):
            if (max_per_subject is not None) and (i >= max_per_subject):
                break
            jobs.append((subject, i, dict(ex)))

    print(f"Total problems collected: {len(jobs)}", flush=True)
    return jobs


def load_completed_keys(path: str) -> Set[Tuple[str, int]]:
    """
    Read an existing JSONL output file and return the set of (subject, index)
    pairs already completed.
    """
    done: Set[Tuple[str, int]] = set()
    if not os.path.exists(path):
        return done

    print(f"Resuming from existing file: {path}", flush=True)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                subj = rec.get("subject")
                idx = rec.get("index")
                if subj is not None and idx is not None:
                    done.add((str(subj), int(idx)))
            except Exception as e:
                # If a line is corrupted, just report and keep going.
                print(f"Warning: could not parse line while resuming: {e}", flush=True)
    print(f"Found {len(done)} completed records.", flush=True)
    return done


async def call_openrouter_async(
    client: AsyncOpenAI,
    problem_text: str,
    system_prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    retries: int = 3,
    backoff_start: float = 5.0,
) -> str:
    """
    Call OpenRouter (via AsyncOpenAI client) for a single problem.
    Returns the model's text response.
    Retries on exceptions with exponential backoff.
    """
    backoff = backoff_start

    for attempt in range(retries):
        try:
            resp = await client.chat.completions.create(
                model=MODEL_ID,
                extra_headers={
                    "HTTP-Referer": "https://example.com",
                    "X-Title": "hendrycks-math-eval-script",
                },
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": problem_text},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"Attempt {attempt+1}/{retries} failed: {e}", flush=True)
            if attempt == retries - 1:
                raise
            await asyncio.sleep(backoff)
            backoff *= 2

    raise RuntimeError("Exhausted retries unexpectedly")


async def worker_task(
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
    job: Tuple[str, int, Dict[str, Any]],
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    write_lock: asyncio.Lock,
    f_out,  # file handle opened in append or write mode
):
    """
    One concurrent worker: solves a single problem and writes the record.
    """
    subject, idx, ex = job
    problem = ex.get("problem", "")
    solution = ex.get("solution", "")
    level = ex.get("level", None)
    qtype = ex.get("type", None)

    async with sem:
        print(f"[{subject} idx={idx}] querying {MODEL_ID}...", flush=True)
        try:
            response_text = await call_openrouter_async(
                client=client,
                problem_text=problem,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            print(f"!! Error for subject={subject}, idx={idx}: {e}", flush=True)
            response_text = None  # or str(e) to record error text

    record: Dict[str, Any] = {
        "subject": subject,
        "index": idx,
        "level": level,
        "type": qtype,
        "problem": problem,
        "solution": solution,
        "model": MODEL_ID,
        "response": response_text,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    # Write record to file (guarded by a lock for thread-safety)
    line = json.dumps(record, ensure_ascii=False)
    async with write_lock:
        f_out.write(line + "\n")
        f_out.flush()


async def run_async(args):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: Please set the OPENROUTER_API_KEY environment variable.")

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    system_prompt = (
        "Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem.\n\nProblem:\n\n"
    )

    print("Subjects:", args.subjects)
    print("Model:", MODEL_ID)
    print("Output:", args.output)
    print("Max per subject:", args.max_per_subject)
    print("Concurrency:", args.concurrency)
    print("Resume:", args.resume)
    print("----", flush=True)

    # Load all jobs
    all_jobs = collect_hendrycks_math_test(args.subjects, args.max_per_subject)

    # Figure out which jobs are already done
    done_keys: Set[Tuple[str, int]] = set()
    file_mode = "w"
    if args.resume:
        done_keys = load_completed_keys(args.output)
        file_mode = "a"  # append to existing file if resuming

    # Filter out completed jobs
    remaining_jobs = [
        job for job in all_jobs
        if (job[0], job[1]) not in done_keys
    ]

    print(
        f"Remaining problems to run: {len(remaining_jobs)} "
        f"(skipped {len(done_keys)} already completed)",
        flush=True,
    )

    if not remaining_jobs:
        print("Nothing to do; all problems already completed.")
        return

    sem = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()

    async with client:
        # Open file once; workers will append under write_lock
        with open(args.output, file_mode, encoding="utf-8") as f_out:
            tasks = [
                asyncio.create_task(
                    worker_task(
                        sem=sem,
                        client=client,
                        job=job,
                        system_prompt=system_prompt,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        write_lock=write_lock,
                        f_out=f_out,
                    )
                )
                for job in remaining_jobs
            ]

            # Iterate as tasks complete
            for coro in asyncio.as_completed(tasks):
                await coro
                if args.sleep > 0:
                    await asyncio.sleep(args.sleep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output JSONL file.",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        nargs="*",
        default=DEFAULT_SUBJECTS,
        help=f"Subset of subjects to run on (default: all: {', '.join(DEFAULT_SUBJECTS)}).",
    )
    parser.add_argument(
        "--max-per-subject",
        type=int,
        default=None,
        help="Optional cap on number of test problems per subject (for debugging).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Number of concurrent API calls.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional delay (seconds) between completed calls.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="max_tokens for each completion.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If set, load existing output file and skip completed problems.",
    )

    args = parser.parse_args()
    asyncio.run(run_async(args))


if __name__ == "__main__":
    main()
