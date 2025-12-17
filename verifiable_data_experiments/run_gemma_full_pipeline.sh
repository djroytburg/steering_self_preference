#!/bin/bash
#SBATCH --job-name=gemma_full
#SBATCH --partition=general
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --output=logs/gemma_full_%j.log
#SBATCH --error=logs/gemma_full_%j.log
#SBATCH --mem=400G

set -euo pipefail

MODEL="google/gemma-3-27b-it"
MODEL_NAME="google_gemma-3-27b-it"
DATA_JSONL="arena_diffs/gemma-3-27b-it_arena.jsonl"
OUTPUT_BASE="results_arena_sample"
SELF_PREF_RESULTS_PATH="${OUTPUT_BASE}"

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

# Prefetch model shards once to avoid long I/O stalls during HF load
echo "Prefetching model shards into RAM for ${MODEL}..."
CACHE_MODEL_NAME="models--${MODEL/\//--}"
CACHE_PATH="/data/hf_cache/hub/${CACHE_MODEL_NAME}/snapshots"
# if [ -d "$CACHE_PATH" ]; then
#     SNAPSHOT=$(ls -t "$CACHE_PATH" | head -n 1)
#     FULL_PATH="${CACHE_PATH}/${SNAPSHOT}"
#     echo "Found snapshot at: ${FULL_PATH}"
#     if [ -d "$FULL_PATH" ]; then
#         # swallow output to avoid huge log
#         cat "${FULL_PATH}"/*.safetensors > /dev/null 2>&1 || true
#         echo "Prefetch complete."
#     else
#         echo "Snapshot directory not found, skipping prefetch."
#     fi
# else
#     echo "Cache path ${CACHE_PATH} not found, skipping prefetch."
# fi

# 1) Skip Arena self-preference: results already present in the output directory
echo "\n== Skipping arena self-preference (pairwise A/B/T) - results assumed present =="
echo "If you need to re-run it, remove this skip and re-enable run_arena_self_preference.py"

# 2) Run pairwise self-recognition (identification) using run_self_recognition.py
echo "\n== Running pairwise self-recognition =="
python3 run_self_recognition.py \
    --output_path "${OUTPUT_BASE}" \
    --judge "${MODEL}" \
    --data_jsonl "${DATA_JSONL}" \
    --self_pref_results_path "${SELF_PREF_RESULTS_PATH}" \
    --bf16 \
    --logprobs_k 50 \
    --max_tokens 10

# 3) Run individual recognition (Yes/No per response)
echo "\n== Running individual self-recognition (Yes/No) =="
python3 run_individual_self_recognition.py \
    --output_path "${OUTPUT_BASE}" \
    --judge "${MODEL}" \
    --data_jsonl "${DATA_JSONL}" \
    --self_pref_results_path "${SELF_PREF_RESULTS_PATH}" \
    --bf16

# 4) Run analyses and generate visualizations
echo "\n== Running analyses and generating plots =="
PAIRWISE_RESULTS="${OUTPUT_BASE}/${MODEL_NAME}/self_recognition/output.jsonl"
INDIV_RESULTS="${OUTPUT_BASE}/${MODEL_NAME}/individual_recognition/output.jsonl"
ANALYSIS_OUT="${OUTPUT_BASE}/${MODEL_NAME}/self_recognition/analysis.json"

python3 analyze_self_recognition.py \
    --results_file "${PAIRWISE_RESULTS}" \
    --output_json "${ANALYSIS_OUT}" \
    --model_name "${MODEL}"

python3 analyze_individual_recognition.py \
    --results_file "${INDIV_RESULTS}" \
    --model_name "${MODEL}"

echo "\nFull pipeline complete. Results and plots are in: ${OUTPUT_BASE}/${MODEL_NAME}/"
