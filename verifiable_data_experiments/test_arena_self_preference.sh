#!/bin/bash
#SBATCH --job-name=arena_sp
#SBATCH --partition=general
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --output=logs/%j_arena.log
#SBATCH --error=logs/%j_arena.log
#SBATCH --mem=800G

source .venv/bin/activate

# ============================================================================
# ARENA SELF-PREFERENCE TESTING
# Models tested on their actual Arena comparisons (wins and losses)
# Output directory: results_arena/
# ============================================================================

# ============================================================================
# 4 GPU TESTS (Currently Available)
# Models that can run on 4x A100/L40S GPUs
# ============================================================================

# Gemma 3 27B - 6,146 comparisons across 51 opponents
python3 run_arena_self_preference.py \
    --judge google/gemma-3-27b-it \
    --data_jsonl arena_diffs/gemma-3-27b-it_arena.jsonl \
    --output_path results_arena \
    --max_tokens 12 \
    --bf16
python3 analyze_self_preference.py --input_dir results_arena/google_gemma-3-27b-it/arena

# Mistral Small 2506 - 1,981 comparisons
python3 run_arena_self_preference.py \
    --judge mistralai/mistral-small-2506 \
    --data_jsonl arena_diffs/mistral-small-2506_arena.jsonl \
    --output_path results_arena \
    --max_tokens 12 \
    --bf16
python3 analyze_self_preference.py --input_dir results_arena/mistralai_mistral-small-2506/arena

# Mistral Small 3.1 24B - 2,814 comparisons
python3 run_arena_self_preference.py \
    --judge mistralai/mistral-small-3.1-24b-instruct-2503 \
    --data_jsonl arena_diffs/mistral-small-3.1-24b-instruct-2503_arena.jsonl \
    --output_path results_arena \
    --max_tokens 12 \
    --bf16
python3 analyze_self_preference.py --input_dir results_arena/mistralai_mistral-small-3.1-24b-instruct-2503/arena

# QwQ-32B - 3,590 comparisons
python3 run_arena_self_preference.py \
    --judge Qwen/QwQ-32B-Preview \
    --data_jsonl arena_diffs/qwq-32b_arena.jsonl \
    --output_path results_arena \
    --max_tokens 12 \
    --bf16
python3 analyze_self_preference.py --input_dir results_arena/Qwen_QwQ-32B-Preview/arena

# Qwen3-30B-A3B - 4,592 comparisons
python3 run_arena_self_preference.py \
    --judge Qwen/Qwen3-30B-A3B \
    --data_jsonl arena_diffs/qwen3-30b-a3b_arena.jsonl \
    --output_path results_arena \
    --max_tokens 12 \
    --bf16
python3 analyze_self_preference.py --input_dir results_arena/Qwen_Qwen3-30B-A3B/arena

# Llama 3.3 70B - 4,244 comparisons (may be tight on 4 GPUs, test carefully)
python3 run_arena_self_preference.py \
    --judge meta-llama/Llama-3.3-70B-Instruct \
    --data_jsonl arena_diffs/llama-3.3-70b-instruct_arena.jsonl \
    --output_path results_arena \
    --max_tokens 12 \
    --bf16
python3 analyze_self_preference.py --input_dir results_arena/meta-llama_Llama-3.3-70B-Instruct/arena

# Smaller models (should run fine on 4 GPUs)
# Llama 4 Scout 17B - 2,418 comparisons
python3 run_arena_self_preference.py \
    --judge meta-llama/Llama-4-Scout-17B-16E-Instruct \
    --data_jsonl arena_diffs/llama-4-scout-17b-16e-instruct_arena.jsonl \
    --output_path results_arena \
    --max_tokens 12 \
    --bf16
python3 analyze_self_preference.py --input_dir results_arena/meta-llama_Llama-4-Scout-17B-16E-Instruct/arena

# Llama 4 Maverick 17B - 4,595 comparisons
python3 run_arena_self_preference.py \
    --judge meta-llama/Llama-4-Maverick-17B-128E-Instruct \
    --data_jsonl arena_diffs/llama-4-maverick-17b-128e-instruct_arena.jsonl \
    --output_path results_arena \
    --max_tokens 12 \
    --bf16
python3 analyze_self_preference.py --input_dir results_arena/meta-llama_Llama-4-Maverick-17B-128E-Instruct/arena

# Gemma 3n E4B - 2,258 comparisons
python3 run_arena_self_preference.py \
    --judge google/gemma-3n-e4b-it \
    --data_jsonl arena_diffs/gemma-3n-e4b-it_arena.jsonl \
    --output_path results_arena \
    --max_tokens 12 \
    --bf16
python3 analyze_self_preference.py --input_dir results_arena/google_gemma-3n-e4b-it/arena


# ============================================================================
# 8 GPU TESTS (Commented out - for when 8 GPUs available)
# Large models requiring more GPU memory
# ============================================================================

# # DeepSeek R1 - 5,451 comparisons, 55.8% win rate (best performer!)
# python3 run_arena_self_preference.py \
#     --judge deepseek-ai/DeepSeek-R1 \
#     --data_jsonl arena_diffs/deepseek-r1-0528_arena.jsonl \
#     --output_path results_arena \
#     --max_tokens 12 \
#     --bf16
# python3 analyze_self_preference.py --input_dir results_arena/deepseek-ai_DeepSeek-R1/arena

# # DeepSeek V3 - 5,054 comparisons
# python3 run_arena_self_preference.py \
#     --judge deepseek-ai/DeepSeek-V3 \
#     --data_jsonl arena_diffs/deepseek-v3-0324_arena.jsonl \
#     --output_path results_arena \
#     --max_tokens 12 \
#     --bf16
# python3 analyze_self_preference.py --input_dir results_arena/deepseek-ai_DeepSeek-V3/arena

# # Mistral Medium 2505 - 7,551 comparisons
# python3 run_arena_self_preference.py \
#     --judge mistralai/Mistral-Medium-2505 \
#     --data_jsonl arena_diffs/mistral-medium-2505_arena.jsonl \
#     --output_path results_arena \
#     --max_tokens 12 \
#     --bf16
# python3 analyze_self_preference.py --input_dir results_arena/mistralai_Mistral-Medium-2505/arena

# # Qwen3 235B variants (likely need 8 GPUs)
# # Qwen3-235B-A22B - 4,250 comparisons
# python3 run_arena_self_preference.py \
#     --judge Qwen/Qwen3-235B-A22B \
#     --data_jsonl arena_diffs/qwen3-235b-a22b_arena.jsonl \
#     --output_path results_arena \
#     --max_tokens 12 \
#     --bf16
# python3 analyze_self_preference.py --input_dir results_arena/Qwen_Qwen3-235B-A22B/arena

# # Qwen3-235B-A22B No Thinking - 7,564 comparisons (most examples!)
# python3 run_arena_self_preference.py \
#     --judge Qwen/Qwen3-235B-A22B-Instruct-No-Thinking \
#     --data_jsonl arena_diffs/qwen3-235b-a22b-no-thinking_arena.jsonl \
#     --output_path results_arena \
#     --max_tokens 12 \
#     --bf16
# python3 analyze_self_preference.py --input_dir results_arena/Qwen_Qwen3-235B-A22B-Instruct-No-Thinking/arena

# # Qwen3-235B-A22B Instruct 2507 - 514 comparisons
# python3 run_arena_self_preference.py \
#     --judge Qwen/Qwen3-235B-A22B-Instruct-2507 \
#     --data_jsonl arena_diffs/qwen3-235b-a22b-instruct-2507_arena.jsonl \
#     --output_path results_arena \
#     --max_tokens 12 \
#     --bf16
# python3 analyze_self_preference.py --input_dir results_arena/Qwen_Qwen3-235B-A22B-Instruct-2507/arena


# ============================================================================
# FULL TEST SUITE (All models - for reference)
# Run this when you have sufficient GPU resources
# ============================================================================

# # Small to Medium Models (4 GPU capable)
# for model_file in \
#     "gemma-3-27b-it" \
#     "gemma-3n-e4b-it" \
#     "llama-4-scout-17b-16e-instruct" \
#     "llama-4-maverick-17b-128e-instruct" \
#     "mistral-small-2506" \
#     "mistral-small-3.1-24b-instruct-2503" \
#     "qwen3-30b-a3b" \
#     "qwq-32b"; do
#     
#     # Convert filename to model path (simple cases)
#     case $model_file in
#         gemma-3-27b-it) model="google/gemma-3-27b-it" ;;
#         gemma-3n-e4b-it) model="google/gemma-3n-e4b-it" ;;
#         llama-4-scout-17b-16e-instruct) model="meta-llama/Llama-4-Scout-17B-16E-Instruct" ;;
#         llama-4-maverick-17b-128e-instruct) model="meta-llama/Llama-4-Maverick-17B-128E-Instruct" ;;
#         mistral-small-2506) model="mistralai/mistral-small-2506" ;;
#         mistral-small-3.1-24b-instruct-2503) model="mistralai/mistral-small-3.1-24b-instruct-2503" ;;
#         qwen3-30b-a3b) model="Qwen/Qwen3-30B-A3B" ;;
#         qwq-32b) model="Qwen/QwQ-32B-Preview" ;;
#     esac
#     
#     echo "Processing $model..."
#     python3 run_arena_self_preference.py \
#         --judge "$model" \
#         --data_jsonl "arena_diffs/${model_file}_arena.jsonl" \
#         --output_path results_arena \
#         --max_tokens 12 \
#         --bf16
#     
#     # Convert model path to results directory format
#     results_dir=$(echo "$model" | tr '/' '_')
#     python3 analyze_self_preference.py --input_dir "results_arena/${results_dir}/arena"
# done

# # Large Models (8+ GPU required)
# for model_file in \
#     "deepseek-r1-0528" \
#     "deepseek-v3-0324" \
#     "llama-3.3-70b-instruct" \
#     "mistral-medium-2505" \
#     "qwen3-235b-a22b" \
#     "qwen3-235b-a22b-instruct-2507" \
#     "qwen3-235b-a22b-no-thinking"; do
#     
#     # Convert filename to model path
#     case $model_file in
#         deepseek-r1-0528) model="deepseek-ai/DeepSeek-R1" ;;
#         deepseek-v3-0324) model="deepseek-ai/DeepSeek-V3" ;;
#         llama-3.3-70b-instruct) model="meta-llama/Llama-3.3-70B-Instruct" ;;
#         mistral-medium-2505) model="mistralai/Mistral-Medium-2505" ;;
#         qwen3-235b-a22b) model="Qwen/Qwen3-235B-A22B" ;;
#         qwen3-235b-a22b-instruct-2507) model="Qwen/Qwen3-235B-A22B-Instruct-2507" ;;
#         qwen3-235b-a22b-no-thinking) model="Qwen/Qwen3-235B-A22B-Instruct-No-Thinking" ;;
#     esac
#     
#     echo "Processing $model..."
#     python3 run_arena_self_preference.py \
#         --judge "$model" \
#         --data_jsonl "arena_diffs/${model_file}_arena.jsonl" \
#         --output_path results_arena \
#         --max_tokens 12 \
#         --bf16
#     
#     # Convert model path to results directory format
#     results_dir=$(echo "$model" | tr '/' '_')
#     python3 analyze_self_preference.py --input_dir "results_arena/${results_dir}/arena"
# done
