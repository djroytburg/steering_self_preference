#!/bin/bash
#SBATCH --job-name=margiela_cane
#SBATCH --partition=general
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --output=logs/%j.log
#SBATCH --error=logs/%j.log
#SBATCH --mem=800G

source .venv/bin/activate

# Qwen3-4B tests
python3 run_self_preference.py --judge Qwen/Qwen3-4B --ref openai/gpt-oss-20b --output_path results --max_tokens 12 --bf16
python3 analyze_self_preference.py --input_dir results/Qwen_Qwen3-4B/openai_gpt-oss-20b

python3 run_self_preference.py --judge Qwen/Qwen3-4B --ref openai/gpt-oss-120b --output_path results --max_tokens 12 --bf16
python3 analyze_self_preference.py --input_dir results/Qwen_Qwen3-4B/openai_gpt-oss-120b

# Qwen3-8B tests
python3 run_self_preference.py --judge Qwen/Qwen3-8B --ref openai/gpt-oss-20b --output_path results --max_tokens 12 --bf16
python3 analyze_self_preference.py --input_dir results/Qwen_Qwen3-8B/openai_gpt-oss-20b

python3 run_self_preference.py --judge Qwen/Qwen3-8B --ref openai/gpt-oss-120b --output_path results --max_tokens 12 --bf16
python3 analyze_self_preference.py --input_dir results/Qwen_Qwen3-8B/openai_gpt-oss-120b

# Magistral-Small-2509 tests
python3 run_self_preference.py --judge mistralai/Magistral-Small-2509 --ref google/gemma-3-12b-it --output_path results --max_tokens 12 --bf16
python3 analyze_self_preference.py --input_dir results/mistralai_Magistral-Small-2509/google_gemma-3-12b-it

python3 run_self_preference.py --judge mistralai/Magistral-Small-2509 --ref google/gemma-3-27b-it --output_path results --max_tokens 12 --bf16
python3 analyze_self_preference.py --input_dir results/mistralai_Magistral-Small-2509/google_gemma-3-27b-it

python3 run_self_preference.py --judge mistralai/Magistral-Small-2509 --ref Qwen/Qwen3-32B --output_path results --max_tokens 12 --bf16
python3 analyze_self_preference.py --input_dir results/mistralai_Magistral-Small-2509/Qwen_Qwen3-32B

python3 run_self_preference.py --judge mistralai/Magistral-Small-2509 --ref Qwen/QwQ-32B-Preview --output_path results --max_tokens 12 --bf16
python3 analyze_self_preference.py --input_dir results/mistralai_Magistral-Small-2509/Qwen_QwQ-32B-Preview

python3 run_self_preference.py --judge mistralai/Magistral-Small-2509 --ref Qwen/Qwen3-0.6B --output_path results --max_tokens 12 --bf16
python3 analyze_self_preference.py --input_dir results/mistralai_Magistral-Small-2509/Qwen_Qwen3-0.6B

python3 run_self_preference.py --judge mistralai/Magistral-Small-2509 --ref Qwen/Qwen3-4B --output_path results --max_tokens 12 --bf16
python3 analyze_self_preference.py --input_dir results/mistralai_Magistral-Small-2509/Qwen_Qwen3-4B

python3 run_self_preference.py --judge mistralai/Magistral-Small-2509 --ref Qwen/Qwen3-8B --output_path results --max_tokens 12 --bf16
python3 analyze_self_preference.py --input_dir results/mistralai_Magistral-Small-2509/Qwen_Qwen3-8B

python3 run_self_preference.py --judge mistralai/Magistral-Small-2509 --ref openai/gpt-oss-20b --output_path results --max_tokens 12 --bf16
python3 analyze_self_preference.py --input_dir results/mistralai_Magistral-Small-2509/openai_gpt-oss-20b