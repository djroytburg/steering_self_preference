source .venv/bin/activate
python3 run_self_preference.py --output_path results --max_tokens 12
python3 analyze_self_preference.py --input_dir results/google_gemma-3-12b-it/deepseek-ai_DeepSeek-R1-Distill-Qwen-32B