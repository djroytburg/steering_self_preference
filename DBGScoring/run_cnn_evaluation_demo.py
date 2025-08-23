#!/usr/bin/env python3
"""
Demo script showing how to run CNN dataset evaluation
"""
import os
import subprocess

def run_cnn_evaluation_demo():
    """
    Demonstrate the full CNN evaluation pipeline
    """
    print("=== CNN Dataset Evaluation Demo ===\n")
    
    # Step 1: Data conversion (already done)
    print("✅ Step 1: Data converted to JSONL format")
    print("   - model_responses_fullset/cnn/gpt-3.5-turbo.jsonl")
    print("   - model_responses_fullset/cnn/llama3.1-8b-instruct.jsonl")
    print("   Command used: python convert_data_generic.py --dataset cnn\n")
    
    # Step 2: Show how to run evaluation
    print("📋 Step 2: Run CNN evaluation with different judges")
    
    evaluation_commands = [
        # Standard CNN evaluation with Claude
        {
            "name": "CNN Evaluation - Claude Judge",
            "cmd": [
                "python", "gen_preference.py",
                "--data_dir", "model_responses_fullset/cnn",
                "--data_type", "cnn",
                "--model_name_1", "llama3.1-8b-instruct",
                "--model_name_2", "gpt-3.5-turbo",
                "--model_type", "anthropic/claude-3-5-sonnet-20241022",
                "--is_instruct"
            ]
        },
        # Self-aware CNN evaluation
        {
            "name": "CNN Evaluation - Llama Self-Aware", 
            "cmd": [
                "python", "gen_preference.py",
                "--data_dir", "model_responses_fullset/cnn",
                "--data_type", "cnn", 
                "--model_name_1", "llama3.1-8b-instruct",
                "--model_name_2", "gpt-3.5-turbo",
                "--model_type", "llama3.1-8b-instruct",
                "--is_instruct",
                "--self_aware_evaluation"
            ]
        }
    ]
    
    for eval_config in evaluation_commands:
        print(f"   {eval_config['name']}:")
        print(f"   {' '.join(eval_config['cmd'])}\n")
    
    # Step 3: Show bias analysis
    print("📊 Step 3: Extract preference examples for bias analysis")
    
    bias_commands = [
        {
            "name": "CNN Standard Bias Analysis",
            "cmd": [
                "python", "extract_preference_examples.py",
                "--target_model", "llama3.1-8b-instruct",
                "--comparison_model", "gpt-3.5-turbo", 
                "--data_type", "cnn",
                "--base_dir", "model_preferences_fullset/cnn"
            ]
        },
        {
            "name": "CNN Self-Aware Bias Analysis",
            "cmd": [
                "python", "extract_preference_examples.py",
                "--target_model", "llama3.1-8b-instruct",
                "--comparison_model", "gpt-3.5-turbo",
                "--data_type", "cnn", 
                "--base_dir", "model_preferences_fullset/cnn",
                "--self_aware_evaluation"
            ]
        }
    ]
    
    for bias_config in bias_commands:
        print(f"   {bias_config['name']}:")
        print(f"   {' '.join(bias_config['cmd'])}\n")
    
    # Step 4: Show expected outputs
    print("📁 Expected Output Structure:")
    expected_files = [
        "model_preferences_fullset/cnn/evaluator_claude-3-5-sonnet-20241022/llama3.1-8b-instruct_gpt-3.5-turbo.jsonl",
        "model_preferences_fullset/cnn/evaluator_llama3.1-8b-instruct/average_llama3.1-8b-instruct_gpt-3.5-turbo.jsonl", 
        "model_preferences_fullset/cnn/evaluator_llama3.1-8b-instruct_aware/average_llama3.1-8b-instruct_gpt-3.5-turbo.jsonl",
        "preference_examples/cnn_llama3.1-8b-instruct_bias_examples.jsonl",
        "preference_examples/cnn_llama3.1-8b-instruct_agreement_examples.jsonl",
        "preference_examples/cnn_llama3.1-8b-instruct_aware_bias_examples.jsonl", 
        "preference_examples/cnn_llama3.1-8b-instruct_aware_agreement_examples.jsonl"
    ]
    
    for file_path in expected_files:
        print(f"   - {file_path}")
    
    print(f"\n✨ CNN evaluation system is ready!")
    print("   The system now supports both XSum and CNN datasets with the same evaluation pipeline.")

if __name__ == "__main__":
    run_cnn_evaluation_demo()
