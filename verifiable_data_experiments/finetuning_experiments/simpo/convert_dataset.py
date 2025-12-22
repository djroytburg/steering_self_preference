#!/usr/bin/env python3
"""
Convert arena self-recognition dataset from SFT format to SimPO format.

SFT format:
{
  "messages": [
    {"role": "user", "content": "Which response did you write? 1 or 2?"},
    {"role": "assistant", "content": "1"}
  ]
}

SimPO format:
{
  "chosen": [
    {"role": "user", "content": "Which response did you write? 1 or 2?"},
    {"role": "assistant", "content": "1"}  # Correct answer
  ],
  "rejected": [
    {"role": "user", "content": "Which response did you write? 1 or 2?"},
    {"role": "assistant", "content": "2"}  # Wrong answer
  ]
}
"""

import json
from pathlib import Path


def convert_arena_to_simpo(input_file, output_file):
    
    # Load the SFT format dataset
    print(f"Loading dataset from: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Found {len(data)} examples")
    
    simpo_data = []
    stats = {"answer_1": 0, "answer_2": 0}
    
    for idx, example in enumerate(data):
        # Extract user message and correct answer
        messages = example['messages']
        user_msg = messages[0]
        correct_answer = messages[1]['content']
        
        # Track statistics
        if correct_answer == "1":
            stats["answer_1"] += 1
        elif correct_answer == "2":
            stats["answer_2"] += 1
        
        # Determine wrong answer (opposite of correct)
        wrong_answer = "2" if correct_answer == "1" else "1"
        
        # Create preference pair with chosen (correct) and rejected (wrong)
        simpo_example = {
            "chosen": [
                user_msg,
                {"role": "assistant", "content": correct_answer}
            ],
            "rejected": [
                user_msg,
                {"role": "assistant", "content": wrong_answer}
            ]
        }
        
        simpo_data.append(simpo_example)
    
    # Save as JSONL (SimPO expects JSONL format, not JSON array)
    print(f"Saving to: {output_file}")
    with open(output_file, 'w') as f:
        for item in simpo_data:
            f.write(json.dumps(item) + '\n')
    
    # Print summary statistics
    print(f"Total examples converted: {len(simpo_data)}")
    print(f"Examples with correct answer '1': {stats['answer_1']}")
    print(f"Examples with correct answer '2': {stats['answer_2']}")
    print(f"\nOutput saved to: {output_file}")
    
    return simpo_data


def main():
    current_dir = Path(__file__).parent
    input_file = current_dir.parent / "arena_finetuning_dataset.json"
    output_file = current_dir / "arena_finetuning_dataset_simpo.jsonl"
    
    # Run conversion
    convert_arena_to_simpo(input_file, output_file)

if __name__ == "__main__":
    main()

