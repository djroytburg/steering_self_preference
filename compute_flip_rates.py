import json
import sys
from collections import defaultdict

def compute_flip_rates(filepath):
    """
    Compute flip rates from a JSONL file.
    Flip rate: fraction of entries where top predicted token != unbiased_output,
    grouped by dataset and multiplier.
    """
    data = defaultdict(lambda: defaultdict(list))
    
    with open(filepath, 'r') as f:
        for line in f:
            entry = json.loads(line)
            dataset = entry['dataset']
            multiplier = entry['mult']
            unbiased_output = entry['unbiased_output']
            output = entry['output']
            
            # Find top token
            top_token = max(output, key=lambda x: x[1])[0]
            is_flip = top_token != unbiased_output
            
            data[dataset][multiplier].append(is_flip)
    
    results = {}
    for dataset, mult_dict in data.items():
        results[dataset] = {}
        for mult, flips in mult_dict.items():
            flip_rate = sum(flips) / len(flips) if flips else 0.0
            results[dataset][mult] = flip_rate
    
    return results

def main():
    if len(sys.argv) != 2:
        print("Usage: python compute_flip_rates.py <filepath>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    results = compute_flip_rates(filepath)
    
    print("Flip Rates (fraction where top token != unbiased_output):")
    for dataset, mult_dict in results.items():
        print(f"\n{dataset}:")
        for mult, rate in sorted(mult_dict.items()):
            print(f"  {mult}: {rate:.3f}")

def test_compute_flip_rates():
    import tempfile
    import os
    # Mock data
    mock_data = [
        {"dataset": "bias", "mult": -0.5, "unbiased_output": "1", "output": [["1", 0.9], ["2", 0.1]]},  # no flip
        {"dataset": "bias", "mult": -0.5, "unbiased_output": "1", "output": [["2", 0.8], ["1", 0.2]]},  # flip
        {"dataset": "bias", "mult": 0.1, "unbiased_output": "2", "output": [["2", 0.7], ["1", 0.3]]},   # no flip
        {"dataset": "lsp", "mult": -0.5, "unbiased_output": "1", "output": [["1", 0.6], ["2", 0.4]]},    # no flip
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for entry in mock_data:
            f.write(json.dumps(entry) + '\n')
        temp_path = f.name
    
    try:
        results = compute_flip_rates(temp_path)
        
        # Assertions
        assert results["bias"][-0.5] == 0.5  # 1 flip out of 2
        assert results["bias"][0.1] == 0.0   # 0 flips out of 1
        assert results["lsp"][-0.5] == 0.0   # 0 flips out of 1
        
        print("All unit tests passed!")
    finally:
        os.unlink(temp_path)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_compute_flip_rates()
    else:
        main()