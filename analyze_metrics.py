import json
import argparse
from collections import defaultdict

def main(filepath):
    # Load the JSONL file
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            results.append(json.loads(line))

    # Group by dataset and mult
    grouped = defaultdict(lambda: defaultdict(list))
    for entry in results:
        dataset = entry['dataset']
        mult = entry['mult']
        grouped[dataset][mult].append(entry)

    # Compute the metrics
    metrics = {}
    for dataset, mults in grouped.items():
        metrics[dataset] = {}
        for mult, entries in mults.items():
            indicators = []
            for entry in entries:
                unbiased = entry['unbiased_output']
                output = entry['output']
                if output:
                    # max prob where token == unbiased_output
                    probs = [a[1] for a in output if a[0] == unbiased]
                    if probs:
                        max_prob = max(probs)
                        if dataset == 'bias':
                            indicator = 1 if max_prob > 0.5 else 0
                        else:  # lsp and agreement
                            indicator = 1 if max_prob < 0.5 else 0
                        indicators.append(indicator)
            if indicators:
                metrics[dataset][mult] = sum(indicators) / len(indicators)
            else:
                metrics[dataset][mult] = 0

    # Print results
    for dataset in sorted(metrics.keys()):
        print(f'Dataset: {dataset}')
        for mult in sorted(metrics[dataset].keys()):
            prob = metrics[dataset][mult]
            print(f'  Mult {mult}: {prob:.3f}')
        print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze steering evaluation metrics from JSONL file.')
    parser.add_argument('filepath', type=str, help='Path to the JSONL file')
    args = parser.parse_args()
    main(args.filepath)