import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Load results
results = []
with open("bad_vector_evaluation_results.jsonl") as f:
    for line in f:
        results.append(json.loads(line))

# Remap desired output for LSP/self_other if needed
def remap_desired_output(desired_output, experiment, dataset):
    if experiment == "self_other" and dataset == "lsp":
        return "Mine"
    return desired_output

# Find probability for desired output token
def get_prob_for_desired(output_tuples, desired_output):
    undesired_output = "Mine" if desired_output == "Other" \
        else "2" if desired_output == '1' \
        else "1" if desired_output == '2' \
        else "Other" if desired_output in ["Mine", "Self"] \
        else None

    for token, prob in output_tuples:
        # Accept both "Mine"/"Other" and "1"/"2"
        if token.strip() == desired_output:
            return prob
        elif token.strip() == undesired_output:
            return -prob
    return None

# Aggregate results
agg = defaultdict(lambda: defaultdict(list))  # (experiment, dataset) -> multiplier -> list of probs

for r in results:
    experiment = r["experiment"]
    dataset = r["dataset"]
    multiplier = r["multiplier"]
    desired_output = remap_desired_output(r["desired_output"], experiment, dataset)
    prob = get_prob_for_desired(r["output_tuples"], desired_output)
    if prob is not None:
        agg[(experiment, dataset)][multiplier].append(prob)

# Prepare plot
plt.figure(figsize=(10, 6))
colors = {
    ("12", "positives"): "blue",
    ("12", "negatives"): "red",
    ("12", "lsp"): "purple",
    ("self_other", "positives"): "cyan",
    ("self_other", "negatives"): "orange",
    ("self_other", "lsp"): "green",
}
labels = {
    ("12", "positives"): "1/2 Positives",
    ("12", "negatives"): "1/2 Negatives",
    ("12", "lsp"): "1/2 LSP",
    ("self_other", "positives"): "Self/Other Positives",
    ("self_other", "negatives"): "Self/Other Negatives",
    ("self_other", "lsp"): "Self/Other LSP",
}
print(agg)
for key, mult_dict in agg.items():
    multipliers = sorted(mult_dict.keys())
    means = [np.mean(mult_dict[m]) for m in multipliers]
    stds = [np.std(mult_dict[m]) for m in multipliers]
    # Plot line
    plt.plot(multipliers, means, label=labels[key], color=colors[key])
    # Plot dots
    plt.scatter(multipliers, means, color=colors[key])
    # Optionally, add error bars
    #plt.errorbar(multipliers, means, yerr=stds, fmt='o', color=colors[key], alpha=0.3)

plt.xlabel("Steering Vector Multiplier")
plt.ylabel("Probability of Desired Output")
plt.title("Steering Vector Effect on Desired Output Probability")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("trajectory_bad_vector.png", dpi=300, bbox_inches='tight')
plt.show()