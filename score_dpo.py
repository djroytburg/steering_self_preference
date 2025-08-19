import matplotlib.pyplot as plt
import numpy as np

# Example data (replace with your actual numbers)
data_types = ['Bias', 'Agreement', 'LSP']
base_values = [0.39473684210526316, 0.4, 0.5230769230769231]  # Example: base scores for each type
dpo_values = [0.47368421052631576, 0.9076923076923077, 1.0]   # Example: dpo scores for each type

x = np.arange(len(data_types))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(7,5))
rects1 = ax.bar(x - width/2, base_values, width, label='Base', color='skyblue')
rects2 = ax.bar(x + width/2, dpo_values, width, label='DPO', color='orange')

# Add labels, title, and legend
ax.set_ylabel('Score')
ax.set_title('Bias Comparison: Base vs DPO')
ax.set_xticks(x)
ax.set_xticklabels(data_types)
ax.legend()

# Optionally, add value labels on bars
for rects in [rects1, rects2]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

plt.tight_layout()
plt.savefig("bias_comparison_bar.png", dpi=300)
plt.show()