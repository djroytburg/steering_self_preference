import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_jsonl_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.patches as mpatches


def judge_normal(data, model_1, model_2):
    """Calculate win rates for normal judges using probability scores."""
    m1_w, m2_w = 0, 0
    for example in data:
        model_1_prob = example.get(f"{model_1}_prob", 0)
        model_2_prob = example.get(f"{model_2}_prob", 0)
        
        if model_1_prob > model_2_prob:
            m1_w += 1
        elif model_2_prob > model_1_prob:
            m2_w += 1
        else:
            m1_w += 0.5
            m2_w += 0.5
    
    return m1_w, m2_w


def calculate_win_rates(model_pair_lists, data_type, base_dir, gold_judges=None):
    """
    Calculate win rates for model pairs with configurable gold judges.
    
    Args:
        model_pair_lists: List of model pairs in format "model1_model2"
        data_type: Type of evaluation task
        base_dir: Base directory for data files
        gold_judges: List of gold judge model names
    
    Returns:
        List of results dictionaries
    """
    if gold_judges is None:
        gold_judges = ["deepseek-v3"]
    
    print(f"Using gold judges: {gold_judges}")
    
    all_results = []
    
    for model_pair in model_pair_lists:
        model_1, model_2 = model_pair.split('_')
        
        # Load normal evaluator data
        evaluator_1_filepath = os.path.join(base_dir, f'evaluator_{model_1}/average_{model_1}_{model_2}.jsonl')
        evaluator_1_data = load_jsonl_data(evaluator_1_filepath)
        data_len = len(evaluator_1_data)
        ev1_mo1, ev1_mo2 = judge_normal(data=evaluator_1_data, model_1=model_1, model_2=model_2)
        
        evaluator_2_filepath = os.path.join(base_dir, f'evaluator_{model_2}/average_{model_1}_{model_2}.jsonl')
        evaluator_2_data = load_jsonl_data(evaluator_2_filepath)
        assert len(evaluator_2_data) == data_len
        ev2_mo1, ev2_mo2 = judge_normal(data=evaluator_2_data, model_1=model_1, model_2=model_2)
        
        # Load gold judge data
        gold_judge_data = {}
        for judge in gold_judges:
            judge_data_path = os.path.join(base_dir, f"evaluator_{judge}", f"merge_{model_pair}.jsonl")
            if os.path.exists(judge_data_path):
                gold_judge_data[judge] = load_jsonl_data(judge_data_path)
                print(f"Loaded {len(gold_judge_data[judge])} items from {judge}")
            else:
                print(f"Warning: Gold judge data not found at {judge_data_path}")
                continue
        
        if not gold_judge_data:
            print(f"Error: No gold judge data found for {model_pair}")
            continue
            
        # Get data IDs and create mappings
        first_judge = list(gold_judge_data.keys())[0]
        data_ids = [item['id'] for item in gold_judge_data[first_judge]]
        assert len(data_ids) == data_len
        
        judge_id_to_data = {}
        for judge, judge_data in gold_judge_data.items():
            judge_id_to_data[judge] = {item['id']: item for item in judge_data}
        
        golden_mo1, golden_mo2 = 0, 0
        
        # Aggregate preferences from all gold judges
        for sub_id in data_ids:
            all_preferences = []
            
            for judge in gold_judges:
                if judge in judge_id_to_data and sub_id in judge_id_to_data[judge]:
                    judge_preferences = judge_id_to_data[judge][sub_id]["preferences"]
                    all_preferences.extend(judge_preferences)
            
            # Count votes across all judges
            cur_mo1 = all_preferences.count(model_1)
            cur_mo2 = all_preferences.count(model_2)

            if cur_mo1 > cur_mo2:
                golden_mo1 += 1
            elif cur_mo2 > cur_mo1:
                golden_mo2 += 1
            else:
                golden_mo1 += 0.5
                golden_mo2 += 0.5
            
        # Calculate DBG scores (Difference Between judge scores and Gold judgments)
        model_1_self_win_rate = ev1_mo1 / data_len * 100
        model_2_self_win_rate = ev2_mo2 / data_len * 100
        model_1_gold_win_rate = golden_mo1 / data_len * 100
        model_2_gold_win_rate = golden_mo2 / data_len * 100
        
        model_1_dbg_score = model_1_self_win_rate - model_1_gold_win_rate
        model_2_dbg_score = model_2_self_win_rate - model_2_gold_win_rate
        
        all_results.append({
            model_1: (ev1_mo1 / data_len * 100, ev1_mo2 / data_len * 100),
            model_2: (ev2_mo1 / data_len * 100, ev2_mo2 / data_len * 100),
            "Golden": (golden_mo1 / data_len * 100, golden_mo2 / data_len * 100)
        })
        
        # Print results
        print(f"\n=== {model_pair} Results ===")
        print(f"{model_1} as judge: {model_1_self_win_rate:.1f}% vs {ev1_mo2 / data_len * 100:.1f}%")
        print(f"{model_2} as judge: {ev2_mo1 / data_len * 100:.1f}% vs {model_2_self_win_rate:.1f}%")
        print(f"Gold standard ({', '.join(gold_judges)}): {model_1_gold_win_rate:.1f}% vs {model_2_gold_win_rate:.1f}%")
        print(f"🎯 DBG Scores (Self-Preference Bias):")
        print(f"   {model_1}: {model_1_dbg_score:+.1f}% {'(self-preference bias)' if model_1_dbg_score > 0 else '(no bias)' if model_1_dbg_score == 0 else '(anti-self bias)'}")
        print(f"   {model_2}: {model_2_dbg_score:+.1f}% {'(self-preference bias)' if model_2_dbg_score > 0 else '(no bias)' if model_2_dbg_score == 0 else '(anti-self bias)'}")
        print("=" * 50)
    
    return all_results


def plot_win_rate(model_1, model_2, result_item, data_type):
    """Plot horizontal bar chart for a single model pair."""
    categories = ["Model B", "Golden", "Model A"]
    values = [
        [result_item[model_2][0], result_item[model_2][1]],
        [result_item["Golden"][0], result_item["Golden"][1]],
        [result_item[model_1][0], result_item[model_1][1]],
    ]
    
    colors = ["#3371b3", "#aed4e5"]
    fig, ax = plt.subplots(figsize=(6, 2))
    y_pos = np.arange(len(categories))

    for i in range(len(values[0])):
        left_values = [sum(values[row][:i]) for row in range(len(values))]
        ax.barh(y_pos, [values[row][i] for row in range(len(values))], 
                color=colors[i], left=left_values)

    # Add percentage labels
    for row in range(len(values)):
        for i in range(len(values[row])):
            x_text = sum(values[row][:i]) + values[row][i] / 2
            if values[row][i] < 6:
                x_text += 3
            ax.text(x_text, row, f"{values[row][i]:.1f}%", ha='center', va='center', color='white', fontsize=10)

    ax.set_title(f"Model A: {model_1}\nModel B: {model_2}", fontsize=13, fontweight='bold', loc="left", pad=2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories)
    ax.set_xticks([])
    ax.set_xticklabels([])
    
    # Clean styling
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.savefig(f"figure/win_rate/scaling/{data_type}_{model_1}_{model_2}.pdf", dpi=600, bbox_inches='tight')


def plot_win_rates_5fig(model_1_list, model_2_list, result_items, save_name):
    """Plot 5 figures in 3+2 layout."""
    num_plots = len(model_1_list)
    fig = plt.figure(figsize=(16, 4.2))

    spec_top = fig.add_gridspec(nrows=1, ncols=3, left=0., right=1, top=1, bottom=0.6, wspace=0.01)
    spec_bottom = fig.add_gridspec(nrows=1, ncols=2, left=0.17, right=0.83, top=0.4, bottom=0, wspace=0.01)

    axes = []
    for i in range(3):
        axes.append(fig.add_subplot(spec_top[0, i]))
    for i in range(2):
        axes.append(fig.add_subplot(spec_bottom[0, i]))
    
    for idx in range(num_plots):
        ax = axes[idx]
        model_1 = model_1_list[idx]
        model_2 = model_2_list[idx]
        result_item = result_items[idx]

        categories = ["Model B", "Gold", "Model A"]
        values = [
            [result_item[model_2][0], result_item[model_2][1]],
            [result_item["Golden"][0], result_item["Golden"][1]],
            [result_item[model_1][0], result_item[model_1][1]],
        ]

        colors = ["#3371b3", "#81b5d5"]
        y_pos = np.arange(len(categories))

        for i in range(len(values[0])):
            left_values = [sum(values[row][:i]) for row in range(len(values))]
            ax.barh(y_pos, [values[row][i] for row in range(len(values))], 
                    color=colors[i], left=left_values)

        for row in range(len(values)):
            for i in range(len(values[row])):
                x_text = sum(values[row][:i]) + values[row][i] / 2
                if values[row][i] < 6:
                    x_text += 2
                ax.text(x_text, row, f"{values[row][i]:.1f}%", ha='center', va='center', color='white', fontsize=14)

        ax.set_title(f"Model A: {model_1}\nModel B: {model_2}", fontsize=15, fontweight='bold', loc="left", pad=2)

        if idx % 3 == 0:
            ax.set_yticks(y_pos)
            ax.set_yticklabels(categories, fontsize=14)
            ax.set_ylabel("Judge Model", fontsize=14, labelpad=15)
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])

        ax.set_xticks([])
        ax.set_xticklabels([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.savefig(f"{save_name}.pdf", dpi=1200, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def plot_win_rates_4fig(model_1_list, model_2_list, result_items, save_name):
    """Plot 4 figures in horizontal layout."""
    num_plots = len(model_1_list)
    fig = plt.figure(figsize=(22, 2.3))
    spec_top = fig.add_gridspec(nrows=1, ncols=4, left=0, right=1, top=1, bottom=0, wspace=0.0)

    axes = []
    for i in range(4):
        axes.append(fig.add_subplot(spec_top[0, i]))
    
    for idx in range(num_plots):
        ax = axes[idx]
        model_1 = model_1_list[idx]
        model_2 = model_2_list[idx]
        result_item = result_items[idx]

        categories = ["Model B", "Gold", "Model A"]
        values = [
            [result_item[model_2][0], result_item[model_2][1]],
            [result_item["Golden"][0], result_item["Golden"][1]],
            [result_item[model_1][0], result_item[model_1][1]],
        ]

        colors = ["#3371b3", "#81b5d5"]
        y_pos = np.arange(len(categories))

        for i in range(len(values[0])):
            left_values = [sum(values[row][:i]) for row in range(len(values))]
            ax.barh(y_pos, [values[row][i] for row in range(len(values))], 
                    color=colors[i], left=left_values)

        for row in range(len(values)):
            for i in range(len(values[row])):
                x_text = sum(values[row][:i]) + values[row][i] / 2
                if values[row][i] < 6:
                    x_text += 2
                ax.text(x_text, row, f"{values[row][i]:.1f}%", ha='center', va='center', color='white', fontsize=18)

        ax.set_title(f"Model A: {model_1}\nModel B: {model_2}", fontsize=19, fontweight='bold', loc="left", pad=2)

        if idx % 4 == 0:
            ax.set_yticks(y_pos)
            ax.set_yticklabels(categories, fontsize=18)
            ax.set_ylabel("Judge Model", fontsize=18, labelpad=15)
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])

        ax.set_xticks([])
        ax.set_xticklabels([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    print(f"Save figure to {save_name}.pdf")
    plt.savefig(f"{save_name}.pdf", dpi=1200, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def plot_win_rates_7fig(model_1_list, model_2_list: list, result_items, save_name):
    """Plot line chart for scaling analysis."""
    cur_ev1_mo2, cur_ev2_mo2, cur_golden_mo2 = [], [], []
    for idx in range(len(result_items)):
        cur_ev1 = model_1_list[idx]
        cur_ev2 = model_2_list[idx]
        cur_res = result_items[idx]
        
        cur_ev1_mo2.append(cur_res[cur_ev1][1])
        cur_ev2_mo2.append(cur_res[cur_ev2][1])
        cur_golden_mo2.append(cur_res["Golden"][1])
    
    plt.figure(figsize=(8.5, 6))

    x_name = [item.split('-')[1] for item in model_2_list]
    x_range = range(len(x_name))

    marker_colors = {
        'ev1': '#1f77b4',   # Blue
        'ev2': '#ff7f0e',   # Orange
        'golden': '#2ca02c' # Green
    }

    # Connect points with vertical lines
    for i in x_range:
        y_values = sorted([cur_ev2_mo2[i], cur_golden_mo2[i]])
        plt.plot([x_name[i]] * 2, y_values, color='darkgray', linestyle=":", alpha=1, linewidth=1.5)

    plt.plot(x_name, cur_ev1_mo2, linestyle='--', marker='o', label='Llama-3.1-70B-Instruct', 
            color=marker_colors['ev1'], markerfacecolor=marker_colors['ev1'], markeredgecolor=marker_colors['ev1'])
    plt.plot(x_name, cur_ev2_mo2, linestyle='--', marker='s', label='Qwen2.5-Instruct',
            color=marker_colors['ev2'], markerfacecolor=marker_colors['ev2'], markeredgecolor=marker_colors['ev2'])
    plt.plot(x_name, cur_golden_mo2, linestyle='--', marker='^', label='Gold Judgments',
            color=marker_colors['golden'], markerfacecolor=marker_colors['golden'], markeredgecolor=marker_colors['golden'])

    plt.xlabel('Qwen2.5-Instruct series models', fontsize=16)
    plt.ylabel('Qwen2.5-Instruct Win Rate (%)', fontsize=16)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=15, rotation=0)
    plt.yticks(fontsize=15)
    plt.ylim(0, 70)
    plt.tight_layout()

    print(f"Save figure to {save_name}.pdf")
    plt.savefig(f"{save_name}.pdf", dpi=1500, bbox_inches='tight', pad_inches=0.05)


def main():
    """Main function with configurable gold judges."""
    # Configuration
    model_pair_lists = [
        'llama3.1-8b-instruct_gpt-3.5-turbo',
    ]

    # Data configuration
    data_type = "cnn"
    base_dir = "model_preferences_fullset/cnn"

    # Gold judge configuration - easily modifiable
    gold_judges = ["deepseek-v3-0324", "phi-4", "claude-3-5-sonnet-20241022"]  # Can be ["deepseek-v3", "gpt-4o-mini", "gemini-2.5-pro"]
    
    # Calculate win rates
    all_results = calculate_win_rates(model_pair_lists, data_type, base_dir, gold_judges)
    
    # Prepare plotting data
    model_1_list, model_2_list, result_items = [], [], []
    for cur_res in all_results:
        res_keys = list(cur_res.keys())
        model_1_list.append(res_keys[0])
        model_2_list.append(res_keys[1])
        result_items.append(cur_res)

    # Generate appropriate plots based on number of results
    if len(all_results) == 5:
        print("5 Fig")
        save_name = f"figure/win_rate/main/{data_type}_base_instruct"
        plot_win_rates_5fig(model_1_list, model_2_list, result_items, save_name)
        
    elif len(all_results) == 4:
        print("4 Fig")
        save_name = f"figure/win_rate/main/{data_type}_big_small"
        plot_win_rates_4fig(model_1_list, model_2_list, result_items, save_name)
        
    elif len(all_results) == 7:
        print("7 Fig")
        save_name = f"figure/win_rate/analysis/{data_type}_scaling"
        plot_win_rates_7fig(model_1_list, model_2_list, result_items, save_name)


if __name__ == "__main__":
    main()
