import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from glob import glob
import plotly.express as px
import plotly.graph_objects as go
import argparse

# --- Data Loading ---
caa_files = sorted(glob('steering_evals/caa/optimization/*.jsonl'))
opt_files = sorted(glob('steering_evals/optimization/*.jsonl'))
base_files = sorted(glob('preference_extraction/aware/*.jsonl') + glob('preference_extraction/unaware/*.jsonl'))

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

caa_results = [load_jsonl(f) for f in caa_files]
opt_results = [load_jsonl(f) for f in opt_files]
base_results = []
for f in base_files:
    base_results.extend(load_jsonl(f))

# --- Aggregation ---
agg = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

def get_awareness(path):
    if 'aware' in path: return 'aware'
    if 'unaware' in path: return 'unaware'
    return 'unknown'

def get_method_layer(path):
    if 'caa' in path:
        layer = os.path.basename(path).split('_')[1] if '_' in os.path.basename(path) else 'all'
        return 'CAA', layer
    return 'Optimization', 'opt'

def get_prob_for_desired(output_tuples, desired_output):
    undesired_output = "Mine" if desired_output == "Other" \
        else "2" if desired_output == '1' \
        else "1" if desired_output == '2' \
        else "Other" if desired_output in ["Mine", "Self"] \
        else None
    for token, prob in output_tuples:
        if token.strip() == desired_output:
            return prob
        elif token.strip() == undesired_output:
            return -prob
    return None

def get_bias_type(r):
    return r.get('bias_type', None)

# CAA results
for result_set, filelist in zip(caa_results, caa_files):
    method, layer = get_method_layer(filelist)
    awareness = get_awareness(filelist)
    for r in result_set:
        multiplier = r.get('multiplier', None)
        prob = None
        bias_type = get_bias_type(r)
        # Use base prob if available
        if 'target_model_judgment' in r and 'llama3.1-8b-instruct_prob' in r['target_model_judgment']:
            prob = r['target_model_judgment']['llama3.1-8b-instruct_prob']
        else:
            desired_output = r.get('desired_output', None)
            prob = get_prob_for_desired(r['output'], desired_output)
        if multiplier is not None and prob is not None and bias_type:
            agg[(awareness, method, layer)][multiplier][bias_type].append(prob)

# Optimization results
for result_set, filelist in zip(opt_results, opt_files):
    method, layer = get_method_layer(filelist)
    awareness = get_awareness(filelist)
    for r in result_set:
        multiplier = r.get('multiplier', None)
        prob = None
        bias_type = get_bias_type(r)
        if 'target_model_judgment' in r and 'llama3.1-8b-instruct_prob' in r['target_model_judgment']:
            prob = r['target_model_judgment']['llama3.1-8b-instruct_prob']
        else:
            desired_output = r.get('desired_output', None)
            prob = get_prob_for_desired(r['output'], desired_output)
        if multiplier is not None and prob is not None and bias_type:
            agg[(awareness, method, layer)][multiplier][bias_type].append(prob)

# Base results
for r in base_results:
    awareness = get_awareness(r.get('source', ''))
    method = 'Base'
    layer = 'base'
    multiplier = 0
    prob = None
    bias_type = get_bias_type(r)
    # Use base prob if available
    if 'target_model_judgment' in r and 'llama3.1-8b-instruct_prob' in r['target_model_judgment']:
        prob = r['target_model_judgment']['llama3.1-8b-instruct_prob']
    elif 'target_judgment' in r and 'llama3.1-8b-instruct_prob' in r['target_judgment']:
        prob = r['target_judgment']['llama3.1-8b-instruct_prob']
    else:
        desired_output = r.get('desired_output', None)
        prob = get_prob_for_desired(r['output'], desired_output)
    if prob is not None and bias_type is not None:
        agg[(awareness, method, layer)][multiplier][bias_type].append(prob)

parser = argparse.ArgumentParser(description='Visualize steering trajectory for a specific bias type.')
parser.add_argument('--bias_type', type=str, required=True, help='Which bias type to plot: self_preference_bias, unbiased_agreement, legitimate_self_preference')
args = parser.parse_args()
selected_bias_type = args.bias_type

# --- Prepare DataFrame for Plotly ---
plot_rows = []
color_map = {
    ('aware', 'CAA'): '#FF563F',
    ('unaware', 'CAA'): '#F5C0B8',
    ('aware', 'Optimization'): '#55C89F',
    ('unaware', 'Optimization'): '#363432',
    ('aware', 'Base'): '#F9DA81',
    ('unaware', 'Base'): '#F9DA81',
}
shape_map = {
    'CAA': 'square',
    'Optimization': 'circle',
    'Base': 'triangle-up',
}

for (aware, method, layer), mult_dict in agg.items():
    multipliers = sorted(mult_dict.keys())
    for m in multipliers:
        vals = mult_dict[m][selected_bias_type]
        if not vals: continue
        mean = np.mean(np.abs(vals)) if 'bias' in selected_bias_type else np.mean(vals)
        plot_rows.append({
            'Awareness': aware,
            'Method': method,
            'Layer': layer,
            'Multiplier': m,
            'BiasType': selected_bias_type,
            'MeanProb': mean,
            'Color': color_map.get((aware, method), '#cccccc'),
            'Shape': shape_map.get(method, 'circle'),
        })

if not plot_rows:
    print(f"No data available for plotting for bias_type: {selected_bias_type}")
else:
    df = pd.DataFrame(plot_rows)
    fig = go.Figure()
    for (aware, method), group in df.groupby(['Awareness', 'Method']):
        name = f"{method} {aware} {selected_bias_type}"
        if method == 'CAA':
            for layer, layer_group in group.groupby('Layer'):
                fig.add_trace(go.Scatter(
                    x=layer_group['Multiplier'],
                    y=layer_group['MeanProb'],
                    mode='lines+markers',
                    name=f"CAA {layer} {aware} {selected_bias_type}",
                    marker=dict(symbol=shape_map[method], color=color_map[(aware, method)], size=8),
                    line=dict(color=color_map[(aware, method)], width=2),
                ))
        else:
            fig.add_trace(go.Scatter(
                x=group['Multiplier'],
                y=group['MeanProb'],
                mode='lines+markers',
                name=name,
                marker=dict(symbol=shape_map[method], color=color_map[(aware, method)], size=8),
                line=dict(color=color_map[(aware, method)], width=2),
            ))
    fig.update_layout(
        title={
            'text': f"Steering Vector Effect on Output Probability ({selected_bias_type})",
            'font': {'size': 16, 'color': '#0c0c0c', 'family': 'Space Grotesk'},
            'x': 0.5, 'y': 0.96, 'xanchor': 'center', 'yanchor': 'top',
        },
        font={'family': 'Space Grotesk, Work Sans, sans-serif', 'color': '#0c0c0c'},
        margin={'l': 40, 'r': 40, 't': 100, 'b': 40},
        legend={
            'orientation': 'h', 'y': 1.0, 'x': 0.5,
            'xanchor': 'center', 'yanchor': 'bottom',
            'font': {'size': 10, 'color': '#928e8b'},
        },
        xaxis={
            'title': {'text': 'Steering Vector Multiplier',},
            'gridcolor': '#f5f5f5', 'linecolor': '#e5dfdf', 'linewidth': 1.5,
            'tickfont': {'color': '#928E8B'}, 'ticksuffix': '   '
        },
        yaxis={
            'title': {'text': ''},
            'gridcolor': '#f5f5f5', 'linecolor': '#e5dfdf', 'linewidth': 1.5,
            'tickfont': {'color': '#928E8B'}, 'ticksuffix': '   ',
            'range': [0, 1],
        },
        autosize=True,
    )
    fig.add_annotation(xref='paper', yref='paper', x=0.5, y=0.01, text="P(self)", showarrow=False, font=dict(size=14), yanchor='bottom')
    fig.add_annotation(xref='paper', yref='paper', x=0.5, y=0.98, text="P(other)", showarrow=False, font=dict(size=14), yanchor='top')
    fig.update_traces(
        hoverlabel=dict(
            bgcolor='#0c0c0c',
            font_color='#ffffff',
            font_family='Work Sans',
        ),
        hovertemplate='&nbsp;%{x}<br>'+'&nbsp;%{y}<extra></extra>'
    )
    fig.write_image(f"trajectory_steering_plotly_{selected_bias_type}.png", scale=2)
    fig.show()