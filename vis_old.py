import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from glob import glob
import plotly.graph_objects as go
import argparse

# --- Helpers to normalize metadata and outputs ---

def get_awareness_from_path(path):
    if not path:
        return 'unknown'
    p = path.lower()
    # check 'unaware' first because 'unaware' contains the substring 'aware'
    if 'unaware' in p:
        return 'unaware'
    if 'aware' in p:
        return 'aware'
    return 'unknown'


def get_method_layer_from_path(path):
    bn = os.path.basename(path or '')
    p = (path or '').lower()
    if 'caa' in p or bn.startswith('caa'):
        parts = bn.split('_')
        layer = parts[1] if len(parts) > 1 else 'all'
        return 'CAA', layer
    if 'optimization' in p or 'opt' in bn:
        return 'Optimization', 'opt'
    return 'Base', 'base'


def get_prob_for_desired(output, desired=None):
    if output is None:
        return None
    # dict {label: prob}
    if isinstance(output, dict):
        if desired and desired in output:
            try:
                return float(output[desired])
            except Exception:
                pass
        numeric_vals = [v for v in output.values() if isinstance(v, (int, float))]
        return float(max(numeric_vals)) if numeric_vals else None
    # list/tuple
    if isinstance(output, (list, tuple)):
        # list of (label, prob) pairs
        if output and isinstance(output[0], (list, tuple)) and len(output[0]) >= 2:
            items = []
            for x in output:
                try:
                    label = str(x[0])
                    prob = float(x[1])
                    items.append((label, prob))
                except Exception:
                    continue
            if desired:
                for lbl, p in items:
                    if lbl == desired:
                        return p
            if items:
                return max((p for _, p in items))
            return None
        # list of dicts like {'label':..., 'prob':...}
        items = []
        for el in output:
            if isinstance(el, dict):
                if 'label' in el and 'prob' in el:
                    try:
                        items.append((str(el['label']), float(el['prob'])))
                    except Exception:
                        pass
                elif 'token' in el and 'prob' in el:
                    try:
                        items.append((str(el['token']), float(el['prob'])))
                    except Exception:
                        pass
                else:
                    for k, v in el.items():
                        if isinstance(v, (int, float)):
                            items.append((str(k), float(v)))
                            break
        if items:
            if desired:
                for lbl, p in items:
                    if lbl == desired:
                        return p
            return max((p for _, p in items))
    if isinstance(output, (int, float)):
        return float(output)
    return None


def get_bias_type(r):
    bias = r.get('bias_type') or r.get('dataset') or r.get('bias') or None
    remap = {
        "self_preference_bias": "bias",
        "unbiased_agreement": "agreement",
        "legitimate_self_preference": "lsp",
    }
    return remap.get(bias, bias)


# --- Load JSONL and tag each record with source metadata ---

def load_jsonl_and_tag(path):
    method, layer = get_method_layer_from_path(path)
    awareness = get_awareness_from_path(path)
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            # attach metadata
            r['__source_path'] = path
            r['__method'] = method
            r['__layer'] = layer
            r['__awareness'] = awareness
            # normalized multiplier
            m = r.get('mult', r.get('multiplier', None))
            if m is None:
                r['__multiplier'] = 0.0 if method == 'Base' else None
            else:
                try:
                    r['__multiplier'] = float(m)
                except Exception:
                    r['__multiplier'] = None
            # bias type and normalized prob
            r['__bias_type'] = get_bias_type(r)
            prob = None
            tmj = r.get('target_model_judgment')
            if isinstance(tmj, dict):
                if 'llama3.1-8b-instruct_prob' in tmj:
                    prob = tmj['llama3.1-8b-instruct_prob']
                else:
                    for v in tmj.values():
                        if isinstance(v, (int, float)):
                            prob = v
                            break
            if prob is None:
                tj = r.get('target_judgment')
                if isinstance(tj, dict):
                    if 'llama3.1-8b-instruct_prob' in tj:
                        prob = tj['llama3.1-8b-instruct_prob']
                    else:
                        for v in tj.values():
                            if isinstance(v, (int, float)):
                                prob = v
                                break
            if prob is None:
                desired = r.get('desired_output', None)
                prob = get_prob_for_desired(r.get('output', None), desired)
            try:
                r['__prob'] = float(prob) if prob is not None else None
            except Exception:
                r['__prob'] = None
            records.append(r)
    return records


# --- Gather files and load/tag records ---

caa_files = sorted(glob('steering_evals/caa/**/*.jsonl', recursive=True))
opt_files = sorted(glob('steering_evals/optimization/**/*.jsonl', recursive=True))
base_files = sorted(glob('preference_extraction/aware/*.jsonl') + glob('preference_extraction/unaware/*.jsonl'))

caa_records = []
for p in caa_files:
    caa_records.extend(load_jsonl_and_tag(p))

opt_records = []
for p in opt_files:
    opt_records.extend(load_jsonl_and_tag(p))

base_records = []
for p in base_files:
    base_records.extend(load_jsonl_and_tag(p))

all_records = caa_records + opt_records + base_records

# Debug summary: print counts, show available bias types and method/awareness breakdown,
# and save a small sample for inspection
from collections import Counter as _Counter
print(f"DEBUG: loaded {len(all_records)} records total")
num_prob = sum(1 for r in all_records if r.get('__prob') is not None)
num_bias = sum(1 for r in all_records if r.get('__bias_type') is not None)
num_mult = sum(1 for r in all_records if r.get('__multiplier') is not None)
print(f"DEBUG: __prob present: {num_prob}, __bias_type present: {num_bias}, __multiplier present: {num_mult}")
combos = _Counter((r.get('__method'), r.get('__awareness'), r.get('__multiplier'), r.get('__bias_type')) for r in all_records)
print("DEBUG: top 20 (method, awareness, multiplier, bias_type) combos and counts:")
for combo, cnt in combos.most_common(20):
    print(f"  {combo}: {cnt}")

# Print available bias types and counts so user can choose --bias_type appropriately
bias_counts = _Counter(r.get('__bias_type') for r in all_records if r.get('__bias_type') is not None)
print("\nDEBUG: available bias types and counts:")
for b, c in bias_counts.most_common():
    print(f"  {b}: {c}")

# Print method/awareness breakdown
method_counts = _Counter((r.get('__method'), r.get('__awareness')) for r in all_records)
print("\nDEBUG: method / awareness counts:")
for k, c in method_counts.most_common():
    print(f"  {k}: {c}")

# Show a sample of distinct multiplier values (up to 20)
mults = sorted({r.get('__multiplier') for r in all_records if r.get('__multiplier') is not None})
print("\nDEBUG: example multiplier values (up to 20):", mults[:20])

sample_path = os.path.join(os.path.dirname(__file__), 'visualize_trajectory_debug_sample.jsonl')
with open(sample_path, 'w') as _f:
    for r in all_records[:50]:
        _f.write(json.dumps(r) + "\n")
print(f"DEBUG: wrote sample of up to 50 enriched records to {sample_path}")

print('\nTIP: Rerun the script with --bias_type <one of the bias types above> (use the original legacy names e.g. self_preference_bias -> mapped to "bias").')

# --- Aggregate enriched records ---

agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for r in all_records:
    aware = r.get('__awareness', 'unknown')
    method = r.get('__method', 'Base')
    layer = r.get('__layer', 'base')
    multiplier = r.get('__multiplier')
    bias_type = r.get('__bias_type')
    prob = r.get('__prob')
    if multiplier is None:
        continue
    if prob is None or bias_type is None:
        continue
    agg[(aware, method, layer)][multiplier][bias_type].append(prob)

# --- CLI ---

parser = argparse.ArgumentParser(description='Visualize steering trajectory for a specific bias type.')
parser.add_argument('--bias_type', type=str, required=True, help='Which bias type to plot: self_preference_bias, unbiased_agreement, legitimate_self_preference')
args = parser.parse_args()
selected_bias_type = args.bias_type
bias_remap = {
    "self_preference_bias": "bias",
    "unbiased_agreement": "agreement",
    "legitimate_self_preference": "lsp",
}
selected_bias_type_normalized = bias_remap.get(selected_bias_type, selected_bias_type)

# --- Prepare DataFrame for Plotly ---

plot_rows = []
color_map = {
    ('aware', 'CAA'): '#FF563F',
    ('unaware', 'CAA'): '#F5C0B8',
    ('aware', 'Optimization'): '#55C89F',
    ('unaware', 'Optimization'): '#363432',
    ('aware', 'Base'): '#F9DA81',
    ('unaware', 'Base'): '#F9DA81',
    ('unknown', 'Base'): '#d3d3d3',
    ('unknown', 'CAA'): '#e0b0a6',
    ('unknown', 'Optimization'): '#c0c0c0',
}
shape_map = {
    'CAA': 'square',
    'Optimization': 'circle',
    'Base': 'triangle-up',
}

for (aware, method, layer), mult_dict in agg.items():
    multipliers = sorted(mult_dict.keys())
    for m in multipliers:
        vals = mult_dict[m].get(selected_bias_type_normalized, [])
        if not vals:
            continue
        mean = np.mean(np.abs(vals)) if 'bias' in selected_bias_type_normalized else np.mean(vals)
        plot_rows.append({
            'Awareness': aware,
            'Method': method,
            'Layer': layer,
            'Multiplier': m,
            'BiasType': selected_bias_type_normalized,
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
                    marker=dict(symbol=shape_map.get(method, 'square'), color=color_map.get((aware, method), '#cccccc'), size=8),
                    line=dict(color=color_map.get((aware, method), '#cccccc'), width=2),
                ))
        else:
            fig.add_trace(go.Scatter(
                x=group['Multiplier'],
                y=group['MeanProb'],
                mode='lines+markers',
                name=name,
                marker=dict(symbol=shape_map.get(method, 'circle'), color=color_map.get((aware, method), '#cccccc'), size=8),
                line=dict(color=color_map.get((aware, method), '#cccccc'), width=2),
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
            'title': {'text': 'P(self)'},
            'gridcolor': '#f5f5f5', 'linecolor': '#e5dfdf', 'linewidth': 1.5,
            'tickfont': {'color': '#928E8B'}, 'ticksuffix': '   ',
            'range': [0, 1],
        },
        autosize=True,
    )
    fig.update_traces(
        hoverlabel=dict(
            bgcolor='#0c0c0c',
            font_color='#ffffff',
            font_family='Work Sans',
        ),
        hovertemplate='&nbsp;%{x}<br>' + '&nbsp;%{y}<extra></extra>'
    )
    fig.write_image(f"trajectory_steering_plotly_{selected_bias_type}.png", scale=2)
