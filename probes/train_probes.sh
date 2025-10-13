#!/usr/bin/env bash
# Train linear & MLP probes for each layer file and aggregate best accuracies.
# Requires: Python 3, torch, numpy, your train_probe.py. No jq dependency.

set -euo pipefail

usage() {
  cat <<USAGE
Usage:
  $0 --layers DIR --out DIR [options passed to train_probe.py]

Required:
  --layers DIR        Directory containing per-layer .npz files (e.g., layer_01.npz, layer_02.npz, ...)
  --out DIR           Output directory for probe checkpoints, per-run metrics, and the aggregate JSON

Optional:
  --epochs N          Override epochs for both probes (default: train_probe.py default)
  --batch-size N      Override batch size
  --val-frac F        Validation fraction (default: 0.1)
  --cpu               Force CPU
  --mlp-hidden "H..." Hidden dims for MLP probe, e.g. "1024 256" (default: "512")
  --pooling P         mean|last|max (default: train_probe.py default)
  --tag STR           A tag to add into the aggregate JSON metadata

Any other flags are forwarded to train_probe.py as-is.

Example:
  $0 --layers feats/xsum_layers --out runs/xsum_probes \\
     --epochs 20 --batch-size 64 --pooling mean --mlp-hidden "1024 256" --cpu
USAGE
}

# --- Parse args (minimal parser; keep unknowns to forward) ---
LAYER_DIR=""
OUTDIR=""
MLP_HIDDEN="512"
TAG=""
FORWARD_ARGS=()

while (( "$#" )); do
  case "$1" in
    --layers)     LAYER_DIR="${2:-}"; shift 2 ;;
    --out)        OUTDIR="${2:-}"; shift 2 ;;
    --mlp-hidden) MLP_HIDDEN="${2:-}"; shift 2 ;;
    --tag)        TAG="${2:-}"; shift 2 ;;
    -h|--help)    usage; exit 0 ;;
    *)
      FORWARD_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "${LAYER_DIR}" || -z "${OUTDIR}" ]]; then
  echo "[ERROR] --layers and --out are required." >&2
  usage
  exit 1
fi

mkdir -p "${OUTDIR}"

# Find and sort layer files
mapfile -t LAYER_FILES < <(ls "${LAYER_DIR}"/layer_*.npz 2>/dev/null | sort)
if [[ ${#LAYER_FILES[@]} -eq 0 ]]; then
  echo "[ERROR] No files matching ${LAYER_DIR}/layer_*.npz" >&2
  exit 1
fi

echo "[INFO] Found ${#LAYER_FILES[@]} layer files."

# Helper to run one probe
run_probe () {
  local probe="$1"     # "linear" or "mlp"
  local data_path="$2" # layer_XY.npz
  local base="$(basename "${data_path}" .npz)"
  local stem="${OUTDIR}/${base}_${probe}"
  local save_path="${stem}.pt"

  # Build args for MLP hidden dims
  local extra_mlp_args=()
  if [[ "${probe}" == "mlp" ]]; then
    # expand MLP_HIDDEN into args: --hidden-dims H1 H2 ...
    read -r -a HARR <<< "${MLP_HIDDEN}"
    extra_mlp_args=( --hidden-dims "${HARR[@]}" )
  fi

  echo "[RUN] ${probe} :: ${data_path}"
  python3 probes/train_probe.py \
    --data "${data_path}" \
    --save-path "${save_path}" \
    --probe "${probe}" \
    "${extra_mlp_args[@]}" \
    "${FORWARD_ARGS[@]}"

  local metrics_json="${stem}_metrics.json"
  if [[ ! -f "${metrics_json}" ]]; then
    echo "[WARN] Metrics JSON not found: ${metrics_json}"
  fi
}

# Run both probes for each layer
for f in "${LAYER_FILES[@]}"; do
  run_probe "linear" "${f}"
  run_probe "mlp"    "${f}"
done

# Aggregate best validation accuracies into one JSON
AGG_JSON="${OUTDIR}/aggregate_best_acc.json"
echo "[INFO] Writing aggregate JSON to ${AGG_JSON}"

python3 - <<PY
import os, glob, json, time, sys
outdir = r"${OUTDIR}"
tag = r"${TAG}"

def load_best_acc(path):
    try:
        with open(path, "r") as f:
            obj = json.load(f)
        # train_probe.py stores: {"loss":..., "acc":..., "auc":..., "epoch":..., ...}
        return float(obj.get("acc", float("nan")))
    except Exception:
        return float("nan")

# Map: layer_basename -> {linear_acc, mlp_acc}
layers = {}
for metrics_path in glob.glob(os.path.join(outdir, "layer_*_linear_metrics.json")):
    base = os.path.basename(metrics_path).replace("_linear_metrics.json","")
    layers.setdefault(base, {})
    layers[base]["linear_acc"] = load_best_acc(metrics_path)

for metrics_path in glob.glob(os.path.join(outdir, "layer_*_mlp_metrics.json")):
    base = os.path.basename(metrics_path).replace("_mlp_metrics.json","")
    layers.setdefault(base, {})
    layers[base]["mlp_acc"] = load_best_acc(metrics_path)

# Sort keys numerically if possible
def keynum(k):
    # expect "layer_XX"
    try:
        return int(k.split("_")[1])
    except Exception:
        return k

sorted_items = sorted(layers.items(), key=lambda kv: keynum(kv[0]))

report = {
    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    "tag": tag or None,
    "layers": [
        {"layer": name, **vals}
        for name, vals in sorted_items
    ],
    "summary": {
        "best_linear": None,
        "best_mlp": None
    }
}

# Compute bests
best_lin = max(((d.get("linear_acc", float("nan")), d["layer"]) for d in report["layers"]), default=(None,None))
best_mlp = max(((d.get("mlp_acc", float("nan")), d["layer"]) for d in report["layers"]), default=(None,None))

if best_lin[0] == best_lin[0]:  # not NaN
    report["summary"]["best_linear"] = {"layer": best_lin[1], "acc": best_lin[0]}
if best_mlp[0] == best_mlp[0]:
    report["summary"]["best_mlp"] = {"layer": best_mlp[1], "acc": best_mlp[0]}

with open(os.path.join(outdir, "aggregate_best_acc.json"), "w") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report["summary"], indent=2))
PY

echo "[DONE] Aggregate saved to ${AGG_JSON}"
