#!/usr/bin/env bash
# Collect per-layer last-5-token activations into separate .npz files
# Requires: Python, transformers, torch, numpy, your collect_last5_local_llama31_multi.py

set -euo pipefail

# --- Helpers ---
usage() {
  cat <<USAGE
Usage:
  $0 --outdir OUTDIR [--model HF_MODEL_ID] [other args passed to collect_last5_local_llama31_multi.py]
Required:
  --outdir OUTDIR           Directory to write layer-wise .npz files

Optional (auto-detected or defaults to your Python script's default):
  --model HF_MODEL_ID       e.g. meta-llama/Meta-Llama-3.1-8B-Instruct

All other flags/args are forwarded to collect_last5_local_llama31_multi.py.
Example:
  $0 --outdir feats/xsum_layers \
     --dataset xsum --data-type sources \
     --pos-jsonl sp_train.jsonl --neg-jsonl nonsp_train.jsonl \
     --prompt-template "Summarize the following article:\n{text}\nSummary:" \
     --keep-duplicates --batch-size 8 --max-len 512
USAGE
}

# Parse --outdir and --model; forward everything else.
OUTDIR=""
MODEL=""
FORWARD_ARGS=()

while (( "$#" )); do
  case "$1" in
    --outdir)
      OUTDIR="${2:-}"; shift 2 ;;
    --model)
      MODEL="${2:-}"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      FORWARD_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "${OUTDIR}" ]]; then
  echo "[ERROR] --outdir is required" >&2
  usage
  exit 1
fi

mkdir -p "$OUTDIR"

# If user didn't pass --model here, we try to detect it from forwarded args.
if [[ -z "${MODEL}" ]]; then
  # scan FORWARD_ARGS for --model
  for ((i=0; i<${#FORWARD_ARGS[@]}; i++)); do
    if [[ "${FORWARD_ARGS[$i]}" == "--model" ]]; then
      if (( i+1 < ${#FORWARD_ARGS[@]} )); then
        MODEL="${FORWARD_ARGS[$((i+1))]}"
      fi
      break
    fi
  done
fi

# Fallback to your script's default if still empty
if [[ -z "${MODEL}" ]]; then
  MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
fi

TMP_ALL="${OUTDIR}/_ALL_layers_concat.npz"

echo "[INFO] Running base collector once with --layer-agg concat ..."
python3 probes/collect_last5_xsum.py \
  --layer-agg concat \
  --out "${TMP_ALL}" \
  "${FORWARD_ARGS[@]}"

echo "[INFO] Splitting concatenated features into per-layer files using model: ${MODEL}"

python3 - <<PY
import numpy as np
import os, sys
from transformers import AutoConfig

all_path = r"${TMP_ALL}"
outdir = r"${OUTDIR}"
model_id = r"${MODEL}"

if not os.path.exists(all_path):
    raise SystemExit(f"[ERROR] Missing combined NPZ: {all_path}")

cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
hidden_size = getattr(cfg, "hidden_size", None)
num_layers  = getattr(cfg, "num_hidden_layers", None)

if hidden_size is None or num_layers is None:
    raise SystemExit("[ERROR] Could not read hidden_size/num_hidden_layers from model config")

data = np.load(all_path, allow_pickle=True)
X_all = data["x"]            # shape: (N, 5, L*D)
Y     = data["y"]
M     = data["mask"]
IDS   = data["ids"]

N, K, LD = X_all.shape
D = hidden_size
L = LD // D

if L != num_layers:
    print(f"[WARN] Inferred L={L} from array, config says num_hidden_layers={num_layers}. Proceeding with L={L}.")

if LD % D != 0:
    raise SystemExit(f"[ERROR] Last dimension {LD} not divisible by hidden_size {D}")

for l in range(L):
    start = l * D
    end   = (l + 1) * D
    X_l = X_all[:, :, start:end]  # (N, 5, D)
    out_path = os.path.join(outdir, f"layer_{l+1:02d}.npz")  # 1-indexed to match hidden_states[1:]
    np.savez(out_path, x=X_l, y=Y, mask=M, ids=IDS)
    print(f"[OK] Wrote {out_path} with x={X_l.shape}, y={Y.shape}, mask={M.shape}, ids={IDS.shape}")

print("[DONE] Per-layer files ready in:", outdir)
PY

echo "[CLEANUP] Keeping combined file at ${TMP_ALL} for reproducibility."
echo "[SUCCESS] Layer-wise activations saved in ${OUTDIR}"