#!/bin/bash
#SBATCH --job-name=gemma_full
#SBATCH --partition=general
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --output=logs/gemma_full_%j.log
#SBATCH --error=logs/gemma_full_%j.log
#SBATCH --mem=800G

#!/bin/bash
#
# Run CAA steering analysis pipeline for Google Gemma-3-27B
#
# Usage:
#   ./run_gemma_steering.sh [--quantize] [--skip-extraction] [--skip-vector] [--skip-eval]

set -e  # Exit on error
source .venv/bin/activate
# Configuration
MODEL_NAME="google_gemma-3-27b-it"
MODEL_ID="google/gemma-3-27b-it"
ARENA_DATA="results_arena_sample/google_gemma-3-27b-it/arena/output.jsonl"
OUTPUT_BASE="steering_analysis/google_gemma-3-27b-it"

# Default parameters
N_EXAMPLES=50
LAYERS="16 17 18"  # Adjust for Gemma's layer count
MULTIPLIERS="-1.0 -0.5 -0.3 0.0 0.3 0.5 1.0"
N_TEST=100
OFFSET=10
MAX_NEW_TOKENS=20

# Parse command-line arguments
QUANTIZE=""
SKIP_FLAGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --quantize)
            QUANTIZE="--quantize"
            shift
            ;;
        --skip-extraction)
            SKIP_FLAGS="$SKIP_FLAGS --skip_extraction"
            shift
            ;;
        --skip-vector)
            SKIP_FLAGS="$SKIP_FLAGS --skip_vector_creation"
            shift
            ;;
        --skip-eval)
            SKIP_FLAGS="$SKIP_FLAGS --skip_evaluation"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--quantize] [--skip-extraction] [--skip-vector] [--skip-eval]"
            exit 1
            ;;
    esac
done

echo "========================================="
echo "Gemma-3-27B CAA Steering Analysis Pipeline"
echo "========================================="
echo ""
echo "Model:       $MODEL_NAME"
echo "Model ID:    $MODEL_ID"
echo "Arena data:  $ARENA_DATA"
echo "Output:      $OUTPUT_BASE"
echo "N examples:  $N_EXAMPLES"
echo "Layers:      $LAYERS"
echo "Multipliers: $MULTIPLIERS"
echo "N test:      $N_TEST"
echo "Quantize:    $([ -n "$QUANTIZE" ] && echo "Yes" || echo "No")"
echo ""

# Check if arena data exists
if [ ! -f "$ARENA_DATA" ]; then
    echo "Error: Arena data file not found: $ARENA_DATA"
    exit 1
fi

# Run pipeline
python3 run_arena_steering_pipeline.py \
    --model_name "$MODEL_NAME" \
    --model_id "$MODEL_ID" \
    --arena_data "$ARENA_DATA" \
    --output_base "$OUTPUT_BASE" \
    --n_examples $N_EXAMPLES \
    --layers $LAYERS \
    --multipliers $MULTIPLIERS \
    --n_test $N_TEST \
    --offset $OFFSET \
    --max_new_tokens $MAX_NEW_TOKENS \
    $QUANTIZE \
    $SKIP_FLAGS

echo ""
echo "========================================="
echo "Pipeline completed successfully!"
echo "========================================="
echo ""
echo "Results available at: $OUTPUT_BASE"
echo ""
echo "Next steps:"
echo "  1. Check steering examples: $OUTPUT_BASE/steering_examples/"
echo "  2. Review steering vector: $OUTPUT_BASE/steering_vector/"
echo "  3. Analyze evaluation results: $OUTPUT_BASE/evaluation/"
echo "  4. View distribution plots: $OUTPUT_BASE/evaluation/*.png"
echo ""
