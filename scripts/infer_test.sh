#!/usr/bin/env bash
set -euo pipefail

echo "[zi2zi-JiT] Test Inference and Evaluation"

# Configuration
CHECKPOINT="${CHECKPOINT:-run/lora_ft_young/checkpoint-last.pth}"
TEST_NPZ="${TEST_NPZ:-data/young_dataset/test.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-run/eval_test}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DEVICE="${DEVICE:-cuda}"

# Sampling parameters
SAMPLING_METHOD="${SAMPLING_METHOD:-ab2}"
NUM_STEPS="${NUM_STEPS:-20}"
CFG_SCALE="${CFG_SCALE:-2.6}"

echo "[zi2zi-JiT] Configuration:"
echo "  Checkpoint:     $CHECKPOINT"
echo "  Test NPZ:       $TEST_NPZ"
echo "  Output Dir:     $OUTPUT_DIR"
echo "  Batch Size:     $BATCH_SIZE"
echo "  Device:         $DEVICE"
echo "  Sampling:       $SAMPLING_METHOD ($NUM_STEPS steps)"
echo "  CFG Scale:      $CFG_SCALE"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "[zi2zi-JiT] Error: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

# Check if test NPZ exists
if [ ! -f "$TEST_NPZ" ]; then
    echo "[zi2zi-JiT] Error: Test NPZ not found: $TEST_NPZ"
    exit 1
fi

# Run inference with pairwise output (target|generated)
echo ""
echo "[zi2zi-JiT] Running inference..."
python generate_chars.py \
    --checkpoint "$CHECKPOINT" \
    --test_npz "$TEST_NPZ" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --sampling_method "$SAMPLING_METHOD" \
    --num_sampling_steps "$NUM_STEPS" \
    --cfg "$CFG_SCALE" \
    --pairwise target_gen

# Find the generated compare folder
COMPARE_DIR=$(find "$OUTPUT_DIR" -type d -name "compare" | head -1)

if [ -z "$COMPARE_DIR" ]; then
    echo "[zi2zi-JiT] Error: No compare folder found in $OUTPUT_DIR"
    exit 1
fi

echo ""
echo "[zi2zi-JiT] Running evaluation on: $COMPARE_DIR"
python scripts/compute_comparison_metrics.py "$COMPARE_DIR" --device "$DEVICE" --batch-size "$BATCH_SIZE"

echo ""
echo "[zi2zi-JiT] Test inference and evaluation completed"
echo "[zi2zi-JiT] Results saved to: $OUTPUT_DIR"
