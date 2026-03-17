#!/usr/bin/env bash
set -euo pipefail

echo "[zi2zi-JiT] Full Evaluation Pipeline"

# Configuration
SOURCE_FONT="${SOURCE_FONT:-data/simhei.ttf}"
TARGET_FONT="${TARGET_FONT:-data/young_font/young.ttf}"
OUTPUT_BASE="${OUTPUT_BASE:-run/eval_full}"
CHECKPOINT="${CHECKPOINT:-run/lora_ft_young/checkpoint-last.pth}"

# Dataset parameters
TRAIN_CHARS="${TRAIN_CHARS:-500}"
TEST_CHARS="${TEST_CHARS:-100}"
CHARSET="${CHARSET:-gb2312}"

# Generation parameters
BATCH_SIZE="${BATCH_SIZE:-64}"
DEVICE="${DEVICE:-cuda}"
SAMPLING_METHOD="${SAMPLING_METHOD:-ab2}"
NUM_STEPS="${NUM_STEPS:-20}"
CFG_SCALE="${CFG_SCALE:-2.6}"

echo "[zi2zi-JiT] Configuration:"
echo "  Source Font:    $SOURCE_FONT"
echo "  Target Font:    $TARGET_FONT"
echo "  Output Base:    $OUTPUT_BASE"
echo "  Checkpoint:     $CHECKPOINT"
echo "  Train Chars:    $TRAIN_CHARS"
echo "  Test Chars:     $TEST_CHARS"
echo "  Charset:        $CHARSET"
echo "  Sampling:       $SAMPLING_METHOD ($NUM_STEPS steps)"
echo "  CFG Scale:      $CFG_SCALE"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "[zi2zi-JiT] Error: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

# Check if target font exists
if [ ! -f "$TARGET_FONT" ]; then
    echo "[zi2zi-JiT] Error: Target font not found: $TARGET_FONT"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Step 1: Generate test dataset with more characters
echo ""
echo "[zi2zi-JiT] Step 1: Generating test dataset..."
TEST_FONT_DIR="$OUTPUT_BASE/test_font"
TEST_DATASET_DIR="$OUTPUT_BASE/test_dataset"
mkdir -p "$TEST_FONT_DIR"

# Copy target font to test font directory
FONT_NAME=$(basename "$TARGET_FONT" .ttf)
cp "$TARGET_FONT" "$TEST_FONT_DIR/${FONT_NAME}.ttf"

# Generate test dataset
python scripts/generate_font_dataset.py \
    --source-font "$SOURCE_FONT" \
    --font-dir "$TEST_FONT_DIR" \
    --output-dir "$TEST_DATASET_DIR" \
    --train-chars-per-font "$TRAIN_CHARS" \
    --test-chars-per-font "$TEST_CHARS" \
    --charset "$CHARSET" \
    --test-only \
    --train-dir "data/young_dataset/train"

# Step 2: Run inference with pairwise output
echo ""
echo "[zi2zi-JiT] Step 2: Running inference..."
python generate_chars.py \
    --checkpoint "$CHECKPOINT" \
    --test_npz "$TEST_DATASET_DIR/test.npz" \
    --output_dir "$OUTPUT_BASE/inference" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --sampling_method "$SAMPLING_METHOD" \
    --num_sampling_steps "$NUM_STEPS" \
    --cfg "$CFG_SCALE" \
    --pairwise target_gen

# Step 3: Compute metrics
echo ""
echo "[zi2zi-JiT] Step 3: Computing metrics..."
COMPARE_DIR=$(find "$OUTPUT_BASE/inference" -type d -name "compare" | head -1)

if [ -z "$COMPARE_DIR" ]; then
    echo "[zi2zi-JiT] Error: No compare folder found"
    exit 1
fi

python scripts/compute_pairwise_metrics.py "$COMPARE_DIR" --device "$DEVICE" --batch-size "$BATCH_SIZE"

echo ""
echo "[zi2zi-JiT] Full evaluation completed"
echo "[zi2zi-JiT] Results saved to: $OUTPUT_BASE"
