#!/usr/bin/env bash
# zi2zi-JiT LoRA Fine-Tuning Script
# Usage: bash scripts/train_lora.sh [options]

set -e

echo "=========================================="
echo "  zi2zi-JiT LoRA Fine-Tuning"
echo "=========================================="

# Default Configuration
MODEL="${MODEL:-JiT-L/16}"
CHECKPOINT="${CHECKPOINT:-models/zi2zi-JiT-models/zi2zi-JiT-L-16.pth}"
DATA_PATH="${DATA_PATH:-data/young_dataset_v2/train/}"
TEST_NPZ="${TEST_NPZ:-data/young_dataset_v2/test.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-run/lora_ft_young_L_$(date +%Y%m%d_%H%M%S)}"

# Training Parameters
EPOCHS="${EPOCHS:-500}"
BATCH_SIZE="${BATCH_SIZE:-8}"
BLR="${BLR:-8e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
SEED="${SEED:-42}"

# LoRA Parameters
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_TARGETS="${LORA_TARGETS:-qkv,proj,w12,w3}"

# Model Parameters
NUM_FONTS="${NUM_FONTS:-1000}"
NUM_CHARS="${NUM_CHARS:-20000}"
MAX_CHARS_PER_FONT="${MAX_CHARS_PER_FONT:-500}"

# Sampling Parameters
CFG="${CFG:-2.4}"
SAMPLING_METHOD="${SAMPLING_METHOD:-heun}"
NUM_SAMPLING_STEPS="${NUM_SAMPLING_STEPS:-50}"

# Evaluation Parameters
EVAL_FREQ="${EVAL_FREQ:-50}"
SAVE_FREQ="${SAVE_FREQ:-50}"
GEN_BSZ="${GEN_BSZ:-8}"
NUM_IMAGES="${NUM_IMAGES:-400}"

# Device
DEVICE="${DEVICE:-cuda}"

# Device
DEVICE="${DEVICE:-cuda}"

echo ""
echo "Configuration:"
echo "  Model:           $MODEL"
echo "  Checkpoint:      $CHECKPOINT"
echo "  Data Path:       $DATA_PATH"
echo "  Test NPZ:        $TEST_NPZ"
echo "  Output Dir:      $OUTPUT_DIR"
echo ""
echo "Training:"
echo "  Epochs:          $EPOCHS"
echo "  Batch Size:      $BATCH_SIZE"
echo "  Learning Rate:   $BLR"
echo "  Warmup Epochs:   $WARMUP_EPOCHS"
echo "  Seed:            $SEED"
echo ""
echo "LoRA:"
echo "  Rank:            $LORA_R"
echo "  Alpha:           $LORA_ALPHA"
echo "  Targets:         $LORA_TARGETS"
echo ""
echo "Sampling:"
echo "  CFG Scale:       $CFG"
echo "  Method:          $SAMPLING_METHOD"
echo "  Steps:           $NUM_SAMPLING_STEPS"
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo "Please download the model first or set CHECKPOINT environment variable."
    exit 1
fi

# Check if data exists
if [ ! -d "$DATA_PATH" ]; then
    echo "ERROR: Data path not found: $DATA_PATH"
    echo "Please generate the dataset first or set DATA_PATH environment variable."
    exit 1
fi

if [ ! -f "$TEST_NPZ" ]; then
    echo "ERROR: Test NPZ not found: $TEST_NPZ"
    echo "Please generate the dataset first or set TEST_NPZ environment variable."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Save configuration
cat > "$OUTPUT_DIR/config.sh" << EOF
# Training Configuration
MODEL="$MODEL"
CHECKPOINT="$CHECKPOINT"
DATA_PATH="$DATA_PATH"
TEST_NPZ="$TEST_NPZ"
OUTPUT_DIR="$OUTPUT_DIR"
EPOCHS="$EPOCHS"
BATCH_SIZE="$BATCH_SIZE"
BLR="$BLR"
WARMUP_EPOCHS="$WARMUP_EPOCHS"
SEED="$SEED"
LORA_R="$LORA_R"
LORA_ALPHA="$LORA_ALPHA"
LORA_TARGETS="$LORA_TARGETS"
CFG="$CFG"
SAMPLING_METHOD="$SAMPLING_METHOD"
NUM_SAMPLING_STEPS="$NUM_SAMPLING_STEPS"
EOF

echo "Configuration saved to: $OUTPUT_DIR/config.sh"
echo ""
echo "Starting training..."
echo "=========================================="

python lora_single_gpu_finetune_jit.py \
    --data_path "$DATA_PATH" \
    --test_npz_path "$TEST_NPZ" \
    --output_dir "$OUTPUT_DIR" \
    --base_checkpoint "$CHECKPOINT" \
    --model "$MODEL" \
    --num_fonts "$NUM_FONTS" \
    --num_chars "$NUM_CHARS" \
    --max_chars_per_font "$MAX_CHARS_PER_FONT" \
    --img_size 256 \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_targets "$LORA_TARGETS" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --blr "$BLR" \
    --warmup_epochs "$WARMUP_EPOCHS" \
    --save_last_freq "$SAVE_FREQ" \
    --proj_dropout 0.1 \
    --P_mean -0.8 \
    --P_std 0.8 \
    --noise_scale 1.0 \
    --cfg "$CFG" \
    --sampling_method "$SAMPLING_METHOD" \
    --num_sampling_steps "$NUM_SAMPLING_STEPS" \
    --online_eval \
    --eval_step_folders \
    --eval_freq "$EVAL_FREQ" \
    --gen_bsz "$GEN_BSZ" \
    --num_images "$NUM_IMAGES" \
    --seed "$SEED" \
    --device "$DEVICE"

echo ""
echo "=========================================="
echo "Training completed!"
echo "Output saved to: $OUTPUT_DIR"
echo "Checkpoint: $OUTPUT_DIR/checkpoint-last.pth"

echo ""
echo "=========================================="
echo "Running evaluation..."
echo "=========================================="

EVAL_OUTPUT="$OUTPUT_DIR/eval_results"
mkdir -p "$EVAL_OUTPUT"

python generate_chars.py \
    --checkpoint "$OUTPUT_DIR/checkpoint-last.pth" \
    --test_npz "$TEST_NPZ" \
    --output_dir "$EVAL_OUTPUT" \
    --batch_size 64 \
    --device "$DEVICE" \
    --sampling_method ab2 \
    --num_sampling_steps 20 \
    --cfg "$CFG" \
    --pairwise target_gen

COMPARE_DIR=$(find "$EVAL_OUTPUT" -type d -name "compare" | head -1)

if [ -n "$COMPARE_DIR" ]; then
    echo ""
    echo "Computing metrics..."
    python scripts/compute_comparison_metrics.py "$COMPARE_DIR" --device "$DEVICE" --batch-size 64
fi

echo ""
echo "=========================================="
echo "All done!"
echo "Training output: $OUTPUT_DIR"
echo "Evaluation results: $EVAL_OUTPUT"
echo "=========================================="
