#!/usr/bin/env bash
# zi2zi-JiT One-Click Training Pipeline
# Usage: bash scripts/train_lora.sh

set -e

echo "=========================================="
echo "  zi2zi-JiT Training Pipeline"
echo "=========================================="

# ============================================
# Configuration
# ============================================

# Font Configuration
SOURCE_FONT="${SOURCE_FONT:-data/base/思源宋体SC-Light.otf}"
TARGET_FONT="${TARGET_FONT:-young}"
TARGET_FONT_DIR="${TARGET_FONT_DIR:-data/target_font}"
CHARSET="${CHARSET:-gb2312}"
TRAIN_CHARS="${TRAIN_CHARS:-500}"
TEST_CHARS="${TEST_CHARS:-100}"

# Dataset Configuration
DATASET_DIR="${DATASET_DIR:-data/${TARGET_FONT}_dataset}"

# Model Configuration
MODEL="${MODEL:-JiT-L/16}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-models/zi2zi-JiT-models/zi2zi-JiT-L-16.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-run/lora_ft_${TARGET_FONT}_L}"

# Training Parameters
EPOCHS="${EPOCHS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
BLR="${BLR:-8e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
SEED="${SEED:-42}"

# Early Stopping Parameters
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-100}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0001}"

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
# SSIM  (Structural Similarity): 结构相似度，范围 0-1，越大越好
# LPIPS (Learned Perceptual Image Patch Similarity): 感知相似度，越小越好
# L1    (Mean Absolute Error): 平均绝对误差，越小越好
# FID   (Fréchet Inception Distance): 特征分布距离，越小越好
EVAL_FREQ="${EVAL_FREQ:-50}"
SAVE_FREQ="${SAVE_FREQ:-50}"
GEN_BSZ="${GEN_BSZ:-8}"
NUM_IMAGES="${NUM_IMAGES:-400}"

# Device
DEVICE="${DEVICE:-cuda}"

echo ""
echo "Configuration:"
echo "  Source Font:    $SOURCE_FONT"
echo "  Target Fonts:   $TARGET_FONT_DIR"
echo "  Charset:        $CHARSET"
echo "  Train Chars:    $TRAIN_CHARS"
echo "  Test Chars:     $TEST_CHARS"
echo ""
echo "  Dataset Dir:    $DATASET_DIR"
echo "  Model:          $MODEL"
echo "  Output Dir:     $OUTPUT_DIR"
echo ""
echo "  Epochs:         $EPOCHS"
echo "  Early Stop:     patience=$EARLY_STOP_PATIENCE, min_delta=$EARLY_STOP_MIN_DELTA"
echo "  Batch Size:     $BATCH_SIZE"
echo "  CFG Scale:      $CFG"
echo "  Device:         $DEVICE"
echo ""

# ============================================
# Step 1: Generate Dataset
# ============================================

echo "=========================================="
echo "Step 1: Generating Dataset"
echo "=========================================="

if [ ! -f "$SOURCE_FONT" ]; then
    echo "ERROR: Source font not found: $SOURCE_FONT"
    exit 1
fi

if [ ! -d "$TARGET_FONT_DIR" ]; then
    echo "ERROR: Target font directory not found: $TARGET_FONT_DIR"
    exit 1
fi

if [ -d "$DATASET_DIR" ]; then
    if [ "$SKIP_CONFIRM" = "true" ]; then
        echo "Dataset already exists at $DATASET_DIR, regenerating..."
        rm -rf "$DATASET_DIR" 2>/dev/null || true
    else
        echo "Dataset already exists at $DATASET_DIR"
        read -p "Do you want to regenerate the dataset? (y/N): " confirm
        if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
            rm -rf "$DATASET_DIR" 2>/dev/null || true
        else
            echo "Using existing dataset."
        fi
    fi
fi

python scripts/generate_font_dataset.py \
    --source-font "$SOURCE_FONT" \
    --font-file "$TARGET_FONT_FILE" \
    --output-dir "$DATASET_DIR" \
    --auto-split \
    --train-ratio 0.8 \
    --charset "$CHARSET"

DATA_PATH="$DATASET_DIR/train/"
TEST_NPZ="$DATASET_DIR/test.npz"

if [ ! -d "$DATA_PATH" ]; then
    echo "ERROR: Training data not found: $DATA_PATH"
    exit 1
fi

if [ ! -f "$TEST_NPZ" ]; then
    echo "WARNING: Test NPZ not found: $TEST_NPZ"
    echo "Creating empty test NPZ from training data..."
    python -c "
import numpy as np
from pathlib import Path
import json

train_dir = Path('$DATA_PATH')
test_npz = Path('$TEST_NPZ')

samples = []
for font_dir in sorted(train_dir.iterdir()):
    if font_dir.is_dir():
        meta_path = font_dir / 'metadata.json'
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            for img_file in sorted(font_dir.glob('*.png'))[:10]:
                samples.append({
                    'source': str(img_file),
                    'target': str(img_file),
                    'char': img_file.stem
                })

if samples:
    print(f'Created fallback test NPZ with {len(samples)} samples')
    np.savez(test_npz, samples=samples)
else:
    print('ERROR: No training samples found')
    exit(1)
"
fi

# ============================================
# Step 2: LoRA Fine-Tuning
# ============================================

echo ""
echo "=========================================="
echo "Step 2: LoRA Fine-Tuning"
echo "=========================================="

if [ ! -f "$BASE_CHECKPOINT" ]; then
    echo "ERROR: Base checkpoint not found: $BASE_CHECKPOINT"
    exit 1
fi

if [ -d "$OUTPUT_DIR" ]; then
    if [ "$SKIP_CONFIRM" = "true" ]; then
        echo "Output directory already exists at $OUTPUT_DIR, removing..."
        rm -rf "$OUTPUT_DIR" 2>/dev/null || true
    else
        echo "Output directory already exists at $OUTPUT_DIR"
        read -p "Do you want to remove and recreate? (y/N): " confirm
        if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
            rm -rf "$OUTPUT_DIR" 2>/dev/null || true
        else
            echo "Using existing directory."
        fi
    fi
fi

mkdir -p "$OUTPUT_DIR"

cat > "$OUTPUT_DIR/config.sh" << EOF
# Training Configuration - Generated $(date)
SOURCE_FONT="$SOURCE_FONT"
TARGET_FONT_DIR="$TARGET_FONT_DIR"
CHARSET="$CHARSET"
TRAIN_CHARS="$TRAIN_CHARS"
TEST_CHARS="$TEST_CHARS"
DATASET_DIR="$DATASET_DIR"
MODEL="$MODEL"
BASE_CHECKPOINT="$BASE_CHECKPOINT"
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
EOF

echo "Configuration saved to: $OUTPUT_DIR/config.sh"
echo ""
echo "Starting training with early stopping..."
echo "=========================================="

python scripts/train_with_early_stopping.py \
    "$OUTPUT_DIR" \
    "$EARLY_STOP_PATIENCE" \
    "$EARLY_STOP_MIN_DELTA" \
    -- \
    python lora_single_gpu_finetune_jit.py \
    --data_path "$DATA_PATH" \
    --test_npz_path "$TEST_NPZ" \
    --output_dir "$OUTPUT_DIR" \
    --base_checkpoint "$BASE_CHECKPOINT" \
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

TRAIN_SUCCESS=$?

# ============================================
# Step 3: Evaluation
# ============================================

echo ""
echo "=========================================="
echo "Step 3: Evaluation"
echo "=========================================="

EVAL_OUTPUT="$OUTPUT_DIR/eval_result"

if [ $TRAIN_SUCCESS -eq 0 ] && [ -f "$OUTPUT_DIR/checkpoint-last.pth" ]; then
    mkdir -p "$EVAL_OUTPUT"
    
    echo "Generating test images..."
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
    
    COMPARE_DIR=$(find "$EVAL_OUTPUT" -type d -name "compare" 2>/dev/null | head -1)
    
    if [ -n "$COMPARE_DIR" ] && [ -d "$COMPARE_DIR" ]; then
        echo ""
        echo "Computing metrics..."
        python scripts/compute_comparison_metrics.py "$COMPARE_DIR" --device "$DEVICE" --batch-size 64
    else
        echo "WARNING: No comparison directory found, skipping metrics."
    fi
else
    echo "WARNING: Training failed or checkpoint not found, skipping evaluation."
fi

# ============================================
# Summary
# ============================================

echo ""
echo "=========================================="
echo "Pipeline completed!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  Dataset:     $DATASET_DIR"
echo "  Output:      $OUTPUT_DIR"
if [ -f "$OUTPUT_DIR/checkpoint-last.pth" ]; then
    echo "  Checkpoint:  $OUTPUT_DIR/checkpoint-last.pth"
fi
echo "  Eval:        $EVAL_OUTPUT"
echo "=========================================="
