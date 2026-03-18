#!/usr/bin/env bash
# zi2zi-JiT Full Pipeline Script
# Usage: bash scripts/step_full.sh <font_name> [options]
# Example: bash scripts/step_full.sh 字魂细体

set -e

echo "=========================================="
echo "  zi2zi-JiT Full Pipeline"
echo "=========================================="

# ============================================
# Parse Arguments
# ============================================

if [ -z "$1" ]; then
    echo "Usage: bash scripts/step_full.sh <font_name> [skip_steps]"
    echo ""
    echo "Arguments:"
    echo "  font_name   - Target font name (without .ttf extension)"
    echo "                Available fonts in data/target_font/:"
    ls -1 data/target_font/*.ttf 2>/dev/null | xargs -I {} basename {} .ttf | sed 's/^/    - /'
    echo ""
    echo "  skip_steps  - Steps to skip (comma-separated: 1,2,3)"
    echo "                1: Train LoRA"
    echo "                2: Generate Images"
    echo "                3: Build TTF Font"
    echo ""
    echo "Examples:"
    echo "  bash scripts/step_full.sh 字魂细体"
    echo "  bash scripts/step_full.sh 字魂细体 1    # Skip training"
    echo "  bash scripts/step_full.sh 字魂细体 1,2  # Skip training and inference"
    exit 1
fi

TARGET_FONT="$1"
SKIP_STEPS="${2:-}"

# Validate font file exists
FONT_FILE="data/target_font/${TARGET_FONT}.ttf"
if [ ! -f "$FONT_FILE" ]; then
    echo "ERROR: Font file not found: $FONT_FILE"
    echo "Available fonts:"
    ls -1 data/target_font/*.ttf 2>/dev/null | xargs -I {} basename {} .ttf | sed 's/^/  - /'
    exit 1
fi

echo ""
echo "Target Font: $TARGET_FONT"
echo "Font File:   $FONT_FILE"
echo "Skip Steps:  ${SKIP_STEPS:-none}"
echo ""

# Parse skip steps
SKIP_TRAIN=false
SKIP_INFER=false
SKIP_BUILD=false

if [ -n "$SKIP_STEPS" ]; then
    IFS=',' read -ra STEPS <<< "$SKIP_STEPS"
    for step in "${STEPS[@]}"; do
        case $step in
            1) SKIP_TRAIN=true ;;
            2) SKIP_INFER=true ;;
            3) SKIP_BUILD=true ;;
        esac
    done
fi

# Export for sub-scripts
export TARGET_FONT
export TARGET_FONT_FILE="$FONT_FILE"

# ============================================
# Step 1: Train LoRA
# ============================================

if [ "$SKIP_TRAIN" = true ]; then
    echo "=========================================="
    echo "Step 1: Train LoRA - SKIPPED"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Step 1: Train LoRA"
    echo "=========================================="
    
    bash scripts/step1_train_lora.sh
fi

# ============================================
# Step 2: Generate Images (Full Charset)
# ============================================

if [ "$SKIP_INFER" = true ]; then
    echo ""
    echo "=========================================="
    echo "Step 2: Generate Images - SKIPPED"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Step 2: Generate Images"
    echo "=========================================="
    
    bash scripts/step2_infer_full.sh
fi

# ============================================
# Step 3: Build TTF Font
# ============================================

if [ "$SKIP_BUILD" = true ]; then
    echo ""
    echo "=========================================="
    echo "Step 3: Build TTF Font - SKIPPED"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Step 3: Build TTF Font"
    echo "=========================================="
    
    bash scripts/step3_build_font.sh
fi

# ============================================
# Summary
# ============================================

echo ""
echo "=========================================="
echo "Full Pipeline Completed!"
echo "=========================================="
echo ""
echo "Target Font: $TARGET_FONT"
echo ""
echo "Outputs:"
echo "  Dataset:    data/${TARGET_FONT}_dataset/"
echo "  Model:      run/lora_ft_${TARGET_FONT}_L/"
echo "  Images:     run/lora_ft_${TARGET_FONT}_L/full_output/"
echo "  Font:       run/lora_ft_${TARGET_FONT}_L/font_output/${TARGET_FONT}_generated.ttf"
echo ""
echo "=========================================="
