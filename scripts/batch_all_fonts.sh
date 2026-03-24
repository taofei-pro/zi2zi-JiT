#!/usr/bin/env bash
# zi2zi-JiT Batch Processing Script
# Sequentially process all target fonts

set -e

echo "=========================================="
echo "  zi2zi-JiT Batch Font Generation"
echo "=========================================="

# All target fonts
FONTS=(
    "young"
    "1260-Regular"
    "字魂细体"
    "颜体刻本"
)

# Force regeneration even if font exists
FORCE_REGEN="${FORCE_REGEN:-false}"

TOTAL=${#FONTS[@]}
SUCCESS=0
FAILED=0
FAILED_FONTS=()

echo ""
echo "Fonts to process: $TOTAL"
for font in "${FONTS[@]}"; do
    echo "  - $font"
done
echo ""

START_TIME=$(date +%s)

for i in "${!FONTS[@]}"; do
    FONT="${FONTS[$i]}"
    NUM=$((i + 1))
    
    # Check if font already generated successfully
    FONT_PATH="run/lora_ft_${FONT}_L/font_output/${FONT}_generated.ttf"
    if [ -f "$FONT_PATH" ] && [ "$FORCE_REGEN" != "true" ]; then
        echo ""
        echo "=========================================="
        echo "[$NUM/$TOTAL] $FONT - already exists, skipping"
        echo "=========================================="
        echo ""
        ((SUCCESS++)) || true
        continue
    fi
    
    echo ""
    echo "=========================================="
    echo "[$NUM/$TOTAL] Processing: $FONT"
    echo "=========================================="
    echo ""

    # Clean up previous runs
    DATASET_DIR="data/${FONT}_dataset"
    OUTPUT_DIR="run/lora_ft_${FONT}_L"
    rm -rf "$DATASET_DIR" "$OUTPUT_DIR" 2>/dev/null || true
    
    FONT_START=$(date +%s)
    
    # Export for sub-scripts
    export TARGET_FONT="$FONT"
    export SKIP_CONFIRM=true
    
    if bash scripts/step_full.sh "$FONT"; then
        FONT_END=$(date +%s)
        FONT_DURATION=$((FONT_END - FONT_START))
        echo ""
        echo "[$NUM/$TOTAL] $FONT completed in ${FONT_DURATION}s"
        ((SUCCESS++)) || true
    else
        FONT_END=$(date +%s)
        FONT_DURATION=$((FONT_END - FONT_START))
        echo ""
        echo "[$NUM/$TOTAL] $FONT FAILED after ${FONT_DURATION}s"
        FAILED_FONTS+=("$FONT")
        ((FAILED++)) || true
    fi
done

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

echo ""
echo "=========================================="
echo "Batch Processing Completed!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  Total:     $TOTAL"
echo "  Success:   $SUCCESS"
echo "  Failed:    $FAILED"
echo "  Duration:  ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""

if [ ${#FAILED_FONTS[@]} -gt 0 ]; then
    echo "Failed fonts:"
    for font in "${FAILED_FONTS[@]}"; do
        echo "  - $font"
    done
    echo ""
fi

echo "Generated fonts location:"
for font in "${FONTS[@]}"; do
    FONT_PATH="run/lora_ft_${font}_L/font_output/${font}_generated.ttf"
    if [ -f "$FONT_PATH" ]; then
        SIZE=$(du -h "$FONT_PATH" | cut -f1)
        echo "  [OK] $font: $FONT_PATH ($SIZE)"
    else
        echo "  [--] $font: not generated"
    fi
done

echo ""
echo "=========================================="

if [ $FAILED -gt 0 ]; then
    exit 1
fi
