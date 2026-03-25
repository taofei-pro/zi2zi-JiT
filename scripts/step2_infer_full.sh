#!/usr/bin/env bash
# zi2zi-JiT Full Charset Inference Script
# Usage: bash scripts/infer_full.sh

set -e

echo "=========================================="
echo "  zi2zi-JiT Full Charset Inference"
echo "=========================================="

# ============================================
# Configuration
# ============================================

# Model Configuration
TARGET_FONT="${TARGET_FONT:-young}"
CHECKPOINT="${CHECKPOINT:-run/lora_ft_${TARGET_FONT}_L/checkpoint-last.pth}"

# Source Font (for content images)
SOURCE_FONT="${SOURCE_FONT:-data/base/思源宋体SC-Light.otf}"

# Reference images directory (for style guidance)
REF_DIR="${REF_DIR:-data/dataset/${TARGET_FONT}/train/001_${TARGET_FONT}}"

# Charset file (one character per line)
CHARSET_FILE="${CHARSET_FILE:-data/base/target_charset.txt}"

# Output Configuration
OUTPUT_DIR="${OUTPUT_DIR:-run/lora_ft_${TARGET_FONT}_L/full_output}"

# Generation Parameters
BATCH_SIZE="${BATCH_SIZE:-64}"
CFG="${CFG:-2.4}"
SAMPLING_METHOD="${SAMPLING_METHOD:-ab2}"
NUM_STEPS="${NUM_STEPS:-20}"

# Resolution
RESOLUTION="${RESOLUTION:-256}"
REF_SIZE="${REF_SIZE:-128}"
UP_SCALE="${UP_SCALE:-2}"  # 超分辨率放大倍数 (1=不放大, 2=256->512)

# Device
DEVICE="${DEVICE:-cuda}"

echo ""
echo "Configuration:"
echo "  Checkpoint:   $CHECKPOINT"
echo "  Source Font:  $SOURCE_FONT"
echo "  Reference:    $REF_DIR"
echo "  Charset:      $CHARSET_FILE"
echo "  Output:       $OUTPUT_DIR"
echo ""
echo "  Batch Size:   $BATCH_SIZE"
echo "  Sampling:     $SAMPLING_METHOD ($NUM_STEPS steps)"
echo "  CFG Scale:    $CFG"
echo "  Device:       $DEVICE"
echo ""

# ============================================
# Step 1: Create Test NPZ
# ============================================

echo "=========================================="
echo "Step 1: Creating test NPZ for full charset"
echo "=========================================="

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

if [ ! -f "$SOURCE_FONT" ]; then
    echo "ERROR: Source font not found: $SOURCE_FONT"
    exit 1
fi

if [ ! -d "$REF_DIR" ]; then
    echo "ERROR: Reference directory not found: $REF_DIR"
    exit 1
fi

if [ ! -f "$CHARSET_FILE" ]; then
    echo "ERROR: Charset file not found: $CHARSET_FILE"
    exit 1
fi

# Count characters
CHAR_COUNT=$(wc -l < "$CHARSET_FILE")
echo "Total characters in charset: $CHAR_COUNT"

# Create output directory
if [ -d "$OUTPUT_DIR" ]; then
    echo "Output directory already exists at $OUTPUT_DIR, removing..."
    rm -rf "$OUTPUT_DIR"
fi

mkdir -p "$OUTPUT_DIR"

# Create test NPZ using Python
TEST_NPZ="$OUTPUT_DIR/full_charset.npz"

python - << PYTHON_SCRIPT
import numpy as np
from pathlib import Path
from PIL import Image
import sys

charset_path = "$CHARSET_FILE"
source_font_path = "$SOURCE_FONT"
ref_dir = "$REF_DIR"
output_path = "$TEST_NPZ"
ref_size = $REF_SIZE
resolution = $RESOLUTION

with open(charset_path, 'r', encoding='utf-8') as f:
    chars = [line.strip() for line in f if line.strip()]

codepoints = [ord(c) for c in chars]
n = len(codepoints)

print(f"Total characters: {n}")

sys.path.insert(0, '.')
from data_processing.font_utils import GlyphRenderer

source_renderer = GlyphRenderer(source_font_path, resolution)

ref_images = sorted(Path(ref_dir).glob('*.jpg'))
if not ref_images:
    ref_images = sorted(Path(ref_dir).glob('*.png'))

if not ref_images:
    raise ValueError(f"No reference images found in {ref_dir}")

print(f"Found {len(ref_images)} reference images")

ref_img = Image.open(ref_images[0]).convert('RGB')
if ref_img.size == (1024, 256):
    ref_crop = ref_img.crop((512, 0, 640, 128))
else:
    ref_crop = ref_img

ref_array = np.array(ref_crop.resize((ref_size, ref_size), Image.LANCZOS)).transpose(2, 0, 1)

font_labels = np.zeros(n, dtype=np.int64)
char_labels = np.arange(n, dtype=np.int64)
unicode_labels = np.array(codepoints, dtype=np.int64)
content_images = np.empty((n, 3, resolution, resolution), dtype=np.uint8)
style_images = np.empty((n, 3, ref_size, ref_size), dtype=np.uint8)

failed = 0
for i, cp in enumerate(codepoints):
    try:
        source_img = source_renderer.render(cp)
        if source_img is None:
            content_images[i] = 255
            failed += 1
        else:
            content_images[i] = np.array(source_img).transpose(2, 0, 1)
    except Exception as e:
        content_images[i] = 255
        failed += 1
    
    style_images[i] = ref_array
    
    if (i + 1) % 1000 == 0:
        print(f"  Processed {i + 1}/{n}")

print(f"Processed {n} characters, {failed} failed")

np.savez_compressed(
    output_path,
    font_labels=font_labels,
    char_labels=char_labels,
    unicode_labels=unicode_labels,
    content_images=content_images,
    style_images=style_images,
    num_original_samples=np.int64(n),
)

file_size = Path(output_path).stat().st_size / (1024 * 1024)
print(f"Saved to {output_path} ({file_size:.1f} MB)")
PYTHON_SCRIPT

# ============================================
# Step 2: Generate images
# ============================================

echo ""
echo "=========================================="
echo "Step 2: Generating images"
echo "=========================================="

python generate_chars.py \
    --checkpoint "$CHECKPOINT" \
    --test_npz "$TEST_NPZ" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --sampling_method "$SAMPLING_METHOD" \
    --num_sampling_steps "$NUM_STEPS" \
    --cfg "$CFG"

# ============================================
# Step 3: Upscale images (Super Resolution)
# ============================================

GEN_DIR=$(find "$OUTPUT_DIR" -type d -name "generated" 2>/dev/null | head -1)

if [ "$UP_SCALE" -gt 1 ] && [ -n "$GEN_DIR" ]; then
    echo ""
    echo "=========================================="
    echo "Step 3: Upscaling images (${UP_SCALE}x)"
    echo "=========================================="
    
    UPSCALED_DIR="${GEN_DIR}_upscaled"
    mkdir -p "$UPSCALED_DIR"
    
    python - << PYTHON_SCRIPT
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

gen_dir = Path("$GEN_DIR")
upscaled_dir = Path("$UPSCALED_DIR")
up_scale = int("$UP_SCALE")

# 使用高质量Lanczos插值放大
img_paths = sorted(gen_dir.glob("*.png"))
print(f"Upscaling {len(img_paths)} images by {up_scale}x...")

for img_path in tqdm(img_paths, desc="Upscaling"):
    try:
        img = Image.open(img_path).convert('RGB')
        new_size = (img.width * up_scale, img.height * up_scale)
        # 使用LANCZOS (Lanczos插值) 进行高质量放大
        img_up = img.resize(new_size, Image.LANCZOS)
        img_up.save(upscaled_dir / img_path.name, quality=95)
    except Exception as e:
        print(f"  Failed to upscale {img_path.name}: {e}")

print(f"Upscaled images saved to: {upscaled_dir}")
PYTHON_SCRIPT
    
    # 修改 IMAGES_DIR 指向放大后的图片
    IMAGES_DIR="$UPSCALED_DIR"
else
    IMAGES_DIR="$GEN_DIR"
fi

# ============================================
# Summary
# ============================================

echo ""
echo "=========================================="
echo "Inference completed!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  Checkpoint:  $CHECKPOINT"
echo "  Output:      $OUTPUT_DIR"

GEN_DIR=$(find "$OUTPUT_DIR" -type d -name "generated" 2>/dev/null | head -1)
if [ -n "$GEN_DIR" ] && [ -d "$GEN_DIR" ]; then
    GEN_COUNT=$(ls "$GEN_DIR"/*.png 2>/dev/null | wc -l)
    echo "  Images:      $GEN_DIR ($GEN_COUNT files)"
fi

echo "=========================================="
