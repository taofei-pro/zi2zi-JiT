#!/usr/bin/env bash
# zi2zi-JiT Generate TTF Font from generated images
# Usage: bash scripts/generate_ttf.sh

set -e

echo "=========================================="
echo "  zi2zi-JiT Generate TTF Font"
echo "=========================================="

# ============================================
# Configuration
# ============================================

# Target font name (without extension)
TARGET_FONT="${TARGET_FONT:-young}"

# Input: generated images directory
# 自动检测是否有 upscaled 目录
IMAGES_DIR_PATTERN="run/lora_ft_${TARGET_FONT}_L/full_output/*/generated_upscaled"
if ! ls -d $IMAGES_DIR_PATTERN 2>/dev/null | head -1 | grep -q .; then
    IMAGES_DIR_PATTERN="run/lora_ft_${TARGET_FONT}_L/full_output/*/generated"
fi

# Resolve wildcard pattern
IMAGES_DIR=$(ls -d $IMAGES_DIR_PATTERN 2>/dev/null | head -1)
if [ -z "$IMAGES_DIR" ]; then
    IMAGES_DIR="${IMAGES_DIR_PATTERN}"
fi

# Output directory
OUTPUT_DIR="${OUTPUT_DIR:-run/lora_ft_${TARGET_FONT}_L/font_output}"

# Font name
FONT_NAME="${FONT_NAME:-${TARGET_FONT}_generated}"

# SVG directory (intermediate)
SVG_DIR="$OUTPUT_DIR/svg"

# Output font path
FONT_PATH="$OUTPUT_DIR/${FONT_NAME}.ttf"

# Potrace parameters
BLACKLEVEL="${BLACKLEVEL:-0.5}"
TURDSIZE="${TURDSIZE:-2}"
ALPHAMAX="${ALPHAMAX:-0.0}"
OPTTOLERANCE="${OPTTOLERANCE:-0.1}"

# Skip SVG generation if already exists
REGENERATE_SVG="${REGENERATE_SVG:-false}"

echo ""
echo "Configuration:"
echo "  Images Dir:   $IMAGES_DIR"
echo "  Output Dir:   $OUTPUT_DIR"
echo "  Font Name:    $FONT_NAME"
echo "  Font Path:    $FONT_PATH"
echo "  Regenerate SVG: $REGENERATE_SVG"
echo ""

# ============================================
# Step 1: Convert PNG to SVG
# ============================================

echo "=========================================="
echo "Step 1: Converting PNG to SVG"
echo "=========================================="

if [ ! -d "$IMAGES_DIR" ]; then
    echo "ERROR: Images directory not found: $IMAGES_DIR"
    exit 1
fi

# Count images
IMAGE_COUNT=$(ls "$IMAGES_DIR"/*.png 2>/dev/null | wc -l)
echo "Found $IMAGE_COUNT images"

if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo "ERROR: No PNG images found in $IMAGES_DIR"
    exit 1
fi

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$SVG_DIR"

# Check if we should skip SVG generation
EXISTING_SVG_COUNT=$(ls "$SVG_DIR"/*.svg 2>/dev/null | wc -l)
if [ "$REGENERATE_SVG" = "false" ] && [ "$EXISTING_SVG_COUNT" -gt 0 ]; then
    echo "Found $EXISTING_SVG_COUNT existing SVG files."
    echo "Skipping SVG generation. Set REGENERATE_SVG=true to force regeneration."
else
    # Convert PNG to SVG using Python
    python - << PYTHON_SCRIPT
from pathlib import Path
from PIL import Image
import vtracer
from tqdm import tqdm

input_dir = Path("$IMAGES_DIR")
output_dir = Path("$SVG_DIR")

output_dir.mkdir(parents=True, exist_ok=True)

img_paths = sorted(input_dir.glob("*.png"))
print(f"Converting {len(img_paths)} images to SVG...")

failed = 0
for img_path in tqdm(img_paths, desc="Converting"):
    try:
        svg_path = output_dir / img_path.with_suffix(".svg").name
        
        # Use vtracer to convert image to SVG
        vtracer.convert_image_to_svg_py(
            str(img_path),
            str(svg_path),
            colormode='binary',
            hierarchical='stacked',
            mode='spline',
            filter_speckle=4,
            color_precision=6,
            layer_difference=16,
            corner_threshold=60,
            length_threshold=4.0,
            max_iterations=10,
            splice_threshold=45,
            path_precision=8
        )
    except Exception as e:
        failed += 1
        if failed <= 5:
            print(f"  Failed to convert {img_path.name}: {e}")

print(f"Converted {len(img_paths) - failed} images, {failed} failed")
PYTHON_SCRIPT
fi

# Count SVG files
SVG_COUNT=$(ls "$SVG_DIR"/*.svg 2>/dev/null | wc -l)
echo "Generated $SVG_COUNT SVG files"

# ============================================
# Step 2: Convert SVG to TTF
# ============================================

echo ""
echo "=========================================="
echo "Step 2: Converting SVG to TTF"
echo "=========================================="

# Check if fontforge is available
if ! command -v fontforge &> /dev/null; then
    echo "ERROR: fontforge is not installed"
    echo "Please install it with: sudo apt install fontforge"
    exit 1
fi

# Create fontforge script
FONTFORGE_SCRIPT="$OUTPUT_DIR/build_font.py"

cat > "$FONTFORGE_SCRIPT" << 'FONTFORGE_SCRIPT'
import fontforge
import os
from pathlib import Path

svg_dir = Path(os.environ.get("SVG_DIR", "."))
out_path = Path(os.environ.get("FONT_PATH", "output.ttf"))

print(f"Building font from {svg_dir}")
print(f"Output: {out_path}")

font = fontforge.font()
font.familyname = os.environ.get("FONT_NAME", "Generated")
font.fullname = os.environ.get("FONT_NAME", "Generated")
font.fontname = os.environ.get("FONT_NAME", "Generated")

svg_files = sorted(svg_dir.glob("*.svg"))
print(f"Found {len(svg_files)} SVG files")

for svg_path in svg_files:
    name = svg_path.stem
    try:
        # Handle 0000_U+XXXX or U+XXXX format
        if "_U+" in name:
            hex_str = name.split("_U+")[1]
        elif name.startswith("U+"):
            hex_str = name[2:]
        else:
            hex_str = name
        codepoint = int(hex_str, 16)
    except ValueError:
        print(f"  Skipping {name} (invalid codepoint)")
        continue
    
    try:
        glyph_name = f"uni{hex_str}"
        glyph = font.createChar(codepoint, glyph_name)
        glyph.importOutlines(str(svg_path))
        
        # Scale and center the glyph
        xmin, ymin, xmax, ymax = glyph.boundingBox()
        if xmax > xmin and ymax > ymin:
            # Scale to fit within 800x800 (80% of 1000 EM)
            scale = 800.0 / max(xmax - xmin, ymax - ymin)
            
            # Apply scaling
            glyph.transform([scale, 0, 0, scale, 0, 0])
            
            # Recalculate bounding box after scaling
            xmin, ymin, xmax, ymax = glyph.boundingBox()
            
            # Center horizontally (width = 1000, center = 500)
            dx = 500 - (xmin + xmax) / 2
            # Center vertically (Ascent=800, Descent=200, center = 300)
            dy = 300 - (ymin + ymax) / 2
            
            # Apply translation
            glyph.transform([1, 0, 0, 1, dx, dy])
            
        glyph.width = 1000
    except Exception as e:
        print(f"  Failed to add {name}: {e}")

font.generate(str(out_path))
print(f"Font generated: {out_path}")
FONTFORGE_SCRIPT

# Run fontforge
export SVG_DIR
export FONT_PATH
export FONT_NAME

fontforge -lang=py -script "$FONTFORGE_SCRIPT"

# ============================================
# Summary
# ============================================

echo ""
echo "=========================================="
echo "TTF Generation completed!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  Images:      $IMAGE_COUNT"
echo "  SVG files:   $SVG_COUNT"
echo "  Font:        $FONT_PATH"

if [ -f "$FONT_PATH" ]; then
    FONT_SIZE=$(du -h "$FONT_PATH" | cut -f1)
    echo "  Font size:   $FONT_SIZE"
fi

echo "=========================================="
