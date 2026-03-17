#!/usr/bin/env python3
"""Generate full charset test NPZ for inference.

This script creates a test.npz file containing all characters from a charset file,
using the trained font's reference images for style guidance.
"""
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import json


def create_full_charset_npz(
    charset_path: str,
    source_font_path: str,
    ref_image_dir: str,
    output_path: str,
    ref_size: int = 128,
    resolution: int = 256,
):
    """Create NPZ file for full charset generation."""
    from data_processing.font_utils import GlyphRenderer, load_font
    
    with open(charset_path, 'r', encoding='utf-8') as f:
        chars = [line.strip() for line in f if line.strip()]
    
    codepoints = [ord(c) for c in chars]
    n = len(codepoints)
    
    print(f"Total characters: {n}")
    
    source_font, _ = load_font(source_font_path)
    source_renderer = GlyphRenderer(source_font_path, resolution)
    
    ref_images = sorted(Path(ref_image_dir).glob('*.jpg'))
    if not ref_images:
        ref_images = sorted(Path(ref_image_dir).glob('*.png'))
    
    if not ref_images:
        raise ValueError(f"No reference images found in {ref_image_dir}")
    
    print(f"Found {len(ref_images)} reference images")
    
    ref_img = Image.open(ref_images[0]).convert('RGB')
    if ref_img.size == (1024, 256):
        ref_crop = ref_img.crop((512, 0, 640, 128))
    else:
        ref_crop = ref_img.resize((ref_size, ref_size), Image.LANCZOS)
    
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--charset', required=True, help='Path to charset file (one char per line)')
    parser.add_argument('--source-font', required=True, help='Path to source font')
    parser.add_argument('--ref-dir', required=True, help='Directory with reference images')
    parser.add_argument('--output', required=True, help='Output NPZ path')
    parser.add_argument('--ref-size', type=int, default=128)
    parser.add_argument('--resolution', type=int, default=256)
    args = parser.parse_args()
    
    create_full_charset_npz(
        args.charset,
        args.source_font,
        args.ref_dir,
        args.output,
        args.ref_size,
        args.resolution,
    )
