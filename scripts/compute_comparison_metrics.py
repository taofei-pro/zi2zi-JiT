"""
Compute pairwise SSIM, LPIPS, L1, and FID metrics from comparison images.

Each comparison image is expected to be a 512x256 image where:
- Left half (256x256) is the target/ground truth
- Right half (256x256) is the generated image

Usage:
    python scripts/compute_comparison_metrics.py <folder> [--device cpu|mps|cuda] [--batch-size 64]
"""

import os
import sys
import glob
import argparse

import torch
import numpy as np
import cv2
from pytorch_msssim import ssim as _ssim_fn
import lpips

from util.metrics import compute_fid


_lpips_fn = None


def _get_lpips_fn(device="cpu"):
    global _lpips_fn
    if _lpips_fn is None:
        _lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
    return _lpips_fn


def extract_pair(img):
    """Extract (target, generated) pair from a single comparison image.

    Args:
        img: (H, W, C) uint8 BGR numpy array where W = 2 * H (e.g. 512x256).

    Returns:
        target: (H, H, 3) uint8 BGR
        generated: (H, H, 3) uint8 BGR
    """
    h, w = img.shape[:2]
    assert w == 2 * h, f"Expected width=2*height, got {w}x{h}"
    
    target = img[:, :h, :]
    generated = img[:, h:, :]
    
    return target, generated


@torch.no_grad()
def compute_comparison_metrics(folder, device="cpu", batch_size=64):
    """Compute averaged SSIM, LPIPS, L1, and FID over all comparison images in *folder*."""
    paths = sorted(glob.glob(os.path.join(folder, "*.png")))
    assert len(paths) > 0, f"No PNG files found in {folder}"

    lpips_fn = _get_lpips_fn(device)
    ssim_sum, lpips_sum, l1_sum, count = 0.0, 0.0, 0.0, 0

    # Collect all pairs first
    all_targets, all_generateds = [], []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"  WARNING: failed to read {p}, skipping")
            continue
        tgt, gen = extract_pair(img)
        all_targets.append(tgt)
        all_generateds.append(gen)

    all_targets = np.stack(all_targets, axis=0)
    all_generateds = np.stack(all_generateds, axis=0)
    total = all_targets.shape[0]
    print(f"Found {len(paths)} comparison images -> {total} pairs")

    all_tgt_tensors = []
    all_gen_tensors = []

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        tgt_bgr = all_targets[start:end]
        gen_bgr = all_generateds[start:end]

        # BGR uint8 -> RGB float [0,1] (B,3,H,W)
        tgt = torch.from_numpy(tgt_bgr[:, :, :, ::-1].copy()).permute(0, 3, 1, 2).float() / 255.0
        gen = torch.from_numpy(gen_bgr[:, :, :, ::-1].copy()).permute(0, 3, 1, 2).float() / 255.0

        B = tgt.shape[0]

        ssim_vals = _ssim_fn(gen, tgt, data_range=1.0, size_average=False)
        ssim_sum += ssim_vals.sum().item()

        gen_lp = (gen.to(device) * 2.0 - 1.0)
        tgt_lp = (tgt.to(device) * 2.0 - 1.0)
        lpips_vals = lpips_fn(gen_lp, tgt_lp)
        lpips_sum += lpips_vals.sum().item()

        l1_vals = torch.abs(gen - tgt).mean(dim=(1, 2, 3))
        l1_sum += l1_vals.sum().item()

        all_tgt_tensors.append(tgt)
        all_gen_tensors.append(gen)

        count += B
        print(f"  Processed {count}/{total} pairs", end="\r")

    print()
    print("  Computing FID...", end=" ", flush=True)
    all_tgt_tensors = torch.cat(all_tgt_tensors, dim=0)
    all_gen_tensors = torch.cat(all_gen_tensors, dim=0)
    fid_val = compute_fid(all_tgt_tensors, all_gen_tensors, device=device, batch_size=batch_size)
    print(f"done")

    return {
        "ssim": ssim_sum / count,
        "lpips": lpips_sum / count,
        "l1": l1_sum / count,
        "fid": fid_val,
        "num_pairs": count,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute metrics from comparison images (target|generated).")
    parser.add_argument("folder", help="Path to folder containing comparison PNG images")
    parser.add_argument("--device", default="cpu", help="Device for LPIPS (cpu, mps, cuda)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for metric computation")
    args = parser.parse_args()

    results = compute_comparison_metrics(args.folder, device=args.device, batch_size=args.batch_size)

    print(f"Results over {results['num_pairs']} pairs:")
    print(f"  SSIM  = {results['ssim']:.4f}")
    print(f"  LPIPS = {results['lpips']:.4f}")
    print(f"  L1    = {results['l1']:.4f}")
    print(f"  FID   = {results['fid']:.4f}")


if __name__ == "__main__":
    main()
