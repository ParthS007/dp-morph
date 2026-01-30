#!/usr/bin/env python3
"""
Process OCT segmentation plot images for comparison collages.

Each saved plot is a 3-panel image: [Original | True Mask | Predicted Mask].
This script crops them so that:
  - Original + True Mask are saved once (reference, same for all configs).
  - Only the Predicted Mask panel is saved per configuration.

Output structure (for LaTeX):
  {output_dir}/{arch}/nonprivate/{dataset}/
    reference.png          # Original | True Mask (once)
    predicted/
      baseline_bs8.png
      baseline_bs16.png
      morph_open_k3_bs8.png
      ...
  {output_dir}/{arch}/dp/{dataset}/{strategy}/
    reference.png
    predicted/
      baseline_eps8_bs8.png
      ...

Usage:
  python process_collage_images.py --arch lfunet
  python process_collage_images.py --arch unet --example-idx 0 --output-dir collage_images
"""

import os
import argparse
from PIL import Image
import numpy as np

DEFAULT_RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "results"
)

ARCH_DISPLAY_NAMES = {
    "lfunet": "LF-UNet",
    "unet": "U-Net",
    "nestedunet": "U-Net++",
}

STRATEGY_ORDER = ["automatic", "flat", "normalized_sgd", "psac"]

# Panel indices: saved plot is [Original | True Mask | Predicted]
PANEL_ORIGINAL = 0
PANEL_TRUE = 1
PANEL_PREDICTED = 2


def crop_three_panels(image_path):
    """
    Crop a 3-panel horizontal image into left, middle, right.
    Returns (original_pil, true_pil, predicted_pil) or None if failed.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        arr = np.array(img)
        h, w = arr.shape[:2]
        third = w // 3
        # Left, middle, right (with small overlap tolerance)
        original = arr[:, 0:third]
        true_mask = arr[:, third : 2 * third]
        predicted = arr[:, 2 * third : w]
        return (
            Image.fromarray(original),
            Image.fromarray(true_mask),
            Image.fromarray(predicted),
        )
    except Exception as e:
        print(f"  Warning: could not crop {image_path}: {e}")
        return None


def get_source_image_path(results_path, arch, dataset, exp_type, run_dir, example_idx):
    """Build path to example_{example_idx}.png for a given experiment."""
    return os.path.join(
        results_path,
        f"{arch}-{dataset}-{exp_type}",
        run_dir,
        "test",
        "plots",
        f"example_{example_idx}.png",
    )


def process_nonprivate(results_path, arch, output_dir, example_idx):
    """Process non-private Duke and UMN images."""
    configs = [
        ("baseline", None, None),
        ("morph_open_k3", "open", 3),
        ("morph_open_k5", "open", 5),
        ("morph_close_k3", "close", 3),
        ("morph_close_k5", "close", 5),
        ("morph_both_k3", "both", 3),
        ("morph_both_k5", "both", 5),
    ]

    for dataset in ["duke", "umn"]:
        base_dir = f"{arch}-{dataset}-base"
        morph_dir = f"{arch}-{dataset}-base_morph"
        ref_path = get_source_image_path(
            results_path, arch, dataset, "base", "bs8_run1", example_idx
        )
        if not os.path.exists(ref_path):
            ref_path = get_source_image_path(
                results_path, arch, dataset, "base", "bs16_run1", example_idx
            )
        if not os.path.exists(ref_path):
            print(f"  No reference image for {dataset}, skipping.")
            continue

        out_dataset = os.path.join(output_dir, arch, "nonprivate", dataset)
        os.makedirs(out_dataset, exist_ok=True)
        pred_dir = os.path.join(out_dataset, "predicted")
        os.makedirs(pred_dir, exist_ok=True)

        # Reference: Original | True Mask (once)
        panels = crop_three_panels(ref_path)
        if panels:
            orig_pil, true_pil, _ = panels
            ref_combined = Image.new("RGB", (orig_pil.width * 2, orig_pil.height))
            ref_combined.paste(orig_pil, (0, 0))
            ref_combined.paste(true_pil, (orig_pil.width, 0))
            ref_combined.save(os.path.join(out_dataset, "reference.png"))

        # Predicted-only per config
        for config_id, op, k in configs:
            for bs in [8, 16]:
                if config_id == "baseline":
                    run_dir = f"bs{bs}_run1"
                    src = get_source_image_path(
                        results_path, arch, dataset, "base", run_dir, example_idx
                    )
                else:
                    run_dir = f"bs{bs}_run1_{op}_k{k}_layers3-4-5"
                    src = get_source_image_path(
                        results_path, arch, dataset, "base_morph", run_dir, example_idx
                    )
                if not os.path.exists(src):
                    continue
                panels = crop_three_panels(src)
                if panels:
                    _, _, pred_pil = panels
                    out_name = f"{config_id}_bs{bs}.png"
                    pred_pil.save(os.path.join(pred_dir, out_name))


def process_dp(results_path, arch, output_dir, example_idx):
    """Process DP Duke and UMN images (reference once per dataset/strategy, predicted per config)."""
    key_configs = [
        ("baseline", None, None),
        ("morph_open_k3", "open", 3),
        ("morph_both_k5", "both", 5),
    ]
    key_combos = [(8, 8), (200, 16)]  # (eps, bs)

    for dataset in ["duke", "umn"]:
        for strategy in STRATEGY_ORDER:
            base_dir = f"{arch}-{dataset}-dp_{strategy}"
            morph_dir = f"{arch}-{dataset}-dp_{strategy}_morph"
            ref_path = get_source_image_path(
                results_path,
                arch,
                dataset,
                f"dp_{strategy}",
                "bs8_eps8_run1",
                example_idx,
            )
            if not os.path.exists(ref_path):
                ref_path = get_source_image_path(
                    results_path,
                    arch,
                    dataset,
                    f"dp_{strategy}",
                    "bs16_eps200_run1",
                    example_idx,
                )
            if not os.path.exists(ref_path):
                continue

            out_strategy = os.path.join(output_dir, arch, "dp", dataset, strategy)
            os.makedirs(out_strategy, exist_ok=True)
            pred_dir = os.path.join(out_strategy, "predicted")
            os.makedirs(pred_dir, exist_ok=True)

            panels = crop_three_panels(ref_path)
            if panels:
                orig_pil, true_pil, _ = panels
                ref_combined = Image.new("RGB", (orig_pil.width * 2, orig_pil.height))
                ref_combined.paste(orig_pil, (0, 0))
                ref_combined.paste(true_pil, (orig_pil.width, 0))
                ref_combined.save(os.path.join(out_strategy, "reference.png"))

            for config_id, op, k in key_configs:
                for eps, bs in key_combos:
                    if config_id == "baseline":
                        run_dir = f"bs{bs}_eps{eps}_run1"
                        src = get_source_image_path(
                            results_path,
                            arch,
                            dataset,
                            f"dp_{strategy}",
                            run_dir,
                            example_idx,
                        )
                    else:
                        run_dir = f"bs{bs}_eps{eps}_run1_{op}_k{k}"
                        src = get_source_image_path(
                            results_path,
                            arch,
                            dataset,
                            f"dp_{strategy}_morph",
                            run_dir,
                            example_idx,
                        )
                    if not os.path.exists(src):
                        continue
                    panels = crop_three_panels(src)
                    if panels:
                        _, _, pred_pil = panels
                        out_name = f"{config_id}_eps{eps}_bs{bs}.png"
                        pred_pil.save(os.path.join(pred_dir, out_name))


def parse_args():
    p = argparse.ArgumentParser(description="Process OCT plot images for collages.")
    p.add_argument(
        "--arch", "-a", required=True, help="Architecture (lfunet, unet, nestedunet)"
    )
    p.add_argument(
        "--results-path", "-r", default=DEFAULT_RESULTS_PATH, help="Results root"
    )
    p.add_argument(
        "--output-dir", "-o", default="collage_images", help="Output directory"
    )
    p.add_argument(
        "--example-idx", "-e", type=int, default=0, help="Example index (0-19)"
    )
    return p.parse_args()


def main():
    args = parse_args()
    arch = args.arch.lower()
    print(f"Processing collage images for {arch}, example_{args.example_idx}")
    print(f"Results: {args.results_path}")
    print(f"Output:  {args.output_dir}")

    output_root = os.path.join(args.output_dir, arch)
    os.makedirs(output_root, exist_ok=True)

    print("Non-private...")
    process_nonprivate(args.results_path, arch, args.output_dir, args.example_idx)
    print("DP...")
    process_dp(args.results_path, arch, args.output_dir, args.example_idx)
    print("Done.")


if __name__ == "__main__":
    main()
