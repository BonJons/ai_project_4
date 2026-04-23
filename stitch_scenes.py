"""
Where's Waldo? — Patch Stitcher
================================
Reconstructs full scene images from 256×256 patches named {scene}_{row}_{col}.jpg.
Patches are sourced from both waldo/ and notwaldo/ folders.

Output: scenes/{scene_id}.jpg  (1024×1024 per scene for a 4×4 grid)

Run:
    python stitch_scenes.py
    python stitch_scenes.py --data data/ --output scenes/
"""

import os
import argparse
from collections import defaultdict
from PIL import Image


def stitch_scenes(data_root="data/", output_dir="scenes/"):
    os.makedirs(output_dir, exist_ok=True)

    # Collect all patches from both folders into {scene: {(row, col): path}}
    scene_patches = defaultdict(dict)
    for folder in ["waldo", "notwaldo"]:
        folder_path = os.path.join(data_root, folder)
        if not os.path.isdir(folder_path):
            print(f"  [Warning] Folder not found: {folder_path}")
            continue
        for fname in os.listdir(folder_path):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            parts = fname.rsplit(".", 1)[0].split("_")
            if len(parts) != 3:
                continue
            scene, row, col = parts[0], int(parts[1]), int(parts[2])
            scene_patches[scene][(row, col)] = os.path.join(folder_path, fname)

    print(f"\n{'='*60}")
    print(f"  Waldo Patch Stitcher")
    print(f"  Found {len(scene_patches)} scenes in {data_root}")
    print(f"  Output -> {output_dir}")
    print(f"{'='*60}")

    stitched = []
    for scene_id in sorted(scene_patches.keys(), key=lambda x: int(x)):
        patches = scene_patches[scene_id]

        max_row = max(r for r, c in patches) + 1
        max_col = max(c for r, c in patches) + 1
        patch_size = 256

        canvas_w = max_col * patch_size
        canvas_h = max_row * patch_size
        canvas = Image.new("RGB", (canvas_w, canvas_h), color=(200, 200, 200))

        missing = []
        for row in range(max_row):
            for col in range(max_col):
                if (row, col) in patches:
                    patch = Image.open(patches[(row, col)]).convert("RGB")
                    patch = patch.resize((patch_size, patch_size), Image.BILINEAR)
                    canvas.paste(patch, (col * patch_size, row * patch_size))
                else:
                    missing.append((row, col))

        out_path = os.path.join(output_dir, f"scene_{scene_id}.jpg")
        canvas.save(out_path)
        stitched.append(out_path)

        status = f"  Scene {scene_id:>2}: {max_row}×{max_col} grid -> {canvas_w}×{canvas_h}px"
        if missing:
            status += f"  [{len(missing)} missing patch(es) shown as grey]"
        print(status)

    print(f"\n  Done. {len(stitched)} scenes saved to '{output_dir}'")
    return stitched


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   type=str, default="data/",   help="Root folder with waldo/ and notwaldo/")
    parser.add_argument("--output", type=str, default="scenes/", help="Output folder for stitched scenes")
    args = parser.parse_args()

    stitch_scenes(data_root=args.data, output_dir=args.output)
