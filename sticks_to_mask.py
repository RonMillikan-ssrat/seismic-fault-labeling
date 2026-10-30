#!/usr/bin/env python3
"""
sticks_to_mask.py -- convert labelme fault sticks (linestrip/line shapes)
drawn on the PNGs from extract_inlines.py into training masks.

Outputs (in the same folder):
  <stem>_mask.png      binary fault mask, same pixel grid as the section
  <stem>_overlay.png   seismic + red mask, for QC
  labels_sparse.npy    uint8 volume (n_il, n_xl, n_samp):
                         1 = fault, 0 = labeled non-fault, 255 = UNLABELED
                       Unlabeled inlines get 255 so a training loss can
                       ignore them (e.g. PyTorch CrossEntropyLoss
                       ignore_index=255) instead of treating them as "no fault".

Usage:
  python sticks_to_mask.py --dir inlines --width 3
"""
import argparse
import json
import pathlib

import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IGNORE = 255
LINE_TYPES = {"linestrip", "line"}


def rasterize(shapes, height, width, label, line_width):
    """Draw every matching polyline into a (height, width) uint8 mask."""
    canvas = Image.new("L", (width, height), 0)      # PIL size is (W, H)
    draw = ImageDraw.Draw(canvas)
    n = 0
    for s in shapes:
        if s.get("label") != label or s.get("shape_type") not in LINE_TYPES:
            continue
        pts = [(float(x), float(y)) for x, y in s["points"]]  # (col, row)
        if len(pts) < 2:
            continue
        draw.line(pts, fill=1, width=line_width, joint="curve")
        n += 1
    return np.array(canvas, dtype=np.uint8), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="folder from extract_inlines.py")
    ap.add_argument("--label", default="fault", help="labelme label name to use")
    ap.add_argument("--width", type=int, default=3,
                    help="rasterized stick width in pixels (default 3)")
    args = ap.parse_args()

    d = pathlib.Path(args.dir)
    meta = json.loads((d / "meta.json").read_text())
    n_il, n_xl, n_samp = meta["n_il"], meta["n_xl"], meta["n_samp"]

    vol = np.lib.format.open_memmap(d / "labels_sparse.npy", mode="w+",
                                    dtype=np.uint8, shape=(n_il, n_xl, n_samp))
    vol[:] = IGNORE

    labeled, fault_px, total_px = 0, 0, 0
    for sec in meta["sections"]:
        stem, h, w = sec["stem"], sec["height"], sec["width"]
        jpath = d / f"{stem}.json"
        if not jpath.exists():
            continue                                   # not labeled yet
        lm = json.loads(jpath.read_text())
        if (lm.get("imageHeight"), lm.get("imageWidth")) != (h, w):
            raise SystemExit(f"{jpath}: labelme image size "
                             f"{lm.get('imageHeight')}x{lm.get('imageWidth')} "
                             f"!= section {h}x{w}")
        mask, n_sticks = rasterize(lm["shapes"], h, w, args.label, args.width)

        Image.fromarray(mask * 255).save(d / f"{stem}_mask.png")
        seis = np.load(d / f"{stem}.npy")
        rgb = plt.cm.gray(np.clip((seis / meta["clip"] + 1) / 2, 0, 1))[..., :3]
        rgb[mask == 1] = [1.0, 0.0, 0.0]
        plt.imsave(d / f"{stem}_overlay.png", rgb)

        vol[sec["il_index"]] = mask.T                  # (n_xl, n_samp)
        labeled += 1
        fault_px += int(mask.sum())
        total_px += mask.size
        print(f"{stem}: {n_sticks} sticks, {int(mask.sum())} fault px "
              f"({100 * mask.mean():.2f}%)")

    vol.flush()
    if labeled == 0:
        print(f"no labelme .json files found in {d}/ -- label some PNGs first")
        return
    print("-" * 60)
    print(f"labeled inlines      : {labeled} of {n_il} "
          f"({100 * labeled / n_il:.1f}% of the volume)")
    print(f"fault pixel fraction : {100 * fault_px / total_px:.2f}% "
          f"of labeled pixels  (class imbalance)")
    print(f"wrote {d / 'labels_sparse.npy'}  shape {vol.shape}  "
          f"(255 = unlabeled/ignore)")


if __name__ == "__main__":
    main()
