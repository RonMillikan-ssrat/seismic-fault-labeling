#!/usr/bin/env python3
"""
diag_gxl.py -- why does our gxl prediction correlate only ~0.3 with fpx.dat?

Checks:
  1. value statistics of both volumes (range, mean, fraction > 0.5)
  2. correlation of our prediction against fpx.dat read with different
     axis orders (and flips); the right axis order should stand out
  3. side-by-side images of the same slices: seismic | ours | fpx.dat

Usage:
  python diag_gxl.py --gxl gxl/gxl.dat --fpx gxl/fpx.dat --pred pred_gxl
"""
import argparse
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SHAPE_FILE = (512, 384, 128)   # [n3][n2][n1], n1 = time fastest


def stats(name, v):
    print(f"  {name:<10} min {v.min():7.3f}  max {v.max():7.3f}  "
          f"mean {v.mean():6.3f}  >0.5: {100 * (v > 0.5).mean():5.2f}%")


def corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gxl", required=True)
    ap.add_argument("--fpx", required=True)
    ap.add_argument("--pred", required=True, help="folder with gxl_prob.npy")
    args = ap.parse_args()

    ours = np.load(pathlib.Path(args.pred) / "gxl_prob.npy")        # (t, x2, x3)
    raw = np.fromfile(args.fpx, dtype=np.single)
    gx = np.transpose(np.fromfile(args.gxl, dtype=np.single).reshape(SHAPE_FILE))

    print("value statistics:")
    stats("ours", ours)
    stats("fpx.dat", raw)

    # candidate readings of fpx.dat that end up shaped like ours (128,384,512)
    cands = {
        "same layout as gxl (reshape 512,384,128 -> transpose)":
            np.transpose(raw.reshape(SHAPE_FILE)),
        "already time-first (reshape 128,384,512)":
            raw.reshape(128, 384, 512),
    }
    for k in list(cands):
        cands[k + " + time flipped"] = cands[k][::-1]
    print("\ncorrelation of ours vs fpx.dat read as:")
    best = max(cands, key=lambda k: corr(ours, cands[k]))
    for k, v in cands.items():
        print(f"  {corr(ours, v):6.3f}  {k}{'   <-- best' if k == best else ''}")

    ref = cands[best]
    out = pathlib.Path(args.pred)
    for label, sl in [("x2_029", np.s_[:, 29, :]), ("x3_099", np.s_[:, :, 99]),
                      ("t_064", np.s_[64, :, :])]:
        g = gx[sl]
        g = (g - g.mean()) / (g.std() + 1e-12)
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        ax[0].imshow(g, cmap="gray", vmin=-2, vmax=2, aspect="auto")
        ax[0].set_title("seismic")
        ax[1].imshow(ours[sl], cmap="gray_r", vmin=0, vmax=1, aspect="auto")
        ax[1].set_title("ours (fseg-70)")
        ax[2].imshow(ref[sl], cmap="gray_r", vmin=0, vmax=1, aspect="auto")
        ax[2].set_title("fpx.dat (authors)")
        fig.suptitle(f"gxl slice {label}")
        fig.tight_layout()
        fig.savefig(out / f"diag_{label}.png", dpi=100)
        plt.close(fig)
    print(f"\nwrote diag_x2_029.png, diag_x3_099.png, diag_t_064.png to {out}/")


if __name__ == "__main__":
    main()
