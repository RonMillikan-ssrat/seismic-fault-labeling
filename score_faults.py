#!/usr/bin/env python3
"""
score_faults.py -- score FaultSeg probabilities against your sparse fault picks.

Only inlines you labeled are scored (labels != 255). On each one:
  L = your fault pixels, P = prediction >= threshold.

  recall      fraction of your fault pixels with a predicted fault
              within --tol pixels
  precision   fraction of predicted fault pixels within --tol pixels
              of one of your picks

Picks are approximate, so exact pixel matching would be unfair; --tol
allows for that.

Precision is reported twice:
  all       over the whole labeled inline. PESSIMISTIC: you labeled one
            fault, so any other real fault the model finds counts as a
            false positive.
  corridor  only within --corridor pixels of your pick. Asks: near your
            fault, does the model put its fault where you did?

Usage:
  python score_faults.py --labels inlines --pred pred --tol 3
"""
import argparse
import json
import pathlib

import numpy as np
from scipy.ndimage import distance_transform_edt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--tol", type=float, default=3.0,
                    help="match distance in pixels (default 3)")
    ap.add_argument("--corridor", type=float, default=40.0,
                    help="half-width in pixels of the region scored by "
                         "'corridor' precision (default 40)")
    ap.add_argument("--thresholds", type=float, nargs="*",
                    default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    args = ap.parse_args()

    lab = np.load(pathlib.Path(args.labels) / "labels_sparse.npy", mmap_mode="r")
    prob = np.load(pathlib.Path(args.pred) / "prob_crop.npy")
    crop = json.loads((pathlib.Path(args.pred) / "crop.json").read_text())
    lo, hi = crop["lo"], crop["hi"]
    lab = np.asarray(lab[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
    assert lab.shape == prob.shape, (lab.shape, prob.shape)

    rows = [k for k in range(lab.shape[0]) if (lab[k] != 255).any()]
    if not rows:
        raise SystemExit("no labeled inlines inside the crop")
    print(f"scoring {len(rows)} labeled inlines, tol={args.tol}px, "
          f"corridor={args.corridor}px")

    # distance from every pixel to your nearest pick, per inline
    d_lab = {k: distance_transform_edt(lab[k] != 1) for k in rows}

    table = []
    for th in args.thresholds:
        tp_l = n_l = 0               # recall counts
        tp_p = n_p = 0               # precision, whole inline
        tp_c = n_c = 0               # precision, corridor
        for k in rows:
            L = lab[k] == 1
            valid = lab[k] != 255
            P = (prob[k] >= th) & valid
            d_pred = distance_transform_edt(~P) if P.any() else np.full(P.shape, np.inf)
            near_lab = d_lab[k] <= args.tol
            corr = d_lab[k] <= args.corridor

            n_l += L.sum();            tp_l += (L & (d_pred <= args.tol)).sum()
            n_p += P.sum();            tp_p += (P & near_lab).sum()
            n_c += (P & corr).sum();   tp_c += (P & corr & near_lab).sum()
        rec = tp_l / max(n_l, 1)
        pre = tp_p / max(n_p, 1)
        pre_c = tp_c / max(n_c, 1)
        table.append((th, rec, pre, pre_c, int(n_p)))

    print(f"\n{'thresh':>6} {'recall':>7} {'prec(all)':>10} "
          f"{'prec(corridor)':>15} {'pred px':>9}")
    for th, rec, pre, pre_c, n_p in table:
        print(f"{th:6.2f} {rec:7.3f} {pre:10.3f} {pre_c:15.3f} {n_p:9d}")

    fig, ax = plt.subplots(figsize=(6, 5))
    r = [t[1] for t in table]
    ax.plot(r, [t[3] for t in table], "o-", label="precision (corridor)")
    ax.plot(r, [t[2] for t in table], "s--", label="precision (all, pessimistic)")
    for th, rec, _, pre_c, _ in table:
        ax.annotate(f"{th:.1f}", (rec, pre_c), textcoords="offset points",
                    xytext=(4, 4), fontsize=8)
    ax.set_xlabel(f"recall (tol {args.tol:g} px)")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("FaultSeg3D vs. your fault picks (F3)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = pathlib.Path(args.pred) / "precision_recall.png"
    fig.savefig(out, dpi=120)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
