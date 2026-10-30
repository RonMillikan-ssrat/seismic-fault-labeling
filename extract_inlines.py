#!/usr/bin/env python3
"""
extract_inlines.py -- export inline sections from a 3D seismic volume as
1:1-pixel grayscale PNGs for labeling in labelme, plus a metadata file that
maps pixels back to (inline, crossline, time sample).

Input: SEG-Y (via segyio) or a .npy cube with axes (inline, crossline, sample).

Key points:
  * One PNG pixel == one (crossline, sample) cell. No resizing, so label
    coordinates map straight back to the volume.
  * One clip value is used for every exported inline, so all images get
    the same amplitude scaling.

Usage:
  python extract_inlines.py --segy F3.sgy --out inlines --step 10
  python extract_inlines.py --segy F3.sgy --out inlines --inlines 200 210 220
  python extract_inlines.py --npy train_seismic.npy --out inlines --step 10
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def open_segy(path, iline_byte, xline_byte):
    """Return (handle, il_numbers, xl_numbers, samples, get_section).

    Fast path: segyio's regular-grid reader.
    Fallback: if the file is irregular (duplicate or missing traces),
    build the grid from the trace headers ourselves.
    """
    import segyio
    try:
        f = segyio.open(path, "r", iline=iline_byte, xline=xline_byte)
        il = np.asarray(f.ilines)
        xl = np.asarray(f.xlines)
        print(f"SEG-Y: {path}  (regular grid, sorting={f.sorting})")
        return (f, il, xl, np.asarray(f.samples),
                lambda i: np.asarray(f.iline[int(il[i])]))
    except Exception as e:
        print(f"SEG-Y: {path}")
        print(f"  regular-grid read failed: {e}")
        print(f"  falling back to header scan (iline byte {iline_byte}, "
              f"xline byte {xline_byte})")

    f = segyio.open(path, "r", ignore_geometry=True)
    tr_il = np.asarray(f.attributes(iline_byte)[:]).ravel()
    tr_xl = np.asarray(f.attributes(xline_byte)[:]).ravel()
    il = np.unique(tr_il)
    xl = np.unique(tr_xl)
    il_idx = np.searchsorted(il, tr_il)
    xl_idx = np.searchsorted(xl, tr_xl)

    # first trace seen at each (il, xl) cell; -1 = no trace there
    grid = np.full((len(il), len(xl)), -1, dtype=np.int64)
    dup = 0
    for t in range(len(tr_il)):
        a, b = il_idx[t], xl_idx[t]
        if grid[a, b] == -1:
            grid[a, b] = t
        else:
            dup += 1
    missing = int((grid == -1).sum())
    print(f"  traces in file   : {len(tr_il)}")
    print(f"  grid cells       : {len(il)} x {len(xl)} = {grid.size}")
    print(f"  duplicate traces : {dup} (first copy kept)")
    print(f"  empty cells      : {missing} (filled with zeros)")

    ns = len(f.samples)

    def get_section(i):
        sec = np.zeros((len(xl), ns), dtype=np.float32)
        for b, t in enumerate(grid[i]):
            if t >= 0:
                sec[b] = f.trace[int(t)]
        return sec

    return f, il, xl, np.asarray(f.samples), get_section


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--segy")
    src.add_argument("--npy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=10,
                    help="export every Nth inline (default 10)")
    ap.add_argument("--inlines", type=int, nargs="*",
                    help="explicit inline NUMBERS (SEG-Y) or indices (npy)")
    ap.add_argument("--iline-byte", type=int, default=189)
    ap.add_argument("--xline-byte", type=int, default=193)
    ap.add_argument("--clip-pct", type=float, default=99.0,
                    help="percentile of |amplitude| used as display clip")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- open the volume and describe its geometry --------------------
    if args.segy:
        f, il_numbers, xl_numbers, samples, get_section = open_segy(
            args.segy, args.iline_byte, args.xline_byte)
    else:
        vol = np.load(args.npy, mmap_mode="r")
        if vol.ndim != 3:
            sys.exit(f"expected 3D array, got shape {vol.shape}")
        il_numbers = np.arange(vol.shape[0])
        xl_numbers = np.arange(vol.shape[1])
        samples = np.arange(vol.shape[2], dtype=float)
        get_section = lambda il_idx: np.asarray(vol[il_idx])
        print(f"NPY: {args.npy}  shape {vol.shape} (assumed il, xl, sample)")

    n_il, n_xl, n_samp = len(il_numbers), len(xl_numbers), len(samples)
    dt = float(samples[1] - samples[0]) if n_samp > 1 else 0.0
    print(f"  inlines       : {il_numbers[0]}..{il_numbers[-1]}  (n={n_il})")
    print(f"  crosslines    : {xl_numbers[0]}..{xl_numbers[-1]}  (n={n_xl})")
    print(f"  samples       : {samples[0]}..{samples[-1]}  dt={dt}  (n={n_samp})")

    # ---- choose inlines -----------------------------------------------
    if args.inlines:
        lookup = {int(v): i for i, v in enumerate(il_numbers)}
        missing = [v for v in args.inlines if v not in lookup]
        if missing:
            sys.exit(f"inline(s) not in volume: {missing}")
        idxs = [lookup[v] for v in args.inlines]
    else:
        idxs = list(range(0, n_il, args.step))
    print(f"  exporting     : {len(idxs)} inlines")

    # ---- one global clip from the selected sections -------------------
    sections = {i: get_section(i).astype(np.float32) for i in idxs}
    absvals = np.concatenate([np.abs(s).ravel() for s in sections.values()])
    clip = float(np.percentile(absvals, args.clip_pct))
    print(f"  display clip  : +/-{clip:.4g}  ({args.clip_pct} pct of |amp|)")

    # ---- write images, raw sections, metadata --------------------------
    meta = {
        "source": args.segy or args.npy,
        "n_il": n_il, "n_xl": n_xl, "n_samp": n_samp,
        "il_numbers": [int(v) for v in il_numbers],
        "xl_first": int(xl_numbers[0]), "xl_last": int(xl_numbers[-1]),
        "t_first": float(samples[0]), "dt": dt,
        "clip": clip,
        "sections": [],
    }
    for i in idxs:
        sec = sections[i]                  # (n_xl, n_samp)
        img = sec.T                        # rows = time (down), cols = crossline
        stem = f"il_{int(il_numbers[i]):05d}"
        plt.imsave(out / f"{stem}.png", img, cmap="gray", vmin=-clip, vmax=clip)
        np.save(out / f"{stem}.npy", img)
        meta["sections"].append({"stem": stem, "il_index": i,
                                 "il_number": int(il_numbers[i]),
                                 "height": img.shape[0], "width": img.shape[1]})
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {len(idxs)} PNG + NPY pairs and meta.json to {out}/")
    print(f"image size = {n_samp} rows (time) x {n_xl} cols (crossline)")

    if args.segy:
        f.close()


if __name__ == "__main__":
    main()
