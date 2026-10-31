#!/usr/bin/env python3
"""
predict_faultseg.py -- run Wu et al.'s pretrained FaultSeg3D (Keras/TensorFlow)
on F3, then save fault-probability volumes and QC images.

Two modes:
  --gxl   sanity check on the repo's F3 subvolume (gxl.dat, 512x384x128).
          If --fpx is given, also compares against the authors' own prediction.
  --segy  your F3 SEG-Y, cropped to the box around your labeled faults
          (from labels_sparse.npy), plus a margin.

Conventions (taken from the faultSeg training code, utils.py):
  * model input axes are (time, x2, x3, 1) -- TIME FIRST
  * each input is normalized to zero mean, unit std
  * trained on 128^3 cubes -> larger volumes are tiled with overlap and
    blended with Gaussian edge weights (as in the repo's notebook)

The 2019 .hdf5 may not load directly in current Keras, so the network is
rebuilt here (same layers as unet3.py) and the weights copied in by order.

Usage:
  python predict_faultseg.py --weights fseg-70.hdf5 --gxl gxl.dat --fpx fpx.dat --out pred_gxl
  python predict_faultseg.py --weights fseg-70.hdf5 --segy F3.sgy --labels inlines --out pred
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 128          # training cube size
OVERLAP = 12     # overlap between tiles (repo notebook uses 12)


# ---------------------------------------------------------------- model ----
def build_unet():
    """Same architecture as faultSeg/unet3.py."""
    from keras import layers, Model
    c = lambda f: layers.Conv3D(f, 3, activation="relu", padding="same")
    inp = layers.Input((None, None, None, 1))
    c1 = c(16)(c(16)(inp));  p1 = layers.MaxPooling3D(2)(c1)
    c2 = c(32)(c(32)(p1));   p2 = layers.MaxPooling3D(2)(c2)
    c3 = c(64)(c(64)(p2));   p3 = layers.MaxPooling3D(2)(c3)
    c4 = c(128)(c(128)(p3))
    u5 = layers.concatenate([layers.UpSampling3D(2)(c4), c3])
    c5 = c(64)(c(64)(u5))
    u6 = layers.concatenate([layers.UpSampling3D(2)(c5), c2])
    c6 = c(32)(c(32)(u6))
    u7 = layers.concatenate([layers.UpSampling3D(2)(c6), c1])
    c7 = c(16)(c(16)(u7))
    out = layers.Conv3D(1, 1, activation="sigmoid")(c7)
    return Model(inp, out)


def load_legacy_weights(model, path):
    """Copy conv kernels/biases from an old Keras .hdf5 into `model`, in order."""
    import h5py
    with h5py.File(path, "r") as f:
        g = f["model_weights"] if "model_weights" in f else f
        names = [n.decode() if isinstance(n, bytes) else n
                 for n in g.attrs["layer_names"]]
        saved = []
        for n in names:
            wn = [w.decode() if isinstance(w, bytes) else w
                  for w in g[n].attrs.get("weight_names", [])]
            if wn:
                saved.append((n, [np.asarray(g[n][w]) for w in wn]))
    targets = [l for l in model.layers if l.weights]
    if len(saved) != len(targets):
        sys.exit(f"weight file has {len(saved)} layers with weights, "
                 f"model has {len(targets)} -- wrong file or architecture")
    for (name, ws), layer in zip(saved, targets):
        want = [tuple(w.shape) for w in layer.get_weights()]
        got = [w.shape for w in ws]
        if want != got:
            sys.exit(f"shape mismatch at saved layer {name}: {got} vs {want}")
        layer.set_weights(ws)
    print(f"loaded {len(saved)} layers of weights from {path}")


# ------------------------------------------------------------- tiling ------
def edge_weights(n=N, os_=OVERLAP):
    """Gaussian taper on the overlap zone of each face (repo notebook's getMask)."""
    sp = np.ones(n, np.float32)
    sig = 0.5 / (os_ / 4) ** 2
    ramp = np.exp(-((np.arange(os_) - os_ + 1) ** 2) * sig).astype(np.float32)
    sp[:os_] = ramp
    sp[n - os_:] = ramp[::-1]
    return sp[:, None, None] * sp[None, :, None] * sp[None, None, :]


def predict_tiled(model, vol):
    """vol: (time, x2, x3), already normalized. Returns probabilities, same shape."""
    m = np.array(vol.shape)
    step = N - OVERLAP
    nt = np.maximum(1, np.ceil((m - OVERLAP) / step).astype(int))
    padded = nt * step + OVERLAP
    gp = np.zeros(padded, np.float32)
    gp[:m[0], :m[1], :m[2]] = vol
    gy = np.zeros(padded, np.float32)
    wk = np.zeros(padded, np.float32)
    w = edge_weights()
    total, k, t0 = int(np.prod(nt)), 0, time.time()
    for a in range(nt[0]):
        for b in range(nt[1]):
            for c in range(nt[2]):
                s = (slice(a * step, a * step + N),
                     slice(b * step, b * step + N),
                     slice(c * step, c * step + N))
                y = model.predict(gp[s][None, ..., None], verbose=0)[0, ..., 0]
                gy[s] += y * w
                wk[s] += w
                k += 1
                print(f"\r  tile {k}/{total}  ({time.time() - t0:.0f}s)",
                      end="", flush=True)
    print()
    return (gy / np.maximum(wk, 1e-6))[:m[0], :m[1], :m[2]]


def normalize(v):
    return ((v - v.mean()) / (v.std() + 1e-12)).astype(np.float32)


# ----------------------------------------------------------------- QC ------
def qc_png(seis2d, prob2d, path, title, stick2d=None):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    ax[0].imshow(seis2d, cmap="gray", vmin=-2, vmax=2, aspect="auto")
    ax[0].set_title("seismic" + (" + your pick (cyan)" if stick2d is not None else ""))
    if stick2d is not None:
        ax[0].contour(stick2d, levels=[0.5], colors="cyan", linewidths=0.8)
    ax[1].imshow(seis2d, cmap="gray", vmin=-2, vmax=2, aspect="auto")
    im = ax[1].imshow(np.ma.masked_less(prob2d, 0.3), cmap="hot",
                      vmin=0.3, vmax=1, alpha=0.8, aspect="auto")
    ax[1].set_title("FaultSeg probability (>0.3)")
    fig.colorbar(im, ax=ax[1], fraction=0.03)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------- modes -----
def run_gxl(model, args, out):
    raw = np.fromfile(args.gxl, dtype=np.single)
    if raw.size != 512 * 384 * 128:
        sys.exit(f"{args.gxl}: {raw.size} samples, expected 512*384*128")
    # file layout [n3][n2][n1], n1 = time fastest; transpose -> time first
    gx = np.transpose(raw.reshape(512, 384, 128))          # (128, 384, 512)
    gx = normalize(gx)
    print(f"gxl volume (time, x2, x3) = {gx.shape}")
    fp = predict_tiled(model, gx)
    np.save(out / "gxl_prob.npy", fp)
    qc_png(gx[:, 29, :], fp[:, 29, :], out / "gxl_x2_029.png", "gxl.dat, x2=29")
    qc_png(gx[:, :, 99], fp[:, :, 99], out / "gxl_x3_099.png", "gxl.dat, x3=99")
    if args.fpx:
        ref = np.fromfile(args.fpx, dtype=np.single)
        if ref.size == fp.size:
            ref = np.transpose(ref.reshape(512, 384, 128))
            r = np.corrcoef(ref.ravel(), fp.ravel())[0, 1]
            print(f"correlation with authors' fpx.dat: {r:.3f} "
                  f"(close to 1 = your setup matches theirs)")
        else:
            print(f"fpx.dat size {ref.size} != {fp.size}; skipped comparison")


def run_segy(model, args, out):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from extract_inlines import open_segy

    lab_dir = pathlib.Path(args.labels)
    meta = json.loads((lab_dir / "meta.json").read_text())
    labels = np.load(lab_dir / "labels_sparse.npy", mmap_mode="r")  # (il, xl, t)

    fault = np.argwhere(np.asarray(labels) == 1)
    if fault.size == 0:
        sys.exit("no fault pixels in labels_sparse.npy -- run sticks_to_mask.py first")
    lo = np.maximum(fault.min(0) - args.margin, 0)
    hi = np.minimum(fault.max(0) + args.margin + 1, labels.shape)
    print(f"crop (il, xl, t) index ranges: {lo.tolist()} .. {hi.tolist()}")

    f, ilnum, xlnum, samples, get_section = open_segy(
        args.segy, args.iline_byte, args.xline_byte)
    cube = np.stack([get_section(i)[lo[1]:hi[1], lo[2]:hi[2]]
                     for i in range(lo[0], hi[0])]).astype(np.float32)
    f.close()
    print(f"crop cube (il, xl, t) = {cube.shape}, "
          f"{cube.nbytes / 1e6:.0f} MB")

    gx = normalize(np.transpose(cube, (2, 1, 0)))   # -> (t, xl, il), time first
    fp = np.transpose(predict_tiled(model, gx), (2, 1, 0))   # back to (il, xl, t)

    np.save(out / "prob_crop.npy", fp.astype(np.float32))
    (out / "crop.json").write_text(json.dumps(
        {"lo": lo.tolist(), "hi": hi.tolist(),
         "weights": str(args.weights), "segy": str(args.segy)}, indent=1))

    seis_n = normalize(cube)
    labeled = [s["il_index"] for s in meta["sections"]
               if lo[0] <= s["il_index"] < hi[0]
               and (np.asarray(labels[s["il_index"]]) != 255).any()]
    for i in labeled[:: max(1, len(labeled) // 6)]:      # ~6 QC images
        k = i - lo[0]
        stick = (np.asarray(labels[i, lo[1]:hi[1], lo[2]:hi[2]]) == 1).T
        qc_png(seis_n[k].T, fp[k].T, out / f"qc_il_{ilnum[i]:05d}.png",
               f"inline {ilnum[i]}", stick)
    print(f"wrote prob_crop.npy, crop.json and QC PNGs to {out}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="e.g. fseg-70.hdf5")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--gxl")
    src.add_argument("--segy")
    ap.add_argument("--fpx", help="authors' prediction for gxl.dat (optional)")
    ap.add_argument("--labels", help="folder with meta.json + labels_sparse.npy")
    ap.add_argument("--margin", type=int, default=48,
                    help="pixels added around the labeled faults (default 48)")
    ap.add_argument("--iline-byte", type=int, default=189)
    ap.add_argument("--xline-byte", type=int, default=193)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.segy and not args.labels:
        ap.error("--segy needs --labels")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = build_unet()
    try:
        load_legacy_weights(model, args.weights)
    except KeyError as e:
        sys.exit(f"could not read layer list from {args.weights}: {e}")

    run_gxl(model, args, out) if args.gxl else run_segy(model, args, out)


if __name__ == "__main__":
    main()
