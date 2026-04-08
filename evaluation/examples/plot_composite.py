"""Generate the composite Figure for the paper (3×2 grid, 6 panels).

Reads .npz files produced by cases A–F and assembles:
  Row 1:  (a) Steady Dirichlet      (b) Mixed BCs
  Row 2:  (c) Plate with hole       (d) Gaussian source
  Row 3:  (e) Transient L-shape     (f) 3D cross-section

Usage
-----
  python plot_composite.py              # default: output/ dir
  python plot_composite.py --outdir .   # override output location

Runs outside Docker (needs only numpy + matplotlib).
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


def _plot_tri(ax, npz, label, unit="K"):
    """Plot a 2D tricontourf panel."""
    triang = mtri.Triangulation(
        npz["coords"][:, 0], npz["coords"][:, 1], npz["cells"]
    )
    vals = npz["values"]
    tcf = ax.tricontourf(triang, vals, levels=32, cmap="inferno")
    cb = plt.colorbar(tcf, ax=ax, shrink=0.82, pad=0.02)
    cb.set_label(unit, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_title(label, fontsize=9, pad=4)
    ax.set_xlabel("x", fontsize=8)
    ax.set_ylabel("y", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal")


def main():
    parser = argparse.ArgumentParser(description="Composite figure for paper")
    parser.add_argument(
        "--outdir", type=str,
        default=str(pathlib.Path(__file__).parent / "output"),
    )
    parser.add_argument(
        "--paper-dir", type=str,
        default=str(pathlib.Path(__file__).resolve().parents[2] / "paper" / "figs"),
    )
    args = parser.parse_args()
    outdir = pathlib.Path(args.outdir)
    paperdir = pathlib.Path(args.paper_dir)
    paperdir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(7.0, 9.2))

    # ── (a) Steady Dirichlet ──────────────────────────────────────────────
    d = np.load(outdir / "case_a.npz")
    _plot_tri(axes[0, 0], d, "(a) Steady Dirichlet — copper")

    # ── (b) Mixed BCs ─────────────────────────────────────────────────────
    d = np.load(outdir / "case_b.npz")
    _plot_tri(axes[0, 1], d, "(b) Mixed BCs — steel")

    # ── (c) Plate with hole ───────────────────────────────────────────────
    d = np.load(outdir / "case_e.npz")
    _plot_tri(axes[1, 0], d, "(c) Plate with hole — Ti-6Al-4V")

    # ── (d) Gaussian heat source ──────────────────────────────────────────
    d = np.load(outdir / "case_f.npz")
    _plot_tri(axes[1, 1], d, "(d) Gaussian source — Al 6061")

    # ── (e) Transient L-shape (final snapshot) ────────────────────────────
    d = np.load(outdir / "case_c.npz")
    triang = mtri.Triangulation(
        d["coords"][:, 0], d["coords"][:, 1], d["cells"]
    )
    snap_key = f"t_{d['snapshot_times'][-1]:.2f}"
    ax = axes[2, 0]
    tcf = ax.tricontourf(triang, d[snap_key], levels=32, cmap="inferno")
    cb = plt.colorbar(tcf, ax=ax, shrink=0.82, pad=0.02)
    cb.set_label("K", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    t_val = d["snapshot_times"][-1]
    ax.set_title(f"(e) Transient L-shape — Al, t={t_val:.2f}s", fontsize=9, pad=4)
    ax.set_xlabel("x", fontsize=8)
    ax.set_ylabel("y", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal")

    # ── (f) 3D cutaway render ────────────────────────────────────────────
    ax = axes[2, 1]
    img_path = outdir / "case_d_3d.png"
    if img_path.exists():
        from matplotlib.image import imread
        img = imread(str(img_path))
        ax.imshow(img)
        ax.set_title("(f) 3D cutaway — SS 304", fontsize=9, pad=4)
        ax.set_axis_off()
    else:
        d = np.load(outdir / "case_d.npz")
        Xi, Yi = d["Xi"], d["Yi"]
        Zi = d["z_0.5"]
        cf = ax.contourf(Xi, Yi, Zi, levels=32, cmap="inferno")
        cb = plt.colorbar(cf, ax=ax, shrink=0.82, pad=0.02)
        cb.set_label("K", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        ax.set_title("(f) 3D cube slice z=0.5 — SS 304", fontsize=9, pad=4)
        ax.set_xlabel("x", fontsize=8)
        ax.set_ylabel("y", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal")

    fig.tight_layout(h_pad=1.8, w_pad=1.5)
    for dest in [outdir / "figure_gallery.png",
                 paperdir / "sim_examples.png"]:
        fig.savefig(dest, dpi=300, bbox_inches="tight")
        print(f"Saved → {dest}")
    plt.close(fig)


if __name__ == "__main__":
    main()
