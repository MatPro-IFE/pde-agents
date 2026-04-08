"""Generate the composite Figure 5 for the paper.

Reads .npz files produced by cases A–D and assembles a 2×2 figure:
  (a) Steady Dirichlet          (b) Mixed BCs
  (c) Transient L-shape (t=0.50)(d) 3D cross-section (z=0.5)

Usage
-----
  python plot_composite.py              # default: output/ dir, saves paper fig
  python plot_composite.py --outdir .   # override output location

The script is designed to run *outside* Docker (needs only numpy + matplotlib).
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Composite figure for paper")
    parser.add_argument(
        "--outdir", type=str, default=str(pathlib.Path(__file__).parent / "output"),
        help="Directory containing case_*.npz files",
    )
    parser.add_argument(
        "--paper-dir", type=str,
        default=str(pathlib.Path(__file__).resolve().parents[2] / "paper" / "figs"),
        help="Where to save the final paper figure",
    )
    args = parser.parse_args()
    outdir = pathlib.Path(args.outdir)
    paperdir = pathlib.Path(args.paper_dir)
    paperdir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.2))

    # ── Panel (a): Steady Dirichlet ────────────────────────────────────────
    d = np.load(outdir / "case_a.npz")
    triang = mtri.Triangulation(d["coords"][:, 0], d["coords"][:, 1], d["cells"])
    ax = axes[0, 0]
    tcf = ax.tricontourf(triang, d["values"], levels=32, cmap="inferno")
    fig.colorbar(tcf, ax=ax, shrink=0.82, label="°C")
    ax.set_title("(a) Steady Dirichlet — copper", fontsize=9)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal")

    # ── Panel (b): Mixed BCs ───────────────────────────────────────────────
    d = np.load(outdir / "case_b.npz")
    triang = mtri.Triangulation(d["coords"][:, 0], d["coords"][:, 1], d["cells"])
    ax = axes[0, 1]
    tcf = ax.tricontourf(triang, d["values"], levels=32, cmap="inferno")
    fig.colorbar(tcf, ax=ax, shrink=0.82, label="K")
    ax.set_title("(b) Mixed BCs — steel", fontsize=9)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal")

    # ── Panel (c): Transient L-shape (final snapshot) ──────────────────────
    d = np.load(outdir / "case_c.npz")
    triang = mtri.Triangulation(d["coords"][:, 0], d["coords"][:, 1], d["cells"])
    ax = axes[1, 0]
    snap_key = f"t_{d['snapshot_times'][-1]:.2f}"
    tcf = ax.tricontourf(triang, d[snap_key], levels=32, cmap="inferno")
    fig.colorbar(tcf, ax=ax, shrink=0.82, label="K")
    t_val = d["snapshot_times"][-1]
    ax.set_title(f"(c) Transient L-shape — Al, t={t_val:.2f}s", fontsize=9)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal")

    # ── Panel (d): 3D cross-section at z=0.5 ──────────────────────────────
    d = np.load(outdir / "case_d.npz")
    Xi, Yi = d["Xi"], d["Yi"]
    Zi = d["z_0.5"]
    ax = axes[1, 1]
    cf = ax.contourf(Xi, Yi, Zi, levels=32, cmap="inferno")
    fig.colorbar(cf, ax=ax, shrink=0.82, label="K")
    ax.set_title("(d) 3D cube slice z=0.5 — SS 304", fontsize=9)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal")

    fig.tight_layout(pad=1.2)
    for dest in [outdir / "figure5_composite.png",
                 paperdir / "sim_examples.png"]:
        fig.savefig(dest, dpi=300, bbox_inches="tight")
        print(f"Saved → {dest}")
    plt.close(fig)


if __name__ == "__main__":
    main()
