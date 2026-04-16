#!/usr/bin/env python3
"""Analyse how success rate evolves as the KG grows.

Reads simulation_runs from PostgreSQL, correlates chronological order
(proxy for KG size) with windowed success rate, and produces:
  1. A rolling-window success rate plot (Figure for paper)
  2. A cumulative success rate table
  3. JSON results for reproducibility
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "pde_simulations")
DB_USER = os.getenv("POSTGRES_USER", "pde_user")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "pde_secret_change_me")

OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figs")


def fetch_runs():
    """Fetch all simulation runs ordered chronologically."""
    try:
        import psycopg2
    except ImportError:
        sys.exit("psycopg2 not installed. Run: pip install psycopg2-binary")

    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASS,
    )
    cur = conn.cursor()
    cur.execute("""
        SELECT created_at, status, run_id, wall_time
        FROM simulation_runs
        ORDER BY created_at
    """)
    rows = cur.fetchall()
    conn.close()
    return rows


def compute_metrics(rows):
    """Compute cumulative and rolling success rates."""
    n = len(rows)
    success = np.array([1 if r[1] == "SUCCESS" else 0 for r in rows])
    run_numbers = np.arange(1, n + 1)

    cum_success = np.cumsum(success)
    cum_rate = cum_success / run_numbers * 100

    window = 50
    rolling_rate = np.full(n, np.nan)
    for i in range(window - 1, n):
        rolling_rate[i] = success[i - window + 1:i + 1].mean() * 100

    dates = [r[0] for r in rows]

    failure_indices = np.where(success == 0)[0]
    failure_run_numbers = run_numbers[failure_indices]

    return {
        "run_numbers": run_numbers,
        "cum_rate": cum_rate,
        "rolling_rate": rolling_rate,
        "window": window,
        "total": n,
        "total_success": int(cum_success[-1]),
        "total_fail": n - int(cum_success[-1]),
        "dates": dates,
        "failure_run_numbers": failure_run_numbers,
        "success_array": success,
    }


def compute_phase_stats(metrics):
    """Break data into meaningful phases showing KG growth effect."""
    success = metrics["success_array"]
    dates = metrics["dates"]
    n = len(success)

    boundaries = [
        (0, 16, "Bootstrap"),         # Early runs, small KG
        (16, 329, "Sweep 1"),         # First big batch
        (329, 1134, "Sweep 2"),       # Largest batch (parameter sweep)
        (1134, n, "Ablation+"),       # Later experiments + ablation
    ]

    phases = []
    for i, (start, end, name) in enumerate(boundaries):
        if start >= n:
            break
        end = min(end, n)
        seg = success[start:end]
        phases.append({
            "phase": i + 1,
            "name": name,
            "runs": f"{start + 1}–{end}",
            "count": int(end - start),
            "date_start": str(dates[start].date()),
            "date_end": str(dates[end - 1].date()),
            "success_rate": round(seg.mean() * 100, 1),
            "failures": int(seg.size - seg.sum()),
            "kg_size_start": start,
            "kg_size_end": end,
        })
    return phases


def make_plot(metrics, phases, out_path):
    """Generate the KG growth figure."""
    fig, ax1 = plt.subplots(figsize=(6.5, 3.2))

    rn = metrics["run_numbers"]
    ax1.plot(rn, metrics["cum_rate"], color="#2196F3", linewidth=1.5,
             label="Cumulative success rate", zorder=3)
    valid = ~np.isnan(metrics["rolling_rate"])
    ax1.plot(rn[valid], metrics["rolling_rate"][valid],
             color="#FF9800", linewidth=1.2, alpha=0.85,
             label=f"Rolling {metrics['window']}-run window", zorder=3)

    fn = metrics["failure_run_numbers"]
    ax1.scatter(fn, np.full_like(fn, 60, dtype=float),
                marker="x", color="#E53935", s=15, alpha=0.6,
                label="Failed run", zorder=4)

    phase_colors = ["#E3F2FD", "#E8F5E9", "#FFF3E0", "#F3E5F5"]
    for i, p in enumerate(phases):
        ax1.axvspan(p["kg_size_start"], p["kg_size_end"],
                    alpha=0.18, color=phase_colors[i % len(phase_colors)],
                    zorder=0)
        mid = (p["kg_size_start"] + p["kg_size_end"]) / 2
        label_y = 58 if p["count"] < 30 else 63
        ax1.text(mid, label_y,
                 f"{p['name']}\n{p['count']} runs\n{p['success_rate']}%",
                 ha="center", va="bottom", fontsize=6, color="#555")

    ax1.set_xlabel("Run number (chronological ≈ KG size)", fontsize=9)
    ax1.set_ylabel("Success rate (%)", fontsize=9)
    ax1.set_ylim(55, 101)
    ax1.set_xlim(0, metrics["total"] + 10)
    ax1.legend(loc="lower right", fontsize=7, framealpha=0.9)
    ax1.grid(True, alpha=0.2)
    ax1.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def make_pgf_plot(metrics, phases, out_path):
    """Generate a TikZ/PGFPlots version for the paper."""
    rn = metrics["run_numbers"]
    cum = metrics["cum_rate"]
    roll = metrics["rolling_rate"]

    step = max(1, len(rn) // 200)
    coords_cum = " ".join(f"({rn[i]},{cum[i]:.1f})" for i in range(0, len(rn), step))
    valid_mask = ~np.isnan(roll)
    coords_roll = " ".join(
        f"({rn[i]},{roll[i]:.1f})" for i in range(0, len(rn), step) if valid_mask[i]
    )

    fail_coords = " ".join(
        f"({fn},62)" for fn in metrics["failure_run_numbers"]
    )

    phase_shading = ""
    for i, p in enumerate(phases):
        fill_colors = [
            "nodeblue!6", "nodegreen!6", "nodeorange!6", "nodered!6", "nodeblue!4"
        ]
        name = p.get("name", f"Phase {p['phase']}")
        phase_shading += (
            f"    \\fill[{fill_colors[i % 5]}] "
            f"(axis cs:{p['kg_size_start']},55) rectangle "
            f"(axis cs:{p['kg_size_end']},101);\n"
            f"    \\node[font=\\tiny, text=black!50, align=center] "
            f"at (axis cs:{(p['kg_size_start']+p['kg_size_end'])//2},63) "
            f"{{{name}\\\\{p['count']} runs\\\\{p['success_rate']}\\%}};\n"
        )

    tikz = f"""\
% KG growth analysis — success rate vs run number
\\begin{{tikzpicture}}
\\begin{{axis}}[
  width=0.95\\linewidth, height=5.5cm,
  xlabel={{Run number (chronological $\\approx$ KG size)}},
  ylabel={{Success rate (\\%)}},
  xmin=0, xmax={metrics['total']+10},
  ymin=55, ymax=101,
  grid=major, grid style={{black!10}},
  legend pos=south east,
  legend style={{font=\\tiny, fill opacity=0.9, draw opacity=0.5}},
  tick label style={{font=\\scriptsize}},
  label style={{font=\\small}},
]
{phase_shading}
  \\addplot[nodeblue, thick, no markers] coordinates {{{coords_cum}}};
  \\addlegendentry{{Cumulative success rate}}

  \\addplot[nodeorange, semithick, no markers] coordinates {{{coords_roll}}};
  \\addlegendentry{{Rolling {metrics['window']}-run window}}

  \\addplot[only marks, mark=x, mark size=1.5pt, nodered!70] coordinates {{{fail_coords}}};
  \\addlegendentry{{Failed run}}

\\end{{axis}}
\\end{{tikzpicture}}
"""
    with open(out_path, "w") as f:
        f.write(tikz)
    print(f"Saved: {out_path}")


def main():
    print("Fetching runs from PostgreSQL...")
    rows = fetch_runs()
    print(f"  {len(rows)} runs found")

    metrics = compute_metrics(rows)
    phases = compute_phase_stats(metrics)

    print(f"\n  Total: {metrics['total']}  Success: {metrics['total_success']}  "
          f"Failed: {metrics['total_fail']}  Rate: {metrics['cum_rate'][-1]:.1f}%\n")

    print("Phase breakdown:")
    for p in phases:
        print(f"  {p.get('name','Phase')} (Phase {p['phase']}): runs {p['runs']} "
              f"({p['date_start']} to {p['date_end']}), "
              f"n={p['count']}, success={p['success_rate']}%, "
              f"failures={p['failures']}")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    png_path = os.path.join(OUT_DIR, "kg_growth_analysis.png")
    make_plot(metrics, phases, png_path)

    tikz_path = os.path.join(FIG_DIR, "kg_growth.tikz")
    make_pgf_plot(metrics, phases, tikz_path)

    results = {
        "total_runs": metrics["total"],
        "total_success": metrics["total_success"],
        "total_failed": metrics["total_fail"],
        "overall_success_rate": round(float(metrics["cum_rate"][-1]), 1),
        "phases": phases,
        "neo4j_run_nodes": 1329,
        "neo4j_similar_to_edges": 6661,
        "neo4j_avg_similarity": 0.996,
    }
    json_path = os.path.join(OUT_DIR, "kg_growth_analysis.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
