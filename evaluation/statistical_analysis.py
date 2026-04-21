#!/usr/bin/env python3
"""
Statistical analysis for ablation v2 results.

Produces:
  - 95% confidence intervals (Wilson score for proportions)
  - Fisher exact test / chi-squared for success rate differences
  - Effect sizes (Cohen's h for proportions, Cohen's d for continuous)
  - Per-difficulty breakdown with CIs
  - Both overall and success-only quality metrics
  - LaTeX table output for direct paper insertion
  - PGFPlots-ready data files

Usage:
    python evaluation/statistical_analysis.py evaluation/results/ablation_v2_results.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score confidence interval for a proportion.

    Returns (proportion, lower, upper).
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return p, max(0, center - spread), min(1, center + spread)


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for comparing two proportions."""
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def cohens_d(mean1: float, std1: float, n1: int,
             mean2: float, std2: float, n2: int) -> float:
    """Cohen's d effect size for comparing two means."""
    if n1 < 2 or n2 < 2:
        return 0.0
    pooled_std = math.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (mean1 - mean2) / pooled_std


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """Approximate p-value for 2x2 contingency table using chi-squared."""
    n = a + b + c + d
    if n == 0:
        return 1.0
    expected = [(a + b) * (a + c) / n, (a + b) * (b + d) / n,
                (c + d) * (a + c) / n, (c + d) * (b + d) / n]
    observed = [a, b, c, d]
    chi2 = sum((o - e)**2 / max(e, 0.001) for o, e in zip(observed, expected))
    if chi2 == 0:
        return 1.0
    x = math.sqrt(chi2)
    p = math.erfc(x / math.sqrt(2))
    return p


def _mean_std(vals):
    if not vals:
        return 0.0, 0.0
    m = sum(vals) / len(vals)
    s = (sum((v - m)**2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
    return m, s


def _get_pf(task):
    """Get property fidelity score from task, handling field name variants."""
    return task.get("property_fidelity") or task.get("mpf") or 0.0


def analyze_ablation(results_path: str) -> dict:
    """Run full statistical analysis on ablation v2 results."""
    with open(results_path) as f:
        data = json.load(f)

    modes = [m for m in data.keys() if m != "metadata"]
    n_tasks = len(data[modes[0]]["tasks"])

    print(f"\n{'='*80}")
    print(f"  STATISTICAL ANALYSIS — Ablation v2 ({n_tasks} tasks/mode)")
    print(f"{'='*80}")

    analysis = {"modes": {}}

    for mode in modes:
        tasks = data[mode]["tasks"]
        n = len(tasks)
        n_success = sum(1 for t in tasks if t["success"])
        sr, sr_lo, sr_hi = wilson_ci(n_success, n)

        phys_all = [t.get("physics_score", 0.5) for t in tasks]
        pf_all = [_get_pf(t) for t in tasks]
        wt_all = [t.get("wall_time_s", 0) for t in tasks]

        ok_tasks = [t for t in tasks if t["success"]]
        phys_ok = [t.get("physics_score", 0.5) for t in ok_tasks]
        pf_ok = [_get_pf(t) for t in ok_tasks]
        wt_ok = [t.get("wall_time_s", 0) for t in ok_tasks]

        phys_mean, phys_std = _mean_std(phys_all)
        pf_mean, pf_std = _mean_std(pf_all)
        wt_mean, wt_std = _mean_std(wt_all)

        phys_ok_mean, phys_ok_std = _mean_std(phys_ok)
        pf_ok_mean, pf_ok_std = _mean_std(pf_ok)
        wt_ok_mean, wt_ok_std = _mean_std(wt_ok)

        analysis["modes"][mode] = {
            "n": n,
            "n_success": n_success,
            "success_rate": sr,
            "sr_ci_lo": sr_lo,
            "sr_ci_hi": sr_hi,
            # Overall (all tasks)
            "physics_score_mean": phys_mean,
            "physics_score_std": phys_std,
            "mpf_mean": pf_mean,
            "mpf_std": pf_std,
            "wall_time_mean": wt_mean,
            "wall_time_std": wt_std,
            # Success-only
            "physics_ok_mean": phys_ok_mean,
            "physics_ok_std": phys_ok_std,
            "mpf_ok_mean": pf_ok_mean,
            "mpf_ok_std": pf_ok_std,
            "wall_time_ok_mean": wt_ok_mean,
            "wall_time_ok_std": wt_ok_std,
        }

        print(f"\n  {mode.upper()} (n={n}, ok={n_success}):")
        print(f"    Success rate: {sr:.0%} [{sr_lo:.0%}, {sr_hi:.0%}]")
        print(f"    Overall:      phys={phys_mean:.3f}±{phys_std:.3f}  "
              f"mpf={pf_mean:.3f}±{pf_std:.3f}  wt={wt_mean:.1f}±{wt_std:.1f}s")
        print(f"    Success-only: phys={phys_ok_mean:.3f}±{phys_ok_std:.3f}  "
              f"mpf={pf_ok_mean:.3f}±{pf_ok_std:.3f}  wt={wt_ok_mean:.1f}±{wt_ok_std:.1f}s")

    # Pairwise comparisons (KG Smart as reference)
    print(f"\n{'─'*80}")
    print(f"  PAIRWISE COMPARISONS (KG Smart as reference)")
    print(f"{'─'*80}")

    ref = "kg_smart"
    comparisons = {}
    if ref in analysis["modes"]:
        ref_data = data[ref]["tasks"]
        ref_n = len(ref_data)
        ref_succ = sum(1 for t in ref_data if t["success"])

        for mode in modes:
            if mode == ref:
                continue
            mode_data = data[mode]["tasks"]
            mode_n = len(mode_data)
            mode_succ = sum(1 for t in mode_data if t["success"])

            ref_sr = ref_succ / ref_n
            mode_sr = mode_succ / mode_n
            h = cohens_h(ref_sr, mode_sr)
            p_val = fisher_exact_2x2(
                ref_succ, ref_n - ref_succ,
                mode_succ, mode_n - mode_succ,
            )

            ref_phys = [t.get("physics_score", 0.5) for t in ref_data]
            mode_phys = [t.get("physics_score", 0.5) for t in mode_data]
            ref_pm, ref_ps = _mean_std(ref_phys)
            mode_pm, mode_ps = _mean_std(mode_phys)
            d_phys = cohens_d(ref_pm, ref_ps, ref_n, mode_pm, mode_ps, mode_n)

            # Success-only comparisons
            ref_ok = [t for t in ref_data if t["success"]]
            mode_ok = [t for t in mode_data if t["success"]]
            ref_ok_phys = [t.get("physics_score", 0.5) for t in ref_ok]
            mode_ok_phys = [t.get("physics_score", 0.5) for t in mode_ok]
            ref_ok_pm, ref_ok_ps = _mean_std(ref_ok_phys)
            mode_ok_pm, mode_ok_ps = _mean_std(mode_ok_phys)
            d_phys_ok = cohens_d(ref_ok_pm, ref_ok_ps, len(ref_ok),
                                 mode_ok_pm, mode_ok_ps, len(mode_ok))

            comp = {
                "sr_diff": ref_sr - mode_sr,
                "cohens_h": h,
                "p_value_sr": p_val,
                "phys_diff": ref_pm - mode_pm,
                "cohens_d_phys": d_phys,
                "phys_ok_diff": ref_ok_pm - mode_ok_pm,
                "cohens_d_phys_ok": d_phys_ok,
            }
            comparisons[f"{ref}_vs_{mode}"] = comp

            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
            print(f"\n  KG Smart vs {mode}:")
            print(f"    SR diff: {comp['sr_diff']:+.0%}  p={p_val:.4f} {sig}  h={h:.3f}")
            print(f"    Phys diff (all):     {comp['phys_diff']:+.3f}  d={d_phys:.3f}")
            print(f"    Phys diff (ok-only): {comp['phys_ok_diff']:+.3f}  d={d_phys_ok:.3f}")

    analysis["comparisons"] = comparisons

    # Per-difficulty analysis
    print(f"\n{'─'*80}")
    print(f"  PER-DIFFICULTY ANALYSIS")
    print(f"{'─'*80}")

    difficulties = sorted(set(t["difficulty"] for t in data[modes[0]]["tasks"]))
    analysis["by_difficulty"] = {}

    for diff in difficulties:
        print(f"\n  {diff.upper()}:")
        analysis["by_difficulty"][diff] = {}
        for mode in modes:
            tasks_d = [t for t in data[mode]["tasks"] if t["difficulty"] == diff]
            n = len(tasks_d)
            n_s = sum(1 for t in tasks_d if t["success"])
            sr, lo, hi = wilson_ci(n_s, n)
            phys = [t.get("physics_score", 0.5) for t in tasks_d]
            pf = [_get_pf(t) for t in tasks_d]
            pm = sum(phys) / max(len(phys), 1)
            pfm = sum(pf) / max(len(pf), 1)

            ok_d = [t for t in tasks_d if t["success"]]
            ok_phys = [t.get("physics_score", 0.5) for t in ok_d]
            ok_pf = [_get_pf(t) for t in ok_d]
            ok_pm = sum(ok_phys) / max(len(ok_phys), 1) if ok_phys else 0
            ok_pfm = sum(ok_pf) / max(len(ok_pf), 1) if ok_pf else 0

            analysis["by_difficulty"][diff][mode] = {
                "n": n, "sr": sr, "sr_ci_lo": lo, "sr_ci_hi": hi,
                "phys_mean": pm, "pf_mean": pfm,
                "phys_ok_mean": ok_pm, "pf_ok_mean": ok_pfm,
            }
            print(f"    {mode:<16s}: SR={sr:3.0%} [{lo:3.0%},{hi:3.0%}]  "
                  f"phys={pm:.3f}  pf={pfm:.3f}  "
                  f"(ok-only: phys={ok_pm:.3f} pf={ok_pfm:.3f})  n={n}")

    return analysis


def generate_latex_table(analysis: dict, out_path: str):
    """Generate a LaTeX table for the paper — shows both overall and success-only."""
    modes_order = ["kg_off", "kg_on", "kg_smart"]
    modes = [m for m in modes_order if m in analysis["modes"]]
    labels = {"kg_on": "KG On", "kg_off": "KG Off", "kg_smart": "KG Smart"}

    n = analysis["modes"][modes[0]]["n"]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{Ablation study results (frozen KG, $n={n}$ tasks per mode). "
        r"``Overall'' includes failed runs (phys${}=0.5$, MPF${}=0$); "
        r"``Success-only'' isolates output quality.}",
        r"\label{tab:ablation-v2}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l" + "c" * len(modes) + "}",
        r"\toprule",
    ]

    header = "Metric"
    for m in modes:
        header += f" & \\textbf{{{labels.get(m, m)}}}"
    lines.append(header + r" \\")
    lines.append(r"\midrule")

    # Success rate row
    row = "Success rate"
    for m in modes:
        d = analysis["modes"][m]
        best = max(analysis["modes"][mm]["success_rate"] for mm in modes)
        val = f"{d['success_rate']:.0%} [{d['sr_ci_lo']:.0%}, {d['sr_ci_hi']:.0%}]"
        if d["success_rate"] == best:
            val = f"\\textbf{{{val}}}"
        row += f" & {val}"
    lines.append(row + r" \\[2pt]")

    lines.append(r"\multicolumn{" + str(len(modes) + 1) + r"}{l}{\textit{Overall (all tasks):}} \\")

    # Physics score overall
    row = "~~Physics score"
    for m in modes:
        d = analysis["modes"][m]
        row += f" & {d['physics_score_mean']:.3f} $\\pm$ {d['physics_score_std']:.3f}"
    lines.append(row + r" \\")

    # MPF overall
    row = "~~MPF"
    for m in modes:
        d = analysis["modes"][m]
        row += f" & {d['mpf_mean']:.3f} $\\pm$ {d['mpf_std']:.3f}"
    lines.append(row + r" \\[2pt]")

    lines.append(r"\multicolumn{" + str(len(modes) + 1) + r"}{l}{\textit{Success-only:}} \\")

    # Physics score success-only (bold the best)
    row = "~~Physics score"
    best_phys = max(analysis["modes"][mm]["physics_ok_mean"] for mm in modes)
    for m in modes:
        d = analysis["modes"][m]
        val = f"{d['physics_ok_mean']:.3f} $\\pm$ {d['physics_ok_std']:.3f}"
        if d["physics_ok_mean"] == best_phys:
            val = f"\\textbf{{{val}}}"
        row += f" & {val}"
    lines.append(row + r" \\")

    # MPF success-only (bold the best)
    row = "~~MPF"
    best_mpf = max(analysis["modes"][mm]["mpf_ok_mean"] for mm in modes)
    for m in modes:
        d = analysis["modes"][m]
        val = f"{d['mpf_ok_mean']:.3f} $\\pm$ {d['mpf_ok_std']:.3f}"
        if d["mpf_ok_mean"] == best_mpf:
            val = f"\\textbf{{{val}}}"
        row += f" & {val}"
    lines.append(row + r" \\[2pt]")

    # Wall time (bold the fastest)
    row = "Wall time (s)"
    best_wt = min(analysis["modes"][mm]["wall_time_mean"] for mm in modes)
    for m in modes:
        d = analysis["modes"][m]
        val = f"{d['wall_time_mean']:.1f} $\\pm$ {d['wall_time_std']:.1f}"
        if d["wall_time_mean"] == best_wt:
            val = f"\\textbf{{{val}}}"
        row += f" & {val}"
    lines.append(row + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nLaTeX table saved: {out_path}")


def generate_pgf_data(analysis: dict, out_dir: str):
    """Generate PGFPlots-ready data files."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    with open(f"{out_dir}/ablation_v2_physics.dat", "w") as f:
        f.write("mode sr sr_ci_lo sr_ci_hi phys phys_std mpf mpf_std "
                "phys_ok phys_ok_std mpf_ok mpf_ok_std wt wt_std\n")
        for mode, d in analysis["modes"].items():
            label = {"kg_on": "KG_On", "kg_off": "KG_Off",
                     "kg_smart": "KG_Smart"}.get(mode, mode)
            f.write(f"{label} "
                    f"{d['success_rate']:.4f} {d['sr_ci_lo']:.4f} {d['sr_ci_hi']:.4f} "
                    f"{d['physics_score_mean']:.4f} {d['physics_score_std']:.4f} "
                    f"{d['mpf_mean']:.4f} {d['mpf_std']:.4f} "
                    f"{d['physics_ok_mean']:.4f} {d['physics_ok_std']:.4f} "
                    f"{d['mpf_ok_mean']:.4f} {d['mpf_ok_std']:.4f} "
                    f"{d['wall_time_mean']:.4f} {d['wall_time_std']:.4f}\n")

    for diff, modes_data in analysis.get("by_difficulty", {}).items():
        with open(f"{out_dir}/ablation_v2_{diff}.dat", "w") as f:
            f.write("mode sr sr_ci_lo sr_ci_hi phys_mean pf_mean n\n")
            for mode, d in modes_data.items():
                label = {"kg_on": "KG_On", "kg_off": "KG_Off",
                         "kg_smart": "KG_Smart"}.get(mode, mode)
                f.write(f"{label} {d['sr']:.4f} {d['sr_ci_lo']:.4f} {d['sr_ci_hi']:.4f} "
                        f"{d['phys_mean']:.4f} {d['pf_mean']:.4f} {d['n']}\n")

    print(f"PGFPlots data saved: {out_dir}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python statistical_analysis.py <results_json>")
        sys.exit(1)

    results_path = sys.argv[1]
    analysis = analyze_ablation(results_path)

    out_path = Path(results_path).parent / "ablation_v2_statistics.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nStatistics saved: {out_path}")

    table_path = Path(results_path).parent.parent.parent / "paper" / "tables" / "ablation_v2.tex"
    generate_latex_table(analysis, str(table_path))

    pgf_dir = Path(results_path).parent.parent.parent / "paper" / "data"
    generate_pgf_data(analysis, str(pgf_dir))
