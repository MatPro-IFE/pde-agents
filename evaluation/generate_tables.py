#!/usr/bin/env python3
"""
Generate LaTeX tables and summary statistics from evaluation results.

Reads:
  evaluation/results/vv_results.json
  evaluation/results/ablation_v2_results.json   (preferred)
  evaluation/results/ablation_results.json      (v1 fallback)
  evaluation/results/agent_metrics.json

Produces:
  evaluation/results/tables/   — LaTeX .tex snippets for each table
  evaluation/results/summary.txt — Human-readable summary

Usage:
    python evaluation/generate_tables.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
TABLES_DIR = RESULTS_DIR / "tables"


def load_json(name: str) -> dict | None:
    path = RESULTS_DIR / name
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def generate_vv_table(data: dict) -> str:
    """Generate LaTeX table for V&V convergence study."""
    cases = data.get("cases", {})

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Verification: spatial convergence rates for the heat equation solver. "
        r"Expected rate is $\mathcal{O}(h^2)$ for $P_1$ elements.}",
        r"\label{tab:vv-convergence}",
        r"\begin{tabular}{lccccl}",
        r"\toprule",
        r"Benchmark Case & DOFs (finest) & $\|e\|_{L^2}$ (finest) & Rate ($L^2$) & Expected & Status \\",
        r"\midrule",
    ]

    for name, case in cases.items():
        display_name = case.get("case_description", name).replace("_", " ")
        n_dofs = case["n_dofs"][-1] if case.get("n_dofs") else "---"
        l2_errors = case.get("l2_errors", [])
        l2_finest = l2_errors[-1] if l2_errors else 0
        max_l2 = max(l2_errors) if l2_errors else 1.0
        rate = case.get("convergence_rate_l2", 0)
        expected = case.get("expected_rate", 2)
        passed = case.get("passed", False)
        status = r"\checkmark" if passed else r"\texttimes"

        # Cases where P1 is algebraically exact: show "exact" instead of rate
        is_exact = max_l2 < 1e-6
        rate_str = r"\textit{exact}" if is_exact else f"{rate:.2f}"

        lines.append(
            f"  {display_name} & {n_dofs:,} & {l2_finest:.2e} & "
            f"{rate_str} & {expected:.1f} & {status} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def generate_vv_convergence_detail(data: dict) -> str:
    """Per-case h-refinement detail table.

    The caption records how the two norms are computed, because they
    disagree by orders of magnitude on the constant-source case and that
    difference is meaningful rather than an error (see below).
    """
    cases = data.get("cases", {})

    # table* spans both columns: the full case descriptions plus four numeric
    # columns do not fit a single column at this font size.
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering\footnotesize",
        r"\caption{Detailed convergence data: error norms at each mesh "
        r"resolution.  $\|e\|_{L^2}$ is integrated over each element with "
        r"quadrature of degree 8, so it captures the error \emph{between} "
        r"nodes; $\|e\|_{L^\infty}$ is evaluated at the degrees of freedom "
        r"only.  For the constant-source case the P1 solution is nodally "
        r"exact, so $\|e\|_{L^\infty}$ sits at machine precision while "
        r"$\|e\|_{L^2}$ shows the expected $\mathcal{O}(h^2)$ interpolation "
        r"error.  The two norms measure different things, and only the "
        r"integrated norm is used to fit convergence rates.}",
        r"\label{tab:vv-detail}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Case & $N$ & DOFs & $\|e\|_{L^2}$ & $\|e\|_{L^\infty}$ \\",
        r"\midrule",
    ]

    for name, case in cases.items():
        display = case.get("case_description", name)
        for i, nx in enumerate(case.get("resolutions", [])):
            dofs = case["n_dofs"][i]
            l2 = case["l2_errors"][i]
            linf = case["linf_errors"][i]
            prefix = display if i == 0 else ""
            lines.append(
                f"  {prefix} & {nx} & {dofs:,} & {l2:.2e} & {linf:.2e} \\\\"
            )
        lines.append(r"\addlinespace")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


_ABLATION_MODES = (("KG On", "kg_on"), ("KG Off", "kg_off"), ("KG Smart", "kg_smart"))

_ABLATION_METRICS = (
    ("Success rate",            "success_rate",           True),
    ("First-try rate",          "first_try_rate",         True),
    ("Config quality",          "avg_quality",            False),
    ("Property fidelity (MPF)", "avg_property_fidelity",  False),
    ("Physics score",           "avg_physics_score",      False),
    (r"Avg.\ iterations",       "avg_iterations",         False),
    (r"Avg.\ wall time (s)",    "avg_wall_time",          False),
)


def generate_ablation_table(data: dict) -> str:
    """LaTeX table comparing KG modes on aggregate ablation metrics.

    Handles both the 2-way (On/Off) and 3-way (On/Off/Smart) result layouts,
    and derives the task count and difficulty levels from the data rather
    than hard-coding them: the same generator is used for the 10-task v1 and
    50-task v2 runs, and a stale hard-coded caption would silently
    misattribute one run's scope to the other.

    The caption states scope only. Interpretation belongs in the paper body,
    where it can be qualified against the full result set.
    """
    modes = [(label, data.get(key, {}).get("aggregate", {}))
             for label, key in _ABLATION_MODES]
    modes = [(label, agg) for label, agg in modes if agg]
    if not modes:
        return ""

    meta = data.get("metadata", {})
    first_agg = modes[0][1]
    n_tasks = (data.get("n_tasks") or meta.get("n_tasks")
               or first_agg.get("n") or first_agg.get("n_tasks"))
    scope = f"{n_tasks} benchmark tasks" if n_tasks else "the benchmark suite"
    frozen = " with a frozen knowledge graph" if meta.get("kg_read_only") else ""

    def fmt(v, is_pct=False):
        if v is None:
            return "--"
        return f"{v*100:.1f}\\%" if is_pct else f"{v:.2f}"

    col_spec = "l" + "c" * len(modes)
    header = " & ".join(label for label, _ in modes)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{Ablation study: knowledge graph integration modes across "
        rf"{scope}{frozen}.  Best value per row in \textbf{{bold}}.}}",
        r"\label{tab:ablation}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf"Metric & {header} \\",
        r"\midrule",
    ]

    # Higher is better for every metric except iterations and wall time.
    lower_is_better = {"avg_iterations", "avg_wall_time"}

    for label, key, is_pct in _ABLATION_METRICS:
        values = [agg.get(key) for _, agg in modes]
        present = [v for v in values if v is not None]
        if not present:
            continue
        best = min(present) if key in lower_is_better else max(present)
        cells = []
        for v in values:
            text = fmt(v, is_pct)
            cells.append(rf"\textbf{{{text}}}" if v == best else text)
        lines.append(f"  {label} & " + " & ".join(cells) + r" \\")

    # Success rate by difficulty, using whichever levels the data contains.
    levels: list[str] = []
    for _, agg in modes:
        for lvl in agg.get("by_difficulty", {}):
            if lvl not in levels:
                levels.append(lvl)
    if levels:
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{len(modes)+1}}}{{l}}"
                     r"{\textit{Success rate by difficulty}} \\")
        for lvl in levels:
            cells = []
            for _, agg in modes:
                sr = agg.get("by_difficulty", {}).get(lvl, {}).get("success_rate")
                cells.append("--" if sr is None else f"{sr*100:.0f}\\%")
            lines.append(rf"  \quad {lvl.capitalize()} & " + " & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def generate_agent_metrics_table(data: dict) -> str:
    """LaTeX table of agent decision quality metrics."""
    db = data.get("db_stats", {})
    orch = data.get("orchestrator_metrics", {})
    sugg = data.get("suggestion_metrics", {})
    window = data.get("window", {})

    # State the reporting window in the caption: every row must be traceable
    # to one population of runs.
    scope = ""
    if window.get("end"):
        start = window.get("start")
        span = f"{start} to {window['end']}" if start else f"through {window['end']}"
        scope = (f"  FEniCSx runs, {span} (excludes the controlled ablation "
                 f"campaign and non-FEniCSx backends).")

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Agent ecosystem performance metrics from production database."
        + scope + r"}",
        r"\label{tab:agent-metrics}",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Metric & Value \\",
        r"\midrule",
        f"Total simulation runs & {db.get('total_runs', 0)} \\\\",
        f"Overall success rate & {db.get('overall_success_rate', 0)*100:.1f}\\% \\\\",
        f"Unique agent tasks & {db.get('unique_tasks', 0)} \\\\",
        f"First-try success rate & {orch.get('first_try_success_rate', 0)*100:.1f}\\% \\\\",
    ]
    if "retry_rate" in orch:
        lines.append(f"Retry rate & {orch['retry_rate']*100:.1f}\\% \\\\")
    lines.append(
        f"Suggestion acceptance rate & {sugg.get('acceptance_rate', 0)*100:.1f}\\% \\\\"
    )

    # Per-agent step counts (sorted so the table is reproducible)
    for agent, stats in sorted(data.get("task_metrics", {}).items()):
        lines.append(
            "Avg.\\ steps/task (" + agent + ") & " +
            f"{stats.get('avg_steps_per_task', 0):.1f} \\\\"
        )

    # Timing
    st = data.get("timing_metrics", {}).get("simulation_wall_time", {})
    if st.get("avg_s"):
        lines.append("Avg.\\ simulation time & " + f"{st['avg_s']:.2f}s \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary_lines = []

    print(f"\n{'='*60}")
    print(f"  GENERATING PAPER TABLES")
    print(f"{'='*60}\n")

    # V&V results
    vv_data = load_json("vv_results.json")
    if vv_data:
        table = generate_vv_table(vv_data)
        (TABLES_DIR / "vv_convergence.tex").write_text(table)
        detail = generate_vv_convergence_detail(vv_data)
        (TABLES_DIR / "vv_detail.tex").write_text(detail)
        summary_lines.append(f"V&V: {len(vv_data.get('cases', {}))} cases, "
                           f"all_passed={vv_data.get('all_passed')}")
        print(f"  [OK] V&V tables generated ({len(vv_data.get('cases', {}))} cases)")
    else:
        print(f"  [--] V&V results not found (run vv_runner.py first)")

    # Ablation results.  The paper reports the 50-task v2 run, so prefer it
    # and fall back to v1 only when v2 has not been produced.  Each is written
    # to its own file so a v1 run can never silently overwrite the v2 table.
    for source, out_name in (("ablation_v2_results.json", "ablation.tex"),
                             ("ablation_results.json", "ablation_v1.tex")):
        ablation_data = load_json(source)
        if not ablation_data:
            print(f"  [--] {source} not found")
            continue
        table = generate_ablation_table(ablation_data)
        if not table:
            print(f"  [--] {source} has no mode aggregates; skipped")
            continue
        (TABLES_DIR / out_name).write_text(table)
        parts = []
        for label, key in _ABLATION_MODES:
            agg = ablation_data.get(key, {}).get("aggregate", {})
            if agg:
                parts.append(f"{label} success={agg.get('success_rate', 0):.2f}")
        summary_lines.append(f"Ablation ({source}): " + ", ".join(parts))
        print(f"  [OK] {out_name} generated from {source}")

    # Agent metrics
    metrics_data = load_json("agent_metrics.json")
    if metrics_data:
        table = generate_agent_metrics_table(metrics_data)
        (TABLES_DIR / "agent_metrics.tex").write_text(table)
        db = metrics_data.get("db_stats", {})
        summary_lines.append(
            f"Agent metrics: {db.get('total_runs', 0)} runs, "
            f"success_rate={db.get('overall_success_rate', 0):.2f}"
        )
        print(f"  [OK] Agent metrics table generated")
    else:
        print(f"  [--] Agent metrics not found (run agent_quality.py first)")

    # Write summary
    summary_file = RESULTS_DIR / "summary.txt"
    summary_file.write_text("\n".join(summary_lines) + "\n")
    print(f"\n  Summary: {summary_file}")
    print(f"  Tables:  {TABLES_DIR}/")


if __name__ == "__main__":
    main()
