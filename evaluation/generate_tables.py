#!/usr/bin/env python3
"""
Generate LaTeX tables and summary statistics from evaluation results.

Reads:
  evaluation/results/vv_results.json
  evaluation/results/ablation_results.json
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
    """Per-case h-refinement detail table."""
    cases = data.get("cases", {})

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Detailed convergence data: $L^2$ error norms at each mesh resolution.}",
        r"\label{tab:vv-detail}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Case & $N$ & DOFs & $\|e\|_{L^2}$ & $\|e\|_{L^\infty}$ \\",
        r"\midrule",
    ]

    for name, case in cases.items():
        display = case.get("case_description", name)[:30]
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
        r"\end{table}",
    ]
    return "\n".join(lines)


def generate_ablation_table(data: dict) -> str:
    """LaTeX table comparing KG-on vs KG-off aggregate metrics."""
    kg_on = data.get("kg_on", {}).get("aggregate", {})
    kg_off = data.get("kg_off", {}).get("aggregate", {})

    def fmt(v, is_pct=False):
        if is_pct:
            return f"{v*100:.1f}\\%"
        return f"{v:.2f}"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Ablation study: KG-augmented vs.\ baseline agent performance "
        r"across 10 benchmark tasks of varying difficulty.  "
        r"KG Off outperforms KG On on medium/hard tasks because the LLM's "
        r"parametric knowledge suffices for standard materials, while the "
        r"KG interaction overhead currently confuses the agent on complex "
        r"requests---pointing to a clear integration improvement opportunity.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Metric & KG On & KG Off & $\Delta$ \\",
        r"\midrule",
    ]

    metrics = [
        ("Success rate", "success_rate", True),
        ("First-try rate", "first_try_rate", True),
        ("Config quality", "avg_quality", False),
        (r"Avg.\ iterations", "avg_iterations", False),
        (r"Avg.\ wall time (s)", "avg_wall_time", False),
    ]

    for label, key, is_pct in metrics:
        v_on = kg_on.get(key, 0)
        v_off = kg_off.get(key, 0)
        delta = v_on - v_off
        lines.append(
            f"  {label} & {fmt(v_on, is_pct)} & {fmt(v_off, is_pct)} & "
            f"${'+' if delta >= 0 else ''}{fmt(delta, is_pct)}$ \\\\"
        )

    # By difficulty breakdown
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{4}{l}{\textit{Success rate by difficulty}} \\")
    for diff in ("easy", "medium", "hard"):
        on_sr = kg_on.get("by_difficulty", {}).get(diff, {}).get("success_rate", 0)
        off_sr = kg_off.get("by_difficulty", {}).get(diff, {}).get("success_rate", 0)
        delta = on_sr - off_sr
        lines.append(
            f"  \\quad {diff.capitalize()} & {on_sr*100:.0f}\\% & {off_sr*100:.0f}\\% & "
            f"${'+' if delta >= 0 else ''}{delta*100:.0f}\\%$ \\\\"
        )

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

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Agent ecosystem performance metrics from production database.}",
        r"\label{tab:agent-metrics}",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Metric & Value \\",
        r"\midrule",
        f"Total simulation runs & {db.get('total_runs', 0)} \\\\",
        f"Overall success rate & {db.get('overall_success_rate', 0)*100:.1f}\\% \\\\",
        f"Unique agent tasks & {db.get('unique_tasks', 0)} \\\\",
        f"First-try success rate & {orch.get('first_try_success_rate', 0)*100:.1f}\\% \\\\",
        f"Suggestion acceptance rate & {sugg.get('acceptance_rate', 0)*100:.1f}\\% \\\\",
    ]

    # Per-agent step counts
    for agent, stats in data.get("task_metrics", {}).items():
        lines.append(
            "Avg.\\ steps/task (" + agent + ") & " +
            f"{stats.get('avg_steps_per_task', 0):.1f} \\\\"
        )

    # Timing
    st = data.get("timing_metrics", {}).get("simulation_wall_time", {})
    if st.get("avg_s"):
        lines.append("Avg.\\ simulation time & " + f"{st['avg_s']:.1f}s \\\\")

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

    # Ablation results
    ablation_data = load_json("ablation_results.json")
    if ablation_data:
        table = generate_ablation_table(ablation_data)
        (TABLES_DIR / "ablation.tex").write_text(table)
        kg_on = ablation_data.get("kg_on", {}).get("aggregate", {})
        kg_off = ablation_data.get("kg_off", {}).get("aggregate", {})
        summary_lines.append(
            f"Ablation: KG On success={kg_on.get('success_rate', 0):.2f}, "
            f"KG Off success={kg_off.get('success_rate', 0):.2f}"
        )
        print(f"  [OK] Ablation table generated")
    else:
        print(f"  [--] Ablation results not found (run run_ablation.py first)")

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
