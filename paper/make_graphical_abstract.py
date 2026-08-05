#!/usr/bin/env python3
"""Graphical abstract for PDE-Agents CMAME paper.

Elsevier spec: min 1328x531 px, 500:200 ratio, 300 DPI, white bg.
We render at 3x for crisp output: 3984 x 1593 px.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

# ── Modern colour palette (3 hues + neutrals) ───────────────────────────────
BLUE       = "#2563EB"
BLUE_50    = "#EFF6FF"
BLUE_100   = "#DBEAFE"
BLUE_200   = "#BFDBFE"
TEAL       = "#0D9488"
TEAL_50    = "#F0FDFA"
TEAL_100   = "#CCFBF1"
AMBER      = "#D97706"
AMBER_50   = "#FFFBEB"
AMBER_100  = "#FEF3C7"
SLATE_900  = "#0F172A"
SLATE_700  = "#334155"
SLATE_500  = "#64748B"
SLATE_300  = "#CBD5E1"
SLATE_200  = "#E2E8F0"
SLATE_100  = "#F1F5F9"
SLATE_50   = "#F8FAFC"
WHITE      = "#FFFFFF"
RED_600    = "#DC2626"
GREEN_600  = "#16A34A"

DPI = 300
W, H = 13.28, 5.31  # inches → 3984 x 1593 px at 300 DPI

fig, ax = plt.subplots(figsize=(W, H), dpi=DPI)
ax.set_xlim(0, 100)
ax.set_ylim(0, 40)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor(WHITE)
ax.set_facecolor(WHITE)

FONT = "Roboto"


def pill(x, y, w, h, fc, ec=None, lw=1.2, zorder=2, alpha=1.0):
    """Rounded rectangle with large corner radius (pill shape)."""
    r = min(h / 3, 0.8)
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad={r}",
        facecolor=fc, edgecolor=ec or fc,
        linewidth=lw, alpha=alpha, zorder=zorder)
    ax.add_patch(box)
    return box


def flow_arrow(x1, y1, x2, y2, color=SLATE_300, lw=2.0, zorder=3):
    """Clean directional arrow."""
    ar = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", color=color,
        linewidth=lw, mutation_scale=18, zorder=zorder)
    ax.add_patch(ar)


# ═══════════════════════════════════════════════════════════════════════════
#  PANEL 1 — INPUT (left, ~18%)
# ═══════════════════════════════════════════════════════════════════════════
# Section background
pill(1, 1, 16, 38, SLATE_50, SLATE_200, lw=1.0, zorder=1)

ax.text(9, 36, "Natural Language", fontsize=11, fontweight="bold",
        color=SLATE_900, ha="center", va="center", fontfamily=FONT)
ax.text(9, 34, "Input", fontsize=11, fontweight="bold",
        color=SLATE_900, ha="center", va="center", fontfamily=FONT)

# Chat bubble
pill(2.5, 22, 13, 9, WHITE, BLUE_200, lw=1.5, zorder=3)
ax.text(9, 28.5, '"Simulate heat transfer', fontsize=8,
        color=SLATE_700, ha="center", va="center", fontfamily=FONT,
        fontstyle="italic")
ax.text(9, 26, 'in a steel plate', fontsize=8,
        color=SLATE_700, ha="center", va="center", fontfamily=FONT,
        fontstyle="italic")
ax.text(9, 23.5, 'at 500 K"', fontsize=8,
        color=SLATE_700, ha="center", va="center", fontfamily=FONT,
        fontstyle="italic")

# User circle
circle = Circle((9, 16), 2.5, facecolor=BLUE_100, edgecolor=BLUE,
                linewidth=1.5, zorder=3)
ax.add_patch(circle)
# Simple user silhouette using shapes
head = Circle((9, 17.2), 0.9, facecolor=BLUE, edgecolor="none", zorder=4)
ax.add_patch(head)
body = FancyBboxPatch((7.2, 14), 3.6, 2.2, boxstyle="round,pad=0.4",
                       facecolor=BLUE, edgecolor="none", zorder=4)
ax.add_patch(body)

ax.text(9, 10.5, "Domain Expert", fontsize=7.5, color=SLATE_500,
        ha="center", va="center", fontfamily=FONT)
ax.text(9, 8.5, "or Researcher", fontsize=7.5, color=SLATE_500,
        ha="center", va="center", fontfamily=FONT)

# ═══════════════════════════════════════════════════════════════════════════
#  FLOW ARROW 1
# ═══════════════════════════════════════════════════════════════════════════
flow_arrow(17.5, 20, 20.5, 20, color=BLUE, lw=2.5)

# ═══════════════════════════════════════════════════════════════════════════
#  PANEL 2 — SYSTEM (center, ~42%)
# ═══════════════════════════════════════════════════════════════════════════
pill(21, 1, 40, 38, WHITE, BLUE, lw=2.0, zorder=1)

# Title
ax.text(41, 37, "PDE-Agents", fontsize=13, fontweight="bold",
        color=BLUE, ha="center", va="center", fontfamily=FONT)
ax.text(41, 35, "Multi-Agent Orchestrator", fontsize=9,
        color=SLATE_500, ha="center", va="center", fontfamily=FONT)

# ── Supervisor bar ──
pill(24, 30.5, 34, 3, BLUE, BLUE, lw=0, zorder=3, alpha=0.1)
pill(24, 30.5, 34, 3, "none", BLUE, lw=1.5, zorder=4)
ax.text(41, 32, "LangGraph Supervisor Router", fontsize=8,
        fontweight="bold", color=BLUE, ha="center", va="center",
        fontfamily=FONT)

# ── Three agent cards ──
agents = [
    ("Simulation", BLUE, BLUE_50, "SIM"),
    ("Analytics", TEAL, TEAL_50, "ANA"),
    ("Database", AMBER, AMBER_50, "DB"),
]

card_w, card_h = 10.5, 10
card_y = 18.5
card_xs = [24, 35.5, 47]

for i, (name, color, bg, abbr) in enumerate(agents):
    x = card_xs[i]
    pill(x, card_y, card_w, card_h, bg, color, lw=1.5, zorder=3)

    # Icon circle
    icon_c = Circle((x + card_w/2, card_y + 7.5), 1.5,
                     facecolor=color, edgecolor="none", zorder=4)
    ax.add_patch(icon_c)
    ax.text(x + card_w/2, card_y + 7.5, abbr, fontsize=6.5,
            fontweight="bold", color=WHITE, ha="center", va="center",
            fontfamily=FONT, zorder=5)

    ax.text(x + card_w/2, card_y + 4.2, f"{name}", fontsize=8,
            fontweight="bold", color=color, ha="center", va="center",
            fontfamily=FONT)
    ax.text(x + card_w/2, card_y + 2.5, "Agent", fontsize=8,
            fontweight="bold", color=color, ha="center", va="center",
            fontfamily=FONT)

    # Connector: supervisor → agent
    flow_arrow(x + card_w/2, 30.5, x + card_w/2, card_y + card_h,
               color=SLATE_300, lw=1.2)

# ── LLM bar at bottom ──
pill(24, 14, 34, 3.5, SLATE_50, SLATE_200, lw=1.0, zorder=3)
ax.text(41, 15.75, "Local LLMs: Qwen3-Coder-Next  |  Llama 4 Scout  |"
        "  2\u00d7 RTX PRO 6000",
        fontsize=6.5, color=SLATE_500, ha="center", va="center",
        fontfamily=FONT)

# ── Knowledge Graph section ──
pill(24, 2, 34, 11, TEAL_50, TEAL, lw=1.5, zorder=2)
ax.text(41, 11.8, "GraphRAG Knowledge Base", fontsize=8.5,
        fontweight="bold", color=TEAL, ha="center", va="center",
        fontfamily=FONT)

# Clean graph visualization
nodes = [
    (30, 7.5, "Run", 1.5),
    (36, 9, "Material", 1.2),
    (36, 5, "Issue", 1.2),
    (46, 9, "BC", 1.0),
    (46, 5, "Ref", 1.0),
    (52, 7.5, "Domain", 1.0),
]
edges = [
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
    (1, 3), (2, 4), (3, 5),
]
for n1, n2 in edges:
    ax.plot([nodes[n1][0], nodes[n2][0]],
            [nodes[n1][1], nodes[n2][1]],
            color=TEAL, alpha=0.25, lw=1.2, zorder=3)
for nx_, ny_, label, r in nodes:
    c = Circle((nx_, ny_), r, facecolor=TEAL, edgecolor=WHITE,
               linewidth=1.2, alpha=0.8, zorder=4)
    ax.add_patch(c)
    ax.text(nx_, ny_, label, fontsize=5 if r >= 1.2 else 4.5,
            fontweight="bold", color=WHITE, ha="center", va="center",
            fontfamily=FONT, zorder=5)

ax.text(41, 2.8, "Neo4j  \u00b7  768-dim embeddings  \u00b7  "
        "HNSW vector search  \u00b7  warm-start injection",
        fontsize=5.5, color=TEAL, ha="center", va="center",
        fontfamily=FONT, fontstyle="italic")

# ═══════════════════════════════════════════════════════════════════════════
#  FLOW ARROW 2
# ═══════════════════════════════════════════════════════════════════════════
flow_arrow(61.5, 20, 64.5, 20, color=BLUE, lw=2.5)

# ═══════════════════════════════════════════════════════════════════════════
#  PANEL 3 — RESULTS (right, ~35%)
# ═══════════════════════════════════════════════════════════════════════════

# ── FEM equation + solver ──
pill(65, 27, 33.5, 12, BLUE_50, BLUE_200, lw=1.2, zorder=1)

ax.text(81.75, 37, "FEM Simulation Output", fontsize=10,
        fontweight="bold", color=SLATE_900, ha="center", va="center",
        fontfamily=FONT)

ax.text(81.75, 34, r"$\rho\, c_p\, \dfrac{\partial u}{\partial t}"
        r" - \nabla \!\cdot\! (k\,\nabla u) = f$",
        fontsize=12, color=SLATE_900, ha="center", va="center")

ax.text(81.75, 31, "DOLFINx / FEniCSx  \u00b7  PETSc KSP  \u00b7  "
        "\u03b8-scheme  \u00b7  Gmsh geometries",
        fontsize=6, color=SLATE_500, ha="center", va="center",
        fontfamily=FONT)

# Convergence badge
pill(67, 28, 9, 2.3, WHITE, BLUE, lw=1.2, zorder=3)
ax.text(71.5, 29.15, r"$\mathcal{O}(h^2)$ verified", fontsize=7,
        fontweight="bold", color=BLUE, ha="center", va="center",
        fontfamily=FONT)

# ── Key metrics row ──
metrics = [
    ("100%", "Task Success", "KG Smart (50 tasks)", GREEN_600),
    ("97.8%", "Production SR", "1,369 real runs", BLUE),
    ("1.00", "MPF Score", "Novel materials", TEAL),
    ("2.9\u00d7", "KG Advantage", "vs. KG-free baseline", AMBER),
]

mx_start = 66
mx_w = 8
for i, (value, title, sub, color) in enumerate(metrics):
    mx = mx_start + i * mx_w + i * 0.3
    my = 14

    pill(mx, my, mx_w, 12, WHITE, SLATE_200, lw=1.0, zorder=2)

    ax.text(mx + mx_w/2, 23.5, value, fontsize=16, fontweight="bold",
            color=color, ha="center", va="center", fontfamily=FONT)
    ax.text(mx + mx_w/2, 20.5, title, fontsize=7, fontweight="bold",
            color=SLATE_700, ha="center", va="center", fontfamily=FONT)
    ax.text(mx + mx_w/2, 18.5, sub, fontsize=5.5,
            color=SLATE_500, ha="center", va="center", fontfamily=FONT)

    # Decorative bar at top of card
    pill(mx + 0.3, my + 11.2, mx_w - 0.6, 0.5, color, color, lw=0,
         zorder=3)

# ── KG ablation mini-bars ──
pill(65, 1, 33.5, 12, SLATE_50, SLATE_200, lw=1.0, zorder=1)

ax.text(81.75, 12, "Three-Way KG Ablation", fontsize=8.5,
        fontweight="bold", color=SLATE_900, ha="center", va="center",
        fontfamily=FONT)

bar_data = [
    ("Success Rate", 0.72, 1.00, 1.00),
    ("Physics Score", 0.84, 0.853, 0.933),
    ("MPF", 0.76, 0.796, 0.926),
    ("First-Try", 0.56, 0.82, 0.92),
]

bar_x_start = 67
bar_gap = 7.8
bar_max_h = 7.5
bar_y = 2.5
bar_w = 1.6

for i, (label, v_on, v_off, v_smart) in enumerate(bar_data):
    bx = bar_x_start + i * bar_gap

    # KG On
    h = v_on * bar_max_h
    pill(bx, bar_y, bar_w, h, RED_600, RED_600, lw=0, alpha=0.6, zorder=3)

    # KG Off
    h = v_off * bar_max_h
    pill(bx + bar_w + 0.3, bar_y, bar_w, h, SLATE_300, SLATE_300, lw=0,
         alpha=0.8, zorder=3)

    # KG Smart
    h = v_smart * bar_max_h
    pill(bx + 2*(bar_w + 0.3), bar_y, bar_w, h, TEAL, TEAL, lw=0,
         alpha=0.85, zorder=3)

    ax.text(bx + 1.5*(bar_w + 0.3) - 0.15, 1.8, label, fontsize=5,
            color=SLATE_500, ha="center", va="top", fontfamily=FONT)

# Legend (bottom right)
legend_items = [("KG On", RED_600, 0.6), ("KG Off", SLATE_300, 0.8),
                ("KG Smart", TEAL, 0.85)]
for i, (label, color, alpha) in enumerate(legend_items):
    lx = 90 + i * 3.2
    pill(lx, 10.2, 1.0, 0.7, color, color, lw=0, alpha=alpha, zorder=4)
    ax.text(lx + 1.3, 10.55, label, fontsize=4.5, color=SLATE_700,
            ha="left", va="center", fontfamily=FONT)

# ═══════════════════════════════════════════════════════════════════════════
#  SAVE
# ═══════════════════════════════════════════════════════════════════════════
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

out = "/home/ife12524/matpro_files/sadhi/Projects_SIM/pde-agents/paper"
fig.savefig(f"{out}/graphical_abstract.png", dpi=DPI, bbox_inches="tight",
            facecolor=WHITE, pad_inches=0.1)
fig.savefig(f"{out}/graphical_abstract.tiff", dpi=DPI, bbox_inches="tight",
            facecolor=WHITE, pad_inches=0.1,
            pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(f"{out}/graphical_abstract.pdf", dpi=DPI, bbox_inches="tight",
            facecolor=WHITE, pad_inches=0.1)
plt.close()

print("Graphical abstract saved (PNG, TIFF, PDF).")
print(f"  Dimensions: {W*DPI:.0f} x {H*DPI:.0f} px at {DPI} DPI")
