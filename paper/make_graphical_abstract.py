#!/usr/bin/env python3
"""Graphical abstract for PDE-Agents CMAME paper.

Elsevier spec: min 1328x531 px, 500:200 ratio, 300 DPI, white bg.
Output: 3984 x 1593 px (3x Elsevier minimum).

Design: clean left-to-right 3-panel flow, 3-colour palette, Roboto font,
generous whitespace, no overlapping elements.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

# ── Palette: 3 hues + neutrals ──────────────────────────────────────────────
BLUE      = "#2563EB"
BLUE_L    = "#DBEAFE"
BLUE_XL   = "#EFF6FF"
TEAL      = "#0D9488"
TEAL_L    = "#CCFBF1"
TEAL_XL   = "#F0FDFA"
AMBER     = "#D97706"
AMBER_L   = "#FEF3C7"
S9        = "#0F172A"   # slate-900 (darkest text)
S7        = "#334155"   # slate-700
S5        = "#64748B"   # slate-500
S3        = "#CBD5E1"   # slate-300
S2        = "#E2E8F0"   # slate-200
S1        = "#F1F5F9"   # slate-100
S0        = "#F8FAFC"   # slate-50
W         = "#FFFFFF"
RED       = "#DC2626"
GREEN     = "#16A34A"

DPI  = 300
FW   = 13.28   # figure width  (inches)
FH   = 5.31    # figure height (inches)
FONT = "Roboto"
PAD  = 0.25    # universal box rounding — small enough to avoid overlap

fig, ax = plt.subplots(figsize=(FW, FH), dpi=DPI)
ax.set_xlim(0, 100)
ax.set_ylim(0, 40)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor(W)
ax.set_facecolor(W)


def box(x, y, w, h, fc, ec=None, lw=1.2, zorder=2, alpha=1.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={PAD}",
                        facecolor=fc, edgecolor=ec or fc,
                        linewidth=lw, alpha=alpha, zorder=zorder)
    ax.add_patch(p)
    return p


def arr(x1, y1, x2, y2, c=S3, lw=2.0, zo=3):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                  color=c, linewidth=lw, mutation_scale=16,
                                  zorder=zo))


# ═════════════════════════════════════════════════════════════════════════════
#  PANEL 1 — NATURAL LANGUAGE INPUT   (x: 1–17)
# ═════════════════════════════════════════════════════════════════════════════
box(1, 1.5, 15.5, 37, S0, S2, lw=1.0, zorder=1)

ax.text(8.75, 36.5, "Natural Language", fontsize=10.5, fontweight="bold",
        color=S9, ha="center", va="center", fontfamily=FONT)
ax.text(8.75, 34.5, "Input", fontsize=10.5, fontweight="bold",
        color=S9, ha="center", va="center", fontfamily=FONT)

# Chat bubble
box(3, 24, 11.5, 8, W, BLUE_L, lw=1.5, zorder=3)
ax.text(8.75, 29.5, '"Simulate heat transfer', fontsize=7.5,
        color=S7, ha="center", va="center", fontfamily=FONT, fontstyle="italic")
ax.text(8.75, 27.5, 'in a steel plate', fontsize=7.5,
        color=S7, ha="center", va="center", fontfamily=FONT, fontstyle="italic")
ax.text(8.75, 25.5, 'at 500 K"', fontsize=7.5,
        color=S7, ha="center", va="center", fontfamily=FONT, fontstyle="italic")

# User icon (simple geometric)
ax.add_patch(Circle((8.75, 17), 3, facecolor=BLUE_L, edgecolor=BLUE,
                     linewidth=1.2, zorder=3))
ax.add_patch(Circle((8.75, 18.2), 1.0, facecolor=BLUE, zorder=4))
ax.add_patch(FancyBboxPatch((7, 14.5), 3.5, 2.2, boxstyle="round,pad=0.3",
                             facecolor=BLUE, edgecolor="none", zorder=4))

ax.text(8.75, 10, "Domain Expert", fontsize=7, color=S5,
        ha="center", va="center", fontfamily=FONT)
ax.text(8.75, 8, "or Researcher", fontsize=7, color=S5,
        ha="center", va="center", fontfamily=FONT)

# ═════════════════════════════════════════════════════════════════════════════
#  ARROW 1
# ═════════════════════════════════════════════════════════════════════════════
arr(17, 20, 19.5, 20, c=BLUE, lw=2.5)

# ═════════════════════════════════════════════════════════════════════════════
#  PANEL 2 — MULTI-AGENT SYSTEM   (x: 20–60)
# ═════════════════════════════════════════════════════════════════════════════
box(20, 1.5, 39, 37, W, BLUE, lw=2.0, zorder=1)

ax.text(39.5, 37, "PDE-Agents", fontsize=12, fontweight="bold",
        color=BLUE, ha="center", va="center", fontfamily=FONT)
ax.text(39.5, 35.2, "Multi-Agent Orchestrator", fontsize=8.5,
        color=S5, ha="center", va="center", fontfamily=FONT)

# Supervisor bar
box(22.5, 31, 34, 2.5, BLUE, BLUE, lw=0, alpha=0.08, zorder=3)
box(22.5, 31, 34, 2.5, "none", BLUE, lw=1.2, zorder=4)
ax.text(39.5, 32.25, "LangGraph Supervisor Router", fontsize=7.5,
        fontweight="bold", color=BLUE, ha="center", va="center",
        fontfamily=FONT)

# Three agent cards — equal spacing, no overlap
CW, CH = 10, 8.5          # card width, height
CY = 20.5                  # card y
CXS = [23, 34.5, 46]      # card x positions (gap = 1.5 between cards)

agents = [
    ("Simulation", "Agent", BLUE, BLUE_XL, "SIM"),
    ("Analytics", "Agent", TEAL, TEAL_XL, "ANA"),
    ("Database", "Agent", AMBER, AMBER_L, "DB"),
]

for cx, (n1, n2, col, bg, abbr) in zip(CXS, agents):
    box(cx, CY, CW, CH, bg, col, lw=1.3, zorder=3)
    mid = cx + CW / 2

    # Icon circle
    ax.add_patch(Circle((mid, CY + 6.5), 1.2, facecolor=col, zorder=4))
    ax.text(mid, CY + 6.5, abbr, fontsize=5.5, fontweight="bold", color=W,
            ha="center", va="center", fontfamily=FONT, zorder=5)

    ax.text(mid, CY + 3.5, n1, fontsize=7.5, fontweight="bold", color=col,
            ha="center", va="center", fontfamily=FONT)
    ax.text(mid, CY + 1.8, n2, fontsize=7.5, fontweight="bold", color=col,
            ha="center", va="center", fontfamily=FONT)

    # Connector line from supervisor to card
    arr(mid, 31, mid, CY + CH + 0.5, c=S3, lw=1.0)

# LLM bar
box(22.5, 16, 34, 2.8, S1, S2, lw=0.8, zorder=3)
ax.text(39.5, 17.4,
        "Local LLMs:  Qwen3-Coder-Next  |  Llama 4 Scout  |"
        "  2\u00d7 RTX PRO 6000",
        fontsize=5.5, color=S5, ha="center", va="center", fontfamily=FONT)

# Knowledge Graph
box(22.5, 2.5, 34, 12, TEAL_XL, TEAL, lw=1.3, zorder=2)
ax.text(39.5, 13.2, "GraphRAG Knowledge Base", fontsize=8,
        fontweight="bold", color=TEAL, ha="center", va="center",
        fontfamily=FONT)

# Graph nodes — well spaced within KG area (y: 4–11)
gnodes = [
    (29, 8.5, "Run",      1.4),
    (35, 10,  "Material", 1.1),
    (35, 6,   "Issue",    1.1),
    (44, 10,  "BC",       0.9),
    (44, 6,   "Ref",      0.9),
    (50, 8.5, "Domain",   0.9),
]
gedges = [(0,1),(0,2),(0,3),(0,4),(0,5),(1,3),(2,4),(3,5)]
for a, b in gedges:
    ax.plot([gnodes[a][0], gnodes[b][0]], [gnodes[a][1], gnodes[b][1]],
            color=TEAL, alpha=0.2, lw=1.0, zorder=3)
for nx, ny, lbl, r in gnodes:
    ax.add_patch(Circle((nx, ny), r, facecolor=TEAL, edgecolor=W,
                         linewidth=1.0, alpha=0.75, zorder=4))
    ax.text(nx, ny, lbl, fontsize=4.5 if r >= 1.0 else 4, fontweight="bold",
            color=W, ha="center", va="center", fontfamily=FONT, zorder=5)

ax.text(39.5, 3.5,
        "Neo4j  \u00b7  768-dim embeddings  \u00b7  HNSW vector search"
        "  \u00b7  warm-start injection",
        fontsize=5, color=TEAL, ha="center", va="center",
        fontfamily=FONT, fontstyle="italic")

# ═════════════════════════════════════════════════════════════════════════════
#  ARROW 2
# ═════════════════════════════════════════════════════════════════════════════
arr(59.5, 20, 62, 20, c=BLUE, lw=2.5)

# ═════════════════════════════════════════════════════════════════════════════
#  PANEL 3 — OUTPUT & RESULTS   (x: 62.5–99)
# ═════════════════════════════════════════════════════════════════════════════

# ── Top section: FEM equation ──
box(62.5, 28, 36, 10.5, BLUE_XL, BLUE_L, lw=1.0, zorder=1)

ax.text(80.5, 37, "FEM Simulation Output", fontsize=10, fontweight="bold",
        color=S9, ha="center", va="center", fontfamily=FONT)

ax.text(80.5, 34,
        r"$\rho\, c_p\, \dfrac{\partial u}{\partial t}"
        r" - \nabla \!\cdot\! (k\,\nabla u) = f$",
        fontsize=11, color=S9, ha="center", va="center")

ax.text(80.5, 31.2,
        "DOLFINx / FEniCSx  \u00b7  PETSc KSP  \u00b7  "
        "\u03b8-scheme  \u00b7  Gmsh",
        fontsize=5.5, color=S5, ha="center", va="center", fontfamily=FONT)

# O(h^2) badge — right-aligned, no overlap
box(89, 29, 8.5, 2, W, BLUE, lw=1.0, zorder=3)
ax.text(93.25, 30, r"$\mathcal{O}(h^2)$ verified", fontsize=6.5,
        fontweight="bold", color=BLUE, ha="center", va="center",
        fontfamily=FONT)

# ── Middle section: 4 metric cards ──
MW  = 8.2      # card width
MH  = 9        # card height
MG  = 0.6      # gap between cards
MY  = 17.5     # card y
MX0 = 63       # first card x

metrics = [
    ("100%",         "Task Success",  "KG Smart · 50 tasks", GREEN),
    ("97.8%",        "Production SR", "1,369 real runs",     BLUE),
    ("1.00",         "MPF Score",     "Novel materials",     TEAL),
    ("2.9\u00d7",    "KG Advantage",  "vs. KG-free baseline",AMBER),
]

for i, (val, title, sub, col) in enumerate(metrics):
    mx = MX0 + i * (MW + MG)
    box(mx, MY, MW, MH, W, S2, lw=0.8, zorder=2)

    # Colour accent bar at top
    box(mx + 0.4, MY + MH - 0.8, MW - 0.8, 0.4, col, col, lw=0, zorder=3)

    ax.text(mx + MW/2, MY + 6.2, val, fontsize=15, fontweight="bold",
            color=col, ha="center", va="center", fontfamily=FONT)
    ax.text(mx + MW/2, MY + 3.5, title, fontsize=6.5, fontweight="bold",
            color=S7, ha="center", va="center", fontfamily=FONT)
    ax.text(mx + MW/2, MY + 1.8, sub, fontsize=5,
            color=S5, ha="center", va="center", fontfamily=FONT)

# ── Bottom section: ablation bars ──
box(62.5, 1.5, 36, 14.5, S0, S2, lw=0.8, zorder=1)

ax.text(80.5, 15, "Three-Way KG Ablation", fontsize=8,
        fontweight="bold", color=S9, ha="center", va="center",
        fontfamily=FONT)

bar_cats = [
    ("Success\nRate",  0.72, 1.00, 1.00),
    ("Physics\nScore", 0.84, 0.85, 0.93),
    ("MPF",            0.76, 0.80, 0.93),
    ("First-Try\nSuccess", 0.56, 0.82, 0.92),
]

BW   = 1.4     # bar width
BG   = 0.2     # gap within group
BY   = 3.5     # bar bottom y
BMH  = 9.0     # max bar height
BGRP = 8.5     # gap between group centres

for i, (lbl, v1, v2, v3) in enumerate(bar_cats):
    gx = 66 + i * BGRP   # group centre-ish

    for j, (v, c, a) in enumerate([(v1, RED, 0.55),
                                    (v2, S3, 0.85),
                                    (v3, TEAL, 0.85)]):
        bx = gx + j * (BW + BG)
        bh = v * BMH
        box(bx, BY, BW, bh, c, c, lw=0, alpha=a, zorder=3)

    ax.text(gx + 1.5*(BW+BG) - BG/2, 2.5, lbl, fontsize=4.5,
            color=S5, ha="center", va="top", fontfamily=FONT,
            linespacing=1.1)

# Legend — placed in the right area of the ablation box
leg = [("KG On", RED, 0.55), ("KG Off", S3, 0.85), ("KG Smart", TEAL, 0.85)]
for i, (lbl, c, a) in enumerate(leg):
    lx = 88 + i * 3.5
    box(lx, 13.2, 1.0, 0.6, c, c, lw=0, alpha=a, zorder=4)
    ax.text(lx + 1.3, 13.5, lbl, fontsize=4.2, color=S7,
            ha="left", va="center", fontfamily=FONT)

# ═════════════════════════════════════════════════════════════════════════════
#  SAVE
# ═════════════════════════════════════════════════════════════════════════════
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

out = "/home/ife12524/matpro_files/sadhi/Projects_SIM/pde-agents/paper"
for fmt, kw in [("png",  {}),
                ("tiff", {"pil_kwargs": {"compression": "tiff_lzw"}}),
                ("pdf",  {})]:
    fig.savefig(f"{out}/graphical_abstract.{fmt}", dpi=DPI,
                bbox_inches="tight", facecolor=W, pad_inches=0.1, **kw)
plt.close()

print("Graphical abstract saved (PNG, TIFF, PDF).")
print(f"  Dimensions: {FW*DPI:.0f} x {FH*DPI:.0f} px at {DPI} DPI")
