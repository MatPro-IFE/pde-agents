#!/usr/bin/env python3
"""Generate the graphical abstract for the PDE-Agents CMAME paper.

Elsevier spec: 531 × 1328 px minimum, landscape, no text smaller than 8pt.
We render at 300 DPI → 4.43 × 11.07 inches, output as TIFF + EPS + PDF.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patheffects as pe
import numpy as np

# ── Colour palette (matches paper) ──────────────────────────────────────────
BG          = "#FFFFFF"
DARK        = "#1A1A2E"
BLUE        = "#3478BE"      # nodeblue
BLUE_LIGHT  = "#D6E6F5"
GREEN       = "#388E3C"
GREEN_LIGHT = "#DFF0D8"
ORANGE      = "#E65100"
ORANGE_LIGHT= "#FFF3E0"
RED         = "#B71C1C"       # nodered
PURPLE      = "#5E35B1"
PURPLE_LIGHT= "#EDE7F6"
GRAY        = "#616161"
GRAY_LIGHT  = "#F5F5F5"
GOLD        = "#F9A825"
TEAL        = "#00897B"
TEAL_LIGHT  = "#E0F2F1"

DPI = 300
fig, ax = plt.subplots(figsize=(11.07, 4.43), dpi=DPI)
ax.set_xlim(0, 100)
ax.set_ylim(0, 40)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)


def rounded_box(x, y, w, h, color, border_color=None, alpha=1.0, lw=1.5,
                radius=0.6, zorder=2):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad={radius}",
                         facecolor=color, edgecolor=border_color or color,
                         linewidth=lw, alpha=alpha, zorder=zorder,
                         mutation_scale=1)
    ax.add_patch(box)
    return box


def arrow(x1, y1, x2, y2, color=GRAY, lw=1.8, style="-|>", zorder=3):
    ar = FancyArrowPatch((x1, y1), (x2, y2),
                         arrowstyle=style, color=color,
                         linewidth=lw, mutation_scale=14, zorder=zorder,
                         connectionstyle="arc3,rad=0")
    ax.add_patch(ar)


def curved_arrow(x1, y1, x2, y2, color=GRAY, lw=1.5, rad=0.25, zorder=3):
    ar = FancyArrowPatch((x1, y1), (x2, y2),
                         arrowstyle="-|>", color=color,
                         linewidth=lw, mutation_scale=12, zorder=zorder,
                         connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(ar)


# ═══════════════════════════════════════════════════════════════════════════
#  TITLE BAR (top)
# ═══════════════════════════════════════════════════════════════════════════
rounded_box(0.5, 36.0, 99, 3.5, DARK, lw=0, radius=0.4, zorder=1)
ax.text(50, 37.75, "PDE-Agents",
        fontsize=16, fontweight="bold", color="white",
        ha="center", va="center", fontfamily="sans-serif", zorder=5)
ax.text(50, 36.7,
        "LLM-Orchestrated Multi-Agent Framework for Automated FEM Simulations "
        "with Knowledge Graph-Augmented Reasoning",
        fontsize=7, color="#B0BEC5", ha="center", va="center",
        fontfamily="sans-serif", zorder=5)

# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 1: USER INPUT (left)
# ═══════════════════════════════════════════════════════════════════════════
rounded_box(1, 25.5, 12, 9, GRAY_LIGHT, GRAY, alpha=0.5, lw=1.2)
ax.text(7, 33.5, "Natural Language", fontsize=7.5, fontweight="bold",
        color=DARK, ha="center", va="center")
ax.text(7, 32.5, "Interface", fontsize=7.5, fontweight="bold",
        color=DARK, ha="center", va="center")

# Chat bubble
rounded_box(2.2, 28.8, 9.6, 2.8, "white", BLUE, lw=1, radius=0.4)
ax.text(7, 30.8, '"Run a 2D heat simulation', fontsize=5.5, color=DARK,
        ha="center", va="center", fontstyle="italic")
ax.text(7, 29.8, 'on a steel plate at 500 K"', fontsize=5.5, color=DARK,
        ha="center", va="center", fontstyle="italic")

# User icon
circle = Circle((7, 27), 1.0, facecolor=BLUE_LIGHT, edgecolor=BLUE,
                linewidth=1.2, zorder=3)
ax.add_patch(circle)
ax.text(7, 27, "USER", fontsize=6, ha="center", va="center", zorder=4,
        fontweight="bold", color=BLUE)

# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 2: MULTI-AGENT ORCHESTRATOR (center)
# ═══════════════════════════════════════════════════════════════════════════
# Outer orchestrator box
rounded_box(15, 14, 38, 21.5, "white", BLUE, lw=2.0, radius=0.8)
ax.text(34, 34.2, "LangGraph Multi-Agent Orchestrator",
        fontsize=8, fontweight="bold", color=BLUE, ha="center", va="center")

# Supervisor node
rounded_box(27, 31, 14, 2.5, BLUE, BLUE, alpha=0.15, lw=1.5)
ax.text(34, 32.25, "SUPERVISOR ROUTER", fontsize=7, fontweight="bold",
        color=BLUE, ha="center", va="center")

# Agent boxes
agent_data = [
    (16, 22.5, "Simulation\nAgent", BLUE, BLUE_LIGHT, "SIM",
     "FEM setup\n& execution"),
    (28, 22.5, "Analytics\nAgent", GREEN, GREEN_LIGHT, "ANA",
     "Result analysis\n& optimization"),
    (40, 22.5, "Database\nAgent", ORANGE, ORANGE_LIGHT, "DB",
     "Storage\n& retrieval"),
]

for x, y, label, color, bg, icon, desc in agent_data:
    rounded_box(x, y, 10.5, 7.5, bg, color, lw=1.5, radius=0.5)
    icon_circle = Circle((x + 5.25, 29), 1.2, facecolor=color,
                          edgecolor="white", linewidth=1.2, alpha=0.9,
                          zorder=4)
    ax.add_patch(icon_circle)
    ax.text(x + 5.25, 29, icon, fontsize=5.5, ha="center", va="center",
            zorder=5, fontweight="bold", color="white")
    ax.text(x + 5.25, 26.8, label, fontsize=6.5, fontweight="bold",
            color=color, ha="center", va="center", linespacing=1.3)
    ax.text(x + 5.25, 24.2, desc, fontsize=5, color=GRAY,
            ha="center", va="center", linespacing=1.2)

# Arrows: supervisor → agents
for x_off in [21.25, 33.25, 45.25]:
    arrow(x_off, 31, x_off, 30.2, color=BLUE, lw=1.2)

# Feedback loop arrows: agents → supervisor
curved_arrow(21.25, 30.2, 27, 32.25, color=BLUE, lw=0.8, rad=-0.3)
curved_arrow(45.25, 30.2, 41, 32.25, color=BLUE, lw=0.8, rad=0.3)

# LLM badge
rounded_box(17, 15, 33, 2.8, PURPLE_LIGHT, PURPLE, lw=1, radius=0.3)
ax.text(33.5, 16.4, "LOCAL OPEN-SOURCE LLMs   |   Qwen3-Coder-Next (80B)   |   "
        "Llama 4 Scout   |   2x RTX PRO 6000 (196 GB VRAM)",
        fontsize=5, color=PURPLE, ha="center", va="center",
        fontweight="bold")

# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 3: KNOWLEDGE GRAPH (bottom center)
# ═══════════════════════════════════════════════════════════════════════════
rounded_box(15, 1.5, 21, 11.5, TEAL_LIGHT, TEAL, lw=1.8, radius=0.6)
ax.text(25.5, 12, "GraphRAG Knowledge Base", fontsize=7.5,
        fontweight="bold", color=TEAL, ha="center", va="center")

# Mini graph nodes
graph_nodes = [
    (19, 8.5, "Material", BLUE, 1.2),
    (25.5, 9.5, "Run", TEAL, 1.4),
    (32, 8.5, "BC Config", GREEN, 1.2),
    (19, 5, "Known\nIssue", RED, 1.2),
    (25.5, 4, "Reference", PURPLE, 1.2),
    (32, 5, "Thermal\nClass", ORANGE, 1.2),
]
for nx_, ny_, label, color, r in graph_nodes:
    circle = Circle((nx_, ny_), r, facecolor=color, edgecolor="white",
                     linewidth=1.5, alpha=0.85, zorder=4)
    ax.add_patch(circle)
    ax.text(nx_, ny_, label, fontsize=3.8, color="white", ha="center",
            va="center", fontweight="bold", zorder=5, linespacing=1.1)

# Graph edges
graph_edges = [
    (19, 8.5, 25.5, 9.5), (25.5, 9.5, 32, 8.5),
    (19, 5, 25.5, 4), (25.5, 4, 32, 5),
    (19, 8.5, 19, 5), (32, 8.5, 32, 5),
    (25.5, 9.5, 25.5, 4), (25.5, 9.5, 19, 5), (25.5, 9.5, 32, 5),
]
for x1, y1, x2, y2 in graph_edges:
    ax.plot([x1, x2], [y1, y2], color=TEAL, alpha=0.3, lw=1.0, zorder=3)

# KG details
ax.text(25.5, 2.3, "Neo4j  ·  768-dim Embeddings  ·  HNSW Vector Search",
        fontsize=4.5, color=TEAL, ha="center", va="center",
        fontstyle="italic")

# Arrow: orchestrator ↔ KG
arrow(25.5, 14, 25.5, 13.2, color=TEAL, lw=1.5)
ax.text(27.5, 13.6, "warm-start\ninjection", fontsize=4.5, color=TEAL,
        ha="left", va="center", fontstyle="italic", linespacing=1.1)

# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 4: FEM SOLVER (bottom right)
# ═══════════════════════════════════════════════════════════════════════════
rounded_box(38, 1.5, 16, 11.5, BLUE_LIGHT, BLUE, lw=1.8, radius=0.6)
ax.text(46, 12, "FEM Solver", fontsize=7.5, fontweight="bold",
        color=BLUE, ha="center", va="center")

# Heat equation
ax.text(46, 9.8, r"$\rho\, c_p\, \frac{\partial u}{\partial t}"
        r" - \nabla \cdot (k\,\nabla u) = f$",
        fontsize=8, color=DARK, ha="center", va="center")

# DOLFINx badge
rounded_box(40.5, 6.5, 11, 2.2, "white", BLUE, lw=1, radius=0.3)
ax.text(46, 7.6, "DOLFINx / FEniCSx", fontsize=6, fontweight="bold",
        color=BLUE, ha="center", va="center")

# Features
features = ["PETSc KSP solvers", "θ-scheme time integration",
            "Gmsh geometries (9 types)", "Dirichlet / Neumann / Robin BCs"]
for i, f in enumerate(features):
    ax.text(46, 5.5 - i * 1.0, f"· {f}", fontsize=4.5, color=GRAY,
            ha="center", va="center")

# Arrow: orchestrator → FEM
arrow(46, 14, 46, 13.2, color=BLUE, lw=1.5)

# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 5: KEY RESULTS (right panel)
# ═══════════════════════════════════════════════════════════════════════════
rounded_box(56, 1.5, 43, 33.5, GRAY_LIGHT, GRAY, alpha=0.3, lw=1.2,
            radius=0.8)
ax.text(77.5, 34, "Key Results", fontsize=9, fontweight="bold",
        color=DARK, ha="center", va="center")

# ── Result card 1: V&V ──
rounded_box(57.5, 28, 13, 5, "white", BLUE, lw=1.2, radius=0.4)
ax.text(64, 32, "V&V Convergence", fontsize=6, fontweight="bold",
        color=BLUE, ha="center", va="center")
ax.text(64, 30.5, r"$\mathcal{O}(h^2)$", fontsize=14, fontweight="bold",
        color=BLUE, ha="center", va="center")
ax.text(64, 28.8, "All benchmarks pass", fontsize=5, color=GRAY,
        ha="center", va="center")

# ── Result card 2: Success Rate ──
rounded_box(72, 28, 13, 5, "white", GREEN, lw=1.2, radius=0.4)
ax.text(78.5, 32, "Task Success", fontsize=6, fontweight="bold",
        color=GREEN, ha="center", va="center")
ax.text(78.5, 30.5, "100%", fontsize=14, fontweight="bold",
        color=GREEN, ha="center", va="center")
ax.text(78.5, 28.8, "KG Smart (50 tasks)", fontsize=5, color=GRAY,
        ha="center", va="center")

# ── Result card 3: Production ──
rounded_box(86.5, 28, 11.5, 5, "white", ORANGE, lw=1.2, radius=0.4)
ax.text(92.25, 32, "Production Runs", fontsize=6, fontweight="bold",
        color=ORANGE, ha="center", va="center")
ax.text(92.25, 30.5, "1,369", fontsize=14, fontweight="bold",
        color=ORANGE, ha="center", va="center")
ax.text(92.25, 28.8, "97.8% success rate", fontsize=5, color=GRAY,
        ha="center", va="center")

# ── Ablation comparison bar chart ──
rounded_box(57.5, 14.5, 40.5, 12.5, "white", GRAY, lw=1, radius=0.4)
ax.text(77.75, 26, "Three-Way KG Ablation Study", fontsize=7,
        fontweight="bold", color=DARK, ha="center", va="center")

# Bar chart data
categories = ["Success\nRate", "Physics\nScore", "MPF", "First-Try\nSuccess"]
kg_on   = [0.72, 0.84, 0.76, 0.56]
kg_off  = [1.00, 0.853, 0.796, 0.82]
kg_smart = [1.00, 0.933, 0.926, 0.92]

bar_w = 1.8
gap = 9.2
x_start = 61

for i, (cat, v1, v2, v3) in enumerate(zip(categories, kg_on, kg_off, kg_smart)):
    bx = x_start + i * gap
    max_h = 8.5
    y_base = 15.5

    # KG On
    h1 = v1 * max_h
    rounded_box(bx - 3.0, y_base, bar_w, h1, RED, RED, alpha=0.7,
                lw=0, radius=0.2, zorder=3)
    ax.text(bx - 2.1, y_base + h1 + 0.3, f"{v1:.0%}", fontsize=4.5,
            color=RED, ha="center", va="bottom", fontweight="bold", zorder=4)

    # KG Off
    h2 = v2 * max_h
    rounded_box(bx - 0.9, y_base, bar_w, h2, GRAY, GRAY, alpha=0.5,
                lw=0, radius=0.2, zorder=3)
    ax.text(bx + 0.0, y_base + h2 + 0.3, f"{v2:.0%}", fontsize=4.5,
            color=GRAY, ha="center", va="bottom", fontweight="bold", zorder=4)

    # KG Smart
    h3 = v3 * max_h
    rounded_box(bx + 1.2, y_base, bar_w, h3, TEAL, TEAL, alpha=0.85,
                lw=0, radius=0.2, zorder=3)
    ax.text(bx + 2.1, y_base + h3 + 0.3, f"{v3:.0%}", fontsize=4.5,
            color=TEAL, ha="center", va="bottom", fontweight="bold", zorder=4)

    ax.text(bx, 15.0, cat, fontsize=4.5, color=GRAY, ha="center",
            va="top", linespacing=1.1)

# Legend
for i, (label, color) in enumerate([("KG On", RED), ("KG Off", GRAY),
                                     ("KG Smart", TEAL)]):
    lx = 86 + i * 4.2
    rounded_box(lx, 24.5, 1.2, 0.8, color, color, alpha=0.7 if label != "KG Smart" else 0.85,
                lw=0, radius=0.15)
    ax.text(lx + 1.6, 24.9, label, fontsize=4.5, color=color,
            ha="left", va="center", fontweight="bold")

# ── Novel material result ──
rounded_box(57.5, 2, 40.5, 11.5, "white", PURPLE, lw=1.2, radius=0.4)
ax.text(77.75, 12.5, "Novel Material Experiment — KG Value Proof",
        fontsize=6.5, fontweight="bold", color=PURPLE, ha="center",
        va="center")

# MPF comparison
ax.text(66, 10.5, "Material Property Fidelity (MPF)", fontsize=5.5,
        color=DARK, ha="center", va="center", fontweight="bold")

# KG Off bar
rounded_box(60, 7, 12, 2.5, RED, RED, alpha=0.15, lw=1, radius=0.3)
ax.text(61, 8.25, "KG Off", fontsize=5.5, fontweight="bold", color=RED,
        ha="left", va="center")
ax.text(71, 8.25, "0.34", fontsize=9, fontweight="bold", color=RED,
        ha="right", va="center")

# KG Smart bar
rounded_box(60, 3.5, 12, 2.5, TEAL, TEAL, alpha=0.15, lw=1, radius=0.3)
ax.text(61, 4.75, "KG Smart", fontsize=5.5, fontweight="bold", color=TEAL,
        ha="left", va="center")
ax.text(71, 4.75, "1.00", fontsize=9, fontweight="bold", color=TEAL,
        ha="right", va="center")

# Arrow showing improvement
ax.annotate("", xy=(73.5, 4.75), xytext=(73.5, 8.25),
            arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=2.5,
                            mutation_scale=16),
            zorder=5)
ax.text(74.5, 6.5, "2.9×", fontsize=10, fontweight="bold", color=GOLD,
        ha="left", va="center", zorder=5)

# Example materials
materials = [
    ("Pyrathane", "k=312 W/(m·K)", "KG Off used k=0.15\n→ 2,080× error"),
    ("Cryonite", "k=0.42 W/(m·K)", "Extreme insulator\ncorrectly identified"),
    ("Novidium", "k=73 W/(m·K)", "Moderate conductor\nexact retrieval"),
]
for i, (name, prop, note) in enumerate(materials):
    mx = 80 + i * 6
    rounded_box(mx - 2.5, 3.5, 5.8, 6.5, PURPLE_LIGHT, PURPLE, lw=0.8,
                radius=0.3, alpha=0.5)
    ax.text(mx + 0.4, 9, name, fontsize=5, fontweight="bold",
            color=PURPLE, ha="center", va="center")
    ax.text(mx + 0.4, 7.8, prop, fontsize=4, color=DARK,
            ha="center", va="center")
    ax.text(mx + 0.4, 5.5, note, fontsize=3.5, color=GRAY,
            ha="center", va="center", linespacing=1.2)

# ═══════════════════════════════════════════════════════════════════════════
#  CONNECTING ARROWS (left → center, center → right)
# ═══════════════════════════════════════════════════════════════════════════
arrow(13, 30, 14.8, 30, color=DARK, lw=2.0)
arrow(53.2, 25, 55.8, 25, color=DARK, lw=2.0)

# ═══════════════════════════════════════════════════════════════════════════
#  SAVE
# ═══════════════════════════════════════════════════════════════════════════
plt.tight_layout(pad=0.3)

out_dir = "/home/ife12524/matpro_files/sadhi/Projects_SIM/pde-agents/paper"
fig.savefig(f"{out_dir}/graphical_abstract.pdf", dpi=DPI, bbox_inches="tight",
            facecolor=BG)
fig.savefig(f"{out_dir}/graphical_abstract.tiff", dpi=DPI, bbox_inches="tight",
            facecolor=BG, pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(f"{out_dir}/graphical_abstract.png", dpi=DPI, bbox_inches="tight",
            facecolor=BG)
plt.close()

print("Graphical abstract saved as PDF, TIFF, and PNG.")
