# PDE-Agents — Research Paper

**Title:** PDE-Agents: An LLM-Orchestrated Multi-Agent Framework for Automated Finite Element Simulations with Knowledge Graph-Augmented Reasoning

**Target journal:** Engineering with Computers (Springer) — IF ~8.7

---

## File structure

```
main.tex          ← main document (compile this)
references.bib    ← BibTeX references (29 entries)
figs/             ← TikZ/PGFPlots figures (no external image files needed)
  architecture.tikz     — system architecture diagram
  kg_schema.tikz        — Neo4j knowledge graph schema
  convergence.tikz      — V&V spatial convergence plot
  ablation_bar.tikz     — 3-way KG ablation bar chart
tables/           ← LaTeX table fragments (\input{} in main.tex)
  vv_convergence.tex    — V&V convergence results
  ablation.tex          — 3-way KG ablation results
  agent_metrics.tex     — production agent quality metrics
  vv_detail.tex         — detailed V&V case results
```

## Compiling

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or from the project root:

```bash
make paper-pdf
```

## Sync with Overleaf

This folder is kept in sync with Overleaf via a dedicated GitHub repository
and `git subtree`. From the project root:

```bash
make paper-push   # push local edits → GitHub → Overleaf
make paper-pull   # pull Overleaf edits ← GitHub → local
```

See the project `Makefile` for details.
