#!/bin/bash
# Convert all TikZ figures to EPS for CMAME journal submission
cd "$(dirname "$0")"

SRCDIR="$(pwd)/figs"
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

TIKZ_FILES=(architecture kg_visual convergence kg_modes ablation_bar mpf_comparison error_magnification kg_growth)

PREAMBLE='\documentclass[border=2pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{xcolor}
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning, fit,
                 backgrounds, calc, decorations.markings, patterns}
\usepackage{amssymb}
\definecolor{nodeblue}{RGB}{52,120,190}
\definecolor{nodegreen}{RGB}{56,142,60}
\definecolor{nodeorange}{RGB}{230,81,0}
\definecolor{nodegray}{RGB}{97,97,97}
\definecolor{nodered}{RGB}{183,28,28}
\definecolor{lightgray}{RGB}{240,240,240}
\setlength{\columnwidth}{16cm}
\begin{document}'

for name in "${TIKZ_FILES[@]}"; do
  echo "[$(date +%H:%M:%S)] $name: compiling ..."
  cat > "$TMPDIR/${name}.tex" <<LATEX
${PREAMBLE}
\input{${SRCDIR}/${name}.tikz}
\end{document}
LATEX

  pdflatex -interaction=nonstopmode -output-directory="$TMPDIR" "$TMPDIR/${name}.tex" > /dev/null 2>&1
  if [ -f "$TMPDIR/${name}.pdf" ]; then
    echo "[$(date +%H:%M:%S)] $name: converting to EPS ..."
    pdftops -eps "$TMPDIR/${name}.pdf" "$SRCDIR/${name}.eps"
    echo "[$(date +%H:%M:%S)] $name: done ($(du -h "$SRCDIR/${name}.eps" | cut -f1))"
  else
    echo "[$(date +%H:%M:%S)] $name: FAILED (pdflatex error)"
  fi
done

echo ""
echo "=== Summary ==="
ls -lh "$SRCDIR"/*.eps 2>/dev/null || echo "No EPS files found!"
echo "FINISHED"
