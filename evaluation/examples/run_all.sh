#!/usr/bin/env bash
# Run all four example simulations and produce the composite figure.
#
# Usage (from repo root):
#   docker exec pde-fenics bash /workspace/evaluation/examples/run_all.sh
#
# Or via Makefile:
#   make eval-examples
#
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Case A: Steady Dirichlet ==="
python3 "$DIR/case_a_steady_dirichlet.py"

echo "=== Case B: Mixed BCs ==="
python3 "$DIR/case_b_mixed_bcs.py"

echo "=== Case C: Transient L-shape ==="
python3 "$DIR/case_c_lshape_transient.py"

echo "=== Case D: 3D Conduction ==="
python3 "$DIR/case_d_3d_conduction.py"

echo "=== Case E: Plate with Hole ==="
python3 "$DIR/case_e_plate_with_hole.py"

echo "=== Case F: Gaussian Heat Source ==="
python3 "$DIR/case_f_gaussian_source.py"

echo "=== Composite Figure ==="
python3 "$DIR/plot_composite.py"

echo ""
echo "All done. Outputs:"
ls -lh "$DIR/output/"
