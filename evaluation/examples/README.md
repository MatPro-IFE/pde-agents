# Representative Simulation Examples

Four self-contained FEM simulations that demonstrate the range of problems
PDE-Agents can solve. Each script is independent — no agent orchestration
is needed; they call DOLFINx directly. The scripts serve two purposes:

1. **Paper figure** — generate Figure 4 in the paper (composite 2×2 plot).
2. **Reviewer reproducibility** — re-run any case with a single command.

## Cases

| Script | Domain | Material | BCs | Notes |
|--------|--------|----------|-----|-------|
| `case_a_steady_dirichlet.py` | [0,1]² | Copper (k=385) | Dirichlet left/right | Steady-state, linear gradient |
| `case_b_mixed_bcs.py` | [0,1]² | AISI 1010 steel (k=50) | Dirichlet + Neumann + Robin + insulated | Volumetric heat source Q=5 kW/m³ |
| `case_c_lshape_transient.py` | L-shape | Aluminium (k=205) | Dirichlet + insulated | Transient, Gmsh mesh, 100 time steps |
| `case_d_3d_conduction.py` | [0,1]³ | SS 304 (k=16.3) | Dirichlet (bottom+left) + Robin (top) | 3D, asymmetric BCs |
| `case_e_plate_with_hole.py` | [0,1]² with hole | Ti-6Al-4V (k=6.7) | Dirichlet left/right, insulated hole | Gmsh boolean difference, r=0.15 |
| `case_f_gaussian_source.py` | [0,1]² | Al 6061 (k=205) | Dirichlet all edges, Gaussian Q(x,y) | Localised heat source, Q_peak=5×10⁶ |

## Quick Start

All commands assume you are in the repository root (`pde-agents/`).

### Run everything (recommended)

```bash
make eval-examples
```

### Run inside Docker manually

```bash
docker exec pde-fenics bash /workspace/evaluation/examples/run_all.sh
```

### Run a single case

```bash
docker exec pde-fenics python3 /workspace/evaluation/examples/case_a_steady_dirichlet.py
```

### Re-generate composite figure only (no simulation)

If the `.npz` files already exist in `output/`, you can regenerate the figure
without re-running the simulations:

```bash
docker exec pde-fenics python3 /workspace/evaluation/examples/plot_composite.py
```

Or locally (needs `numpy`, `matplotlib`, `scipy`):

```bash
python3 evaluation/examples/plot_composite.py
```

## Outputs

All outputs are written to `evaluation/examples/output/`:

| File | Description |
|------|-------------|
| `case_a.npz` | Mesh + solution for Case A |
| `case_a.png` | Standalone plot for Case A |
| `case_b.npz` | Mesh + solution for Case B |
| `case_b.png` | Standalone plot for Case B |
| `case_c.npz` | Mesh + 3 time snapshots for Case C |
| `case_c.png` | 3-panel time-series plot for Case C |
| `case_d.npz` | Mesh + solution + pre-evaluated slices for Case D |
| `case_d.png` | 3-panel cross-section plot for Case D |
| `case_e.npz` | Mesh + solution for Case E |
| `case_e.png` | Standalone plot for Case E |
| `case_f.npz` | Mesh + solution for Case F |
| `case_f.png` | Standalone plot for Case F |
| `figure_gallery.png` | 3×2 composite (paper figure, 300 dpi) |
| `sim_examples.png` | Same composite, also copied to `paper/figs/` |

## Dependencies

All scripts run inside the `pde-fenics` Docker container which provides:

- DOLFINx 0.10.0.post2
- Gmsh 4.14.0
- Matplotlib 3.9.3
- NumPy 1.26.4
- SciPy (for `plot_composite.py` — not needed if `.npz` files exist)

## Modifying a Case

Each `.py` file has parameters at the top (material properties, mesh size,
BCs, time stepping). To respond to reviewer feedback:

1. Edit the relevant parameter(s).
2. Re-run the single case: `docker exec pde-fenics python3 /workspace/evaluation/examples/case_X_....py`
3. Regenerate the composite: `docker exec pde-fenics python3 /workspace/evaluation/examples/plot_composite.py`
4. Copy to paper: `cp evaluation/examples/output/sim_examples.png paper/figs/`
5. Recompile: `cd paper && pdflatex main.tex`

Or simply: `make eval-examples && make paper-pdf`
