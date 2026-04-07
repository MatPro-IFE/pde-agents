"""
Analytical solutions for heat equation benchmark problems.

These closed-form solutions are used to verify the FEniCSx solver by computing
L2 error norms at various mesh resolutions. Each function takes DOF coordinates
and returns the exact temperature values.

Reference: Carslaw & Jaeger, "Conduction of Heat in Solids", Oxford, 1959.
"""

from __future__ import annotations

import numpy as np


# ─── Case 1: 2D Steady-State Linear Profile ──────────────────────────────────
# PDE:   -k ∇²T = 0  on  [0,Lx] × [0,Ly]
# BCs:   T(0,y) = T_L,  T(Lx,y) = T_R,  ∂T/∂n = 0 on top/bottom
# Exact: T(x,y) = T_L + (T_R - T_L) * x / Lx

def steady_linear_2d(x: np.ndarray, T_L: float = 0.0, T_R: float = 1.0,
                     Lx: float = 1.0) -> np.ndarray:
    """Linear temperature profile between two Dirichlet walls."""
    return T_L + (T_R - T_L) * x[0] / Lx


# ─── Case 2: 2D Steady-State Sinusoidal (Laplace) ────────────────────────────
# PDE:   -∇²T = 0  on  [0,1] × [0,1]
# BCs:   T(x,0) = 0,  T(x,1) = sin(πx),  T(0,y) = 0,  T(1,y) = 0
# Exact: T(x,y) = sin(πx) sinh(πy) / sinh(π)

def steady_sinusoidal_2d(x: np.ndarray) -> np.ndarray:
    """Laplace equation with sinusoidal top BC on unit square."""
    return np.sin(np.pi * x[0]) * np.sinh(np.pi * x[1]) / np.sinh(np.pi)


# ─── Case 3: 2D Transient Decay (Fourier Mode) ──────────────────────────────
# PDE:   ρ cp ∂T/∂t - k ∇²T = 0  on  [0,Lx] × [0,Ly]
# BCs:   T = 0 on all boundaries
# IC:    T(x,y,0) = sin(πx/Lx) sin(πy/Ly)
# Exact: T(x,y,t) = sin(πx/Lx) sin(πy/Ly) exp(-α π²(1/Lx² + 1/Ly²) t)
#        where α = k/(ρ cp)

def transient_fourier_2d(x: np.ndarray, t: float,
                         k: float = 1.0, rho: float = 1.0, cp: float = 1.0,
                         Lx: float = 1.0, Ly: float = 1.0) -> np.ndarray:
    """Single Fourier mode decaying under homogeneous Dirichlet BCs."""
    alpha = k / (rho * cp)
    decay = np.exp(-alpha * np.pi**2 * (1.0 / Lx**2 + 1.0 / Ly**2) * t)
    return np.sin(np.pi * x[0] / Lx) * np.sin(np.pi * x[1] / Ly) * decay


# ─── Case 4: 2D Steady-State with Constant Source ────────────────────────────
# PDE:   -k ∇²T = f  on  [0,1] × [0,1]
# BCs:   T = 0 on all boundaries
# Exact (for f constant, 1D reduction via symmetry):
#        T(x,y) = (f / (2k)) x(1-x)    (1D Poisson, y-independent when
#        top/bottom have T=0 and homogeneous in y)
# More precisely, with T=0 on all 4 walls and f constant:
#   T(x,y) = sum_{m,n odd} (16 f) / (k π^4 m n (m²+n²)) sin(mπx) sin(nπy)
# For the simpler 1D-like case: T=0 left/right, insulated top/bottom, f const:
#   T(x) = (f / (2k)) x(Lx - x)

def steady_source_1d_like(x: np.ndarray, f: float = 1.0, k: float = 1.0,
                          Lx: float = 1.0) -> np.ndarray:
    """Steady-state with constant source, Dirichlet T=0 left/right, insulated top/bottom."""
    return (f / (2.0 * k)) * x[0] * (Lx - x[0])


# ─── Case 5: 1D Transient with Dirichlet BCs (non-homogeneous) ──────────────
# PDE:   ρ cp ∂T/∂t - k ∂²T/∂x² = 0  on  [0,L]
# BCs:   T(0,t) = T_L,  T(L,t) = T_R
# IC:    T(x,0) = T_L  (uniform at left wall temperature)
# Exact: T(x,t) = T_L + (T_R - T_L)(x/L)
#        - 2(T_R-T_L)/π Σ_{n=1}^∞ ((-1)^n / n) sin(nπx/L) exp(-α(nπ/L)² t)
# At large t this converges to the steady-state linear profile.

def transient_step_1d(x: np.ndarray, t: float,
                      T_L: float = 0.0, T_R: float = 1.0,
                      k: float = 1.0, rho: float = 1.0, cp: float = 1.0,
                      Lx: float = 1.0, n_terms: int = 200) -> np.ndarray:
    """1D transient heat equation with step IC approaching steady-state linear profile.

    Uses Fourier series with n_terms. Applicable to 2D grids (ignores y coordinate).
    """
    alpha = k / (rho * cp)
    steady = T_L + (T_R - T_L) * x[0] / Lx
    transient = np.zeros_like(x[0])
    for n in range(1, n_terms + 1):
        coeff = -2.0 * (T_R - T_L) / (np.pi * n) * ((-1) ** n)
        transient += coeff * np.sin(n * np.pi * x[0] / Lx) * \
                     np.exp(-alpha * (n * np.pi / Lx) ** 2 * t)
    return steady + transient


# ─── Benchmark Case Definitions ──────────────────────────────────────────────

BENCHMARK_CASES = {  # type: ignore[assignment]
    "steady_linear_2d": {
        "name": "2D Steady-State Linear Profile",
        "description": "Linear temperature gradient between two Dirichlet walls",
        "reference": "Trivial analytical solution; exact for any mesh",
        "config": {
            "dim": 2, "Lx": 1.0, "Ly": 1.0,
            "k": 1.0, "rho": 1.0, "cp": 1.0,
            "source": 0.0, "u_init": 0.5,
            "t_end": 2.0, "dt": 0.05, "theta": 1.0,
            "bcs": [
                {"type": "dirichlet", "boundary": "left",   "value": 0.0},
                {"type": "dirichlet", "boundary": "right",  "value": 1.0},
                {"type": "neumann",   "boundary": "top",    "value": 0.0},
                {"type": "neumann",   "boundary": "bottom", "value": 0.0},
            ],
        },
        "analytical_fn": steady_linear_2d,
        "analytical_kwargs": {"T_L": 0.0, "T_R": 1.0, "Lx": 1.0},
        "expected_order": 2,
    },

    "transient_fourier_2d": {
        "name": "2D Transient Fourier Mode Decay",
        "description": "Single Fourier mode decaying under homogeneous Dirichlet BCs",
        "reference": "Standard separation-of-variables solution",
        "config": {
            "dim": 2, "Lx": 1.0, "Ly": 1.0,
            "k": 1.0, "rho": 1.0, "cp": 1.0,
            "source": 0.0, "u_init": 0.0,
            "t_end": 0.1, "dt": 0.002, "theta": 1.0,
            "bcs": [
                {"type": "dirichlet", "boundary": "left",   "value": 0.0},
                {"type": "dirichlet", "boundary": "right",  "value": 0.0},
                {"type": "dirichlet", "boundary": "top",    "value": 0.0},
                {"type": "dirichlet", "boundary": "bottom", "value": 0.0},
            ],
        },
        "ic_fn": lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]),
        "analytical_fn": transient_fourier_2d,
        "analytical_kwargs": {"k": 1.0, "rho": 1.0, "cp": 1.0, "Lx": 1.0, "Ly": 1.0},
        "expected_order": 2,
        "note": "Requires custom IC (sinusoidal); dt must be small for temporal accuracy",
    },

    "steady_source_2d": {
        "name": "2D Steady-State with Constant Source (1D-like)",
        "description": "Poisson equation with f=1, T=0 left/right, insulated top/bottom",
        "reference": "1D Poisson exact: T = f/(2k) x(L-x)",
        "config": {
            "dim": 2, "Lx": 1.0, "Ly": 1.0,
            "k": 1.0, "rho": 1.0, "cp": 1.0,
            "source": 1.0, "u_init": 0.0,
            "t_end": 5.0, "dt": 0.1, "theta": 1.0,
            "bcs": [
                {"type": "dirichlet", "boundary": "left",   "value": 0.0},
                {"type": "dirichlet", "boundary": "right",  "value": 0.0},
                {"type": "neumann",   "boundary": "top",    "value": 0.0},
                {"type": "neumann",   "boundary": "bottom", "value": 0.0},
            ],
        },
        "analytical_fn": steady_source_1d_like,
        "analytical_kwargs": {"f": 1.0, "k": 1.0, "Lx": 1.0},
        "expected_order": 2,
    },

    # transient_step_2d is excluded: the IC (u=0) is discontinuous with the
    # right Dirichlet BC (T=1), producing a persistent corner singularity whose
    # L2 contribution is mesh-independent and obscures the spatial convergence.
    # Use transient_fourier_2d (smooth IC compatible with homogeneous BCs) for
    # transient V&V instead.
}
