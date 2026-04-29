"""
Plotly Dash Dashboard for PDE Simulation Results

Tabs:
  1. Overview      - run list, status, quick metrics
  2. Field Viewer  - 2D/3D temperature field (heatmap, surface, profiles,
                     gradient, volume, time animation)
  3. Convergence   - L2 norm history plots
  4. Parametric    - study comparison charts
  5. Agent Chat    - interact with agents via REST API (async, no timeouts)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import requests
import dash
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update, ALL
import dash_bootstrap_components as dbc

RESULTS_PATH  = os.getenv("RESULTS_PATH", "/workspace/results")
AGENTS_API    = os.getenv("AGENTS_API_URL", "http://agents:8000")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB   = os.getenv("POSTGRES_DB",   "pde_simulations")
POSTGRES_USER = os.getenv("POSTGRES_USER", "pde_user")
POSTGRES_PASS = os.getenv("POSTGRES_PASSWORD", "pde_secret_change_me")

# Base host for external nav links (Docs, MLflow, MinIO, Neo4j). If set, used as-is;
# otherwise derived from the request host so links work for any access URL.

# ─── App initialization ───────────────────────────────────────────────────────

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
)
app.title = "PDE Agents Dashboard"
server = app.server

# ─── Color scheme ────────────────────────────────────────────────────────────

COLORSCALE = "Plasma"
BG_COLOR   = "#1a1a2e"
CARD_COLOR = "#16213e"
ACCENT     = "#e94560"
TEXT_COLOR = "#eaeaea"

PLOT_LAYOUT = dict(
    paper_bgcolor=CARD_COLOR,
    plot_bgcolor=CARD_COLOR,
    font_color=TEXT_COLOR,
    margin=dict(l=40, r=20, t=40, b=40),
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_result(run_id: str) -> dict:
    p = Path(RESULTS_PATH) / run_id / "result.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def _load_config(run_id: str) -> dict:
    p = Path(RESULTS_PATH) / run_id / "config.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def _load_final_field(run_id: str) -> np.ndarray | None:
    p = Path(RESULTS_PATH) / run_id / "u_final.npy"
    if not p.exists():
        return None
    return np.load(p)


def _get_run_ids() -> list[str]:
    """Scan results directory for completed runs."""
    results_dir = Path(RESULTS_PATH)
    if not results_dir.exists():
        return []
    return sorted(
        [d.name for d in results_dir.iterdir()
         if d.is_dir() and (d / "result.json").exists()],
        reverse=True,
    )


def _get_db_kpi_stats() -> dict:
    """
    Query PostgreSQL for authoritative KPI stats.

    PostgreSQL is the single source of truth for:
      - Run counts (including failed runs that never wrote a result.json)
      - Wall time (total orchestration time, not just the FEniCS solver time)

    Falls back gracefully to zero-values when the DB is unreachable.
    """
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=POSTGRES_HOST, port=POSTGRES_PORT,
            dbname=POSTGRES_DB, user=POSTGRES_USER, password=POSTGRES_PASS,
            connect_timeout=3,
        )
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*)                                                      AS total,
                    COUNT(*) FILTER (WHERE status::text = 'SUCCESS')             AS success,
                    COUNT(*) FILTER (WHERE status::text = 'FAILED')              AS failed,
                    ROUND(AVG(wall_time)::numeric, 2)                            AS avg_wall,
                    ROUND(MAX(wall_time)::numeric, 1)                            AS max_wall
                FROM simulation_runs
            """)
            row = cur.fetchone()
        conn.close()
        total, success, failed, avg_wall, max_wall = row
        return {
            "total":    total    or 0,
            "success":  success  or 0,
            "failed":   failed   or 0,
            "avg_wall": float(avg_wall) if avg_wall else 0.0,
            "max_wall": float(max_wall) if max_wall else 0.0,
            "source":   "db",
        }
    except Exception as exc:
        return {"total": 0, "success": 0, "failed": 0,
                "avg_wall": 0.0, "max_wall": 0.0, "source": "error", "error": str(exc)}


def _api_call(method: str, endpoint: str, payload: dict = None) -> dict:
    try:
        url = f"{AGENTS_API}/{endpoint.lstrip('/')}"
        if method == "GET":
            r = requests.get(url, timeout=30)
        else:
            # POST to async endpoints returns immediately; 30s is plenty
            r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ─── Field Viewer helpers ──────────────────────────────────────────────────────

def _load_dof_coords(run_id: str) -> "np.ndarray | None":
    p = Path(RESULTS_PATH) / run_id / "dof_coords.npy"
    return np.load(p) if p.exists() else None


def _list_snapshots(run_id: str) -> list[int]:
    d = Path(RESULTS_PATH) / run_id / "snapshots"
    if not d.exists():
        return []
    return sorted(int(f.stem.split("_")[1]) for f in sorted(d.glob("u_*.npy")))


def _load_snapshot(run_id: str, idx: int) -> "np.ndarray | None":
    p = Path(RESULTS_PATH) / run_id / "snapshots" / f"u_{idx:04d}.npy"
    return np.load(p) if p.exists() else None


def _load_snapshot_times(run_id: str) -> "np.ndarray | None":
    p = Path(RESULTS_PATH) / run_id / "snapshot_times.npy"
    return np.load(p) if p.exists() else None


def _interpolate_2d(coords: np.ndarray, values: np.ndarray,
                    resolution: int = 250) -> tuple:
    """Interpolate scattered DOF data onto a regular grid via scipy griddata."""
    from scipy.interpolate import griddata
    x0, x1 = coords[:, 0].min(), coords[:, 0].max()
    y0, y1 = coords[:, 1].min(), coords[:, 1].max()
    xi = np.linspace(x0, x1, resolution)
    yi = np.linspace(y0, y1, resolution)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata(coords[:, :2], values, (Xi, Yi), method="linear")
    return xi, yi, Xi, Yi, Zi


def _slice_3d(coords: np.ndarray, values: np.ndarray,
              axis: int, frac: float, resolution: int = 150):
    """
    Extract a 2D slice perpendicular to `axis` (0=X, 1=Y, 2=Z) at
    fractional position `frac` ∈ [0, 1] and interpolate to a regular grid.
    """
    from scipy.interpolate import griddata

    lo, hi = coords[:, axis].min(), coords[:, axis].max()
    cut = lo + frac * (hi - lo)
    tol = (hi - lo) / max(coords.shape[0] ** (1 / 3), 1) * 1.5

    mask = np.abs(coords[:, axis] - cut) < tol
    if mask.sum() < 4:
        return None, None, None, None, None

    other = [i for i in range(3) if i != axis]
    h_ax, v_ax = other[0], other[1]
    h_coords = coords[mask, h_ax]
    v_coords = coords[mask, v_ax]
    vals = values[mask]

    hi_h, lo_h = h_coords.max(), h_coords.min()
    hi_v, lo_v = v_coords.max(), v_coords.min()
    hi_ = np.linspace(lo_h, hi_h, resolution)
    vi_ = np.linspace(lo_v, hi_v, resolution)
    H, V = np.meshgrid(hi_, vi_)
    Z = griddata(np.column_stack([h_coords, v_coords]), vals, (H, V), method="linear")
    return hi_, vi_, H, V, Z


def _make_empty_fig(msg: str = "No data") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(color=TEXT_COLOR, size=14))
    fig.update_layout(**PLOT_LAYOUT)
    return fig


def _build_field_figure(run_id: str, u: np.ndarray, coords: np.ndarray,
                         config: dict, view: str, **kwargs) -> go.Figure:
    """Build the main field plot for a given view mode."""
    dim = config.get("dim", 2)
    T_lo, T_hi = float(u.min()), float(u.max())
    dT = T_hi - T_lo or 1.0
    cb = dict(title="T [K]", titleside="right",
              tickfont=dict(color=TEXT_COLOR), titlefont=dict(color=TEXT_COLOR))

    # ── 2-D views ─────────────────────────────────────────────────────────────
    if view == "heatmap":
        xi, yi, Xi, Yi, Zi = _interpolate_2d(coords, u)
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            x=xi, y=yi, z=Zi,
            colorscale=COLORSCALE, colorbar=cb,
            zmin=T_lo, zmax=T_hi,
            hovertemplate="x: %{x:.3f}<br>y: %{y:.3f}<br>T: %{z:.2f} K<extra></extra>",
        ))
        n_iso = 12
        fig.add_trace(go.Contour(
            x=xi, y=yi, z=Zi,
            showscale=False,
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
            contours=dict(coloring="none", showlabels=True,
                          labelfont=dict(size=9, color="white"),
                          start=T_lo, end=T_hi, size=dT / n_iso),
            line=dict(color="white", width=0.7),
        ))
        fig.update_layout(title=f"Temperature Heatmap — {run_id}",
                          xaxis_title="x [m]", yaxis_title="y [m]", **PLOT_LAYOUT)

    elif view == "surface":
        xi, yi, Xi, Yi, Zi = _interpolate_2d(coords, u, resolution=150)
        fig = go.Figure(go.Surface(
            x=Xi, y=Yi, z=Zi,
            colorscale=COLORSCALE, colorbar=cb,
            contours={"z": {"show": True, "start": T_lo, "end": T_hi,
                            "size": dT / 10, "color": "white", "width": 1}},
            lighting=dict(ambient=0.6, diffuse=0.8, specular=0.3),
        ))
        fig.update_layout(
            title=f"3-D Surface — {run_id}",
            scene=dict(
                xaxis_title="x [m]", yaxis_title="y [m]", zaxis_title="T [K]",
                xaxis=dict(backgroundcolor=CARD_COLOR, gridcolor="#333", color=TEXT_COLOR),
                yaxis=dict(backgroundcolor=CARD_COLOR, gridcolor="#333", color=TEXT_COLOR),
                zaxis=dict(backgroundcolor=CARD_COLOR, gridcolor="#333", color=TEXT_COLOR),
                bgcolor=CARD_COLOR,
            ),
            **PLOT_LAYOUT,
        )

    elif view == "gradient":
        xi, yi, Xi, Yi, Zi = _interpolate_2d(coords, u)
        # Compute -k∇T (heat flux direction proportional to -∇T)
        k = config.get("k", 1.0)
        dT_dy = np.gradient(Zi, yi, axis=0)
        dT_dx = np.gradient(Zi, xi, axis=1)
        flux_mag = np.sqrt(dT_dx**2 + dT_dy**2) * k
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            x=xi, y=yi, z=flux_mag,
            colorscale="Inferno",
            colorbar=dict(title="|q| [W/m²]", titleside="right",
                          tickfont=dict(color=TEXT_COLOR), titlefont=dict(color=TEXT_COLOR)),
        ))
        # Quiver arrows (sub-sample)
        step = max(1, len(xi) // 20)
        Xq, Yq = Xi[::step, ::step], Yi[::step, ::step]
        Uq = -dT_dx[::step, ::step] * k
        Vq = -dT_dy[::step, ::step] * k
        mag = np.sqrt(Uq**2 + Vq**2) + 1e-12
        scale = (xi[-1] - xi[0]) / len(xi) * step * 0.7
        Uq, Vq = Uq / mag * scale, Vq / mag * scale
        for i in range(Xq.shape[0]):
            for j in range(Xq.shape[1]):
                if not np.isnan(Uq[i, j]):
                    fig.add_annotation(
                        x=Xq[i, j] + Uq[i, j], y=Yq[i, j] + Vq[i, j],
                        ax=Xq[i, j], ay=Yq[i, j],
                        xref="x", yref="y", axref="x", ayref="y",
                        showarrow=True, arrowhead=2, arrowsize=1,
                        arrowwidth=1, arrowcolor="white", opacity=0.6,
                    )
        fig.update_layout(title=f"Heat Flux Magnitude — {run_id}",
                          xaxis_title="x [m]", yaxis_title="y [m]", **PLOT_LAYOUT)

    elif view == "profiles":
        xi, yi, Xi, Yi, Zi = _interpolate_2d(coords, u)
        fig = go.Figure()
        n_lines = 6
        colors = ["#e94560", "#00b4d8", "#90e0ef", "#f72585", "#4cc9f0", "#ffb703"]
        # Horizontal profiles (T vs x at fixed y)
        for i, frac in enumerate(np.linspace(0.1, 0.9, n_lines)):
            row = int(frac * (len(yi) - 1))
            fig.add_trace(go.Scatter(
                x=xi, y=Zi[row, :], mode="lines", name=f"y={yi[row]:.2f}",
                line=dict(color=colors[i % len(colors)], width=1.5),
            ))
        fig.update_layout(
            title=f"Temperature Profiles along X — {run_id}",
            xaxis_title="x [m]", yaxis_title="T [K]",
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_COLOR)),
            **PLOT_LAYOUT,
        )

    # ── 3-D views ─────────────────────────────────────────────────────────────
    elif view in ("z_slice", "x_slice", "y_slice"):
        axis_map = {"z_slice": (2, "Z", "x [m]", "y [m]"),
                    "x_slice": (0, "X", "y [m]", "z [m]"),
                    "y_slice": (1, "Y", "x [m]", "z [m]")}
        ax, label, xl, yl = axis_map[view]
        # frac is passed via kwargs when called from the main callback; default 0.5
        frac = kwargs.get("frac", 0.5)
        h_, v_, H, V, Z = _slice_3d(coords, u, axis=ax, frac=frac)
        if Z is None:
            return _make_empty_fig(f"Not enough points for {label}-slice — try a different position")
        fig = go.Figure(go.Heatmap(
            x=h_, y=v_, z=Z,
            colorscale=COLORSCALE, colorbar=cb,
            zmin=T_lo, zmax=T_hi,
            hovertemplate=f"{xl[0]}: %{{x:.3f}}<br>{yl[0]}: %{{y:.3f}}<br>T: %{{z:.2f}} K<extra></extra>",
        ))
        fig.update_layout(title=f"{label}-Slice (pos={frac:.2f}) — {run_id}",
                          xaxis_title=xl, yaxis_title=yl, **PLOT_LAYOUT)

    elif view == "volume":
        if dim != 3:
            return _make_empty_fig("Volume view is only available for 3-D runs")

        # Build 3 orthogonal interpolated slices (X-mid, Y-mid, Z-mid)
        # plus 3 isosurface levels — much more reliable than go.Volume
        from scipy.interpolate import griddata

        fig = go.Figure()
        scene_3d = dict(
            xaxis=dict(backgroundcolor=CARD_COLOR, gridcolor="#333",
                       color=TEXT_COLOR, title="x [m]"),
            yaxis=dict(backgroundcolor=CARD_COLOR, gridcolor="#333",
                       color=TEXT_COLOR, title="y [m]"),
            zaxis=dict(backgroundcolor=CARD_COLOR, gridcolor="#333",
                       color=TEXT_COLOR, title="z [m]"),
            bgcolor=CARD_COLOR,
        )

        # Draw 3 orthogonal slice planes
        slices_cfg = [
            (0, 0.5, "X"),   # YZ plane at X=0.5
            (1, 0.5, "Y"),   # XZ plane at Y=0.5
            (2, 0.5, "Z"),   # XY plane at Z=0.5
        ]
        res = 60  # lower res for 3D to stay interactive
        for ax_idx, frac, label in slices_cfg:
            lo = coords[:, ax_idx].min()
            hi = coords[:, ax_idx].max()
            cut = lo + frac * (hi - lo)
            tol = (hi - lo) / max(coords.shape[0] ** (1/3), 1) * 2.0
            mask = np.abs(coords[:, ax_idx] - cut) < tol
            if mask.sum() < 6:
                continue
            other = [i for i in range(3) if i != ax_idx]
            a, b = other[0], other[1]
            pa = coords[mask, a]; pb = coords[mask, b]; vals = u[mask]
            ai = np.linspace(pa.min(), pa.max(), res)
            bi = np.linspace(pb.min(), pb.max(), res)
            Ai, Bi = np.meshgrid(ai, bi)
            Zi = griddata(np.column_stack([pa, pb]), vals, (Ai, Bi), method="linear")

            # Reconstruct XYZ for the surface
            xa = ya = za = None
            if ax_idx == 0:
                xa = np.full_like(Ai, cut); ya = Ai; za = Bi
            elif ax_idx == 1:
                xa = Ai; ya = np.full_like(Ai, cut); za = Bi
            else:
                xa = Ai; ya = Bi; za = np.full_like(Ai, cut)

            fig.add_trace(go.Surface(
                x=xa, y=ya, z=za,
                surfacecolor=Zi,
                colorscale=COLORSCALE,
                cmin=T_lo, cmax=T_hi,
                showscale=(label == "Z"),  # show colorbar once
                colorbar=cb if label == "Z" else None,
                opacity=0.85,
                name=f"{label}-slice",
                hovertemplate=f"{label}-slice<br>T: %{{customdata:.2f}} K<extra></extra>",
                customdata=Zi,
            ))

        # Add isosurface ribbons at 3 temperature levels
        n_iso = 3
        for level in np.linspace(T_lo + (T_hi - T_lo) * 0.25,
                                  T_hi - (T_hi - T_lo) * 0.25, n_iso):
            fig.add_trace(go.Isosurface(
                x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                value=u,
                isomin=level - 0.5, isomax=level + 0.5,
                colorscale=COLORSCALE, cmin=T_lo, cmax=T_hi,
                showscale=False,
                opacity=0.25,
                caps=dict(x_show=False, y_show=False, z_show=False),
                name=f"T={level:.0f} K",
            ))

        fig.update_layout(
            title=f"3-D Slices + Isosurfaces — {run_id}",
            scene=scene_3d,
            **PLOT_LAYOUT,
        )

    else:
        return _make_empty_fig(f"Unknown view: {view}")

    return fig


def _build_secondary_figure(run_id: str, u: np.ndarray, coords: np.ndarray,
                              config: dict, view: str) -> go.Figure:
    """Secondary plot: Y-profiles for heatmap, or heat-flux X-component."""
    dim = config.get("dim", 2)
    try:
        if view in ("heatmap", "surface", "animation") and dim == 2:
            xi, yi, Xi, Yi, Zi = _interpolate_2d(coords, u)
            fig = go.Figure()
            n_lines = 5
            colors = ["#e94560", "#00b4d8", "#90e0ef", "#f72585", "#ffb703"]
            for i, frac in enumerate(np.linspace(0.1, 0.9, n_lines)):
                col = int(frac * (len(xi) - 1))
                valid = ~np.isnan(Zi[:, col])
                if valid.sum() < 2:
                    continue
                fig.add_trace(go.Scatter(
                    x=Zi[valid, col], y=yi[valid], mode="lines",
                    name=f"x={xi[col]:.2f}",
                    line=dict(color=colors[i % len(colors)], width=1.8),
                ))
            fig.update_layout(
                title="T(y) Profiles at fixed X",
                xaxis_title="T [K]", yaxis_title="y [m]",
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_COLOR)),
                **PLOT_LAYOUT,
            )
            return fig

        if view == "gradient" and dim == 2:
            xi, yi, Xi, Yi, Zi = _interpolate_2d(coords, u)
            k = config.get("k", 1.0)
            dT_dx = np.gradient(Zi, xi, axis=1)
            fig = go.Figure(go.Heatmap(
                x=xi, y=yi, z=-dT_dx * k,
                colorscale="RdBu",
                colorbar=dict(title="qx [W/m²]", titleside="right",
                              tickfont=dict(color=TEXT_COLOR),
                              titlefont=dict(color=TEXT_COLOR)),
            ))
            fig.update_layout(title="X-component of Heat Flux (qx)",
                              xaxis_title="x [m]", yaxis_title="y [m]",
                              **PLOT_LAYOUT)
            return fig

        if view == "profiles" and dim == 2:
            # Show vertical (Y) profiles
            xi, yi, Xi, Yi, Zi = _interpolate_2d(coords, u)
            fig = go.Figure()
            colors = ["#e94560", "#00b4d8", "#90e0ef", "#f72585", "#ffb703", "#4cc9f0"]
            for i, frac in enumerate(np.linspace(0.1, 0.9, 6)):
                row = int(frac * (len(yi) - 1))
                valid = ~np.isnan(Zi[row, :])
                if valid.sum() < 2:
                    continue
                fig.add_trace(go.Scatter(
                    x=xi[valid], y=Zi[row, valid], mode="lines",
                    name=f"y={yi[row]:.2f}",
                    line=dict(color=colors[i % len(colors)], width=1.8),
                ))
            fig.update_layout(
                title="T(x) Profiles at fixed Y",
                xaxis_title="x [m]", yaxis_title="T [K]",
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_COLOR)),
                **PLOT_LAYOUT,
            )
            return fig
    except Exception:
        pass
    return _make_empty_fig()


def _build_stats_panel(run_id: str, u: np.ndarray, config: dict) -> list:
    """Build a stats summary card body."""
    k   = config.get("k",   1.0)
    rho = config.get("rho", 1.0)
    cp  = config.get("cp",  1.0)
    alpha = k / (rho * cp) if (rho * cp) else 0
    L = 1.0
    tau = L**2 / alpha if alpha else float("inf")

    rows = [
        ("Run ID",        run_id),
        ("Dim",           f"{config.get('dim', '?')}D"),
        ("Mesh",          f"{config.get('nx')}×{config.get('ny')}"
                          + (f"×{config.get('nz')}" if config.get("nz") else "")),
        ("DOFs",          f"{len(u):,}"),
        ("T_min",         f"{u.min():.2f} K"),
        ("T_max",         f"{u.max():.2f} K"),
        ("T_mean",        f"{u.mean():.2f} K"),
        ("k",             f"{k} W/(m·K)"),
        ("ρ c_p",         f"{rho*cp:.1f} J/(m³·K)"),
        ("α = k/ρcₚ",    f"{alpha:.3e} m²/s"),
        ("τ = L²/α",     f"{tau:.1f} s"),
    ]
    return [dbc.Table(
        html.Tbody([html.Tr([
            html.Td(k, style={"color": "#aaa", "fontSize": "0.82rem",
                              "paddingRight": "8px"}),
            html.Td(v, style={"fontSize": "0.82rem"}),
        ]) for k, v in rows]),
        size="sm", bordered=False,
    )]


# ─── Layout ───────────────────────────────────────────────────────────────────

def make_header():
    # Initial hrefs default to '#'; the set_external_nav_links callback
    # updates them immediately on page load using the browser's actual hostname.
    return dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand("🔬 PDE Agents Dashboard", style={"fontSize": "1.4rem", "fontWeight": "bold"}),
            dbc.Nav([
                dbc.NavItem(dbc.NavLink("API Docs",      href="#", target="_blank", external_link=True, id="nav-link-docs")),
                dbc.NavItem(dbc.NavLink("MinIO",         href="#", target="_blank", external_link=True, id="nav-link-minio")),
                dbc.NavItem(dbc.NavLink("Neo4j Browser", href="#", target="_blank", external_link=True,
                    style={"color": "#6fd672"}, id="nav-link-neo4j")),
                dbc.NavItem(dbc.NavLink("NeoDash 🧠",   href="#", target="_blank", external_link=True,
                    style={"color": "#b48eff"}, id="nav-link-neodash")),
            ], navbar=True),
        ]),
        color="dark", dark=True, className="mb-3",
    )


def make_kpi_row():
    return dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H2("–", id="kpi-total-runs", className="text-info"),
                html.P("Total Runs", className="text-muted mb-0"),
                html.Small("from database", className="text-muted",
                           style={"fontSize": "0.7rem", "opacity": "0.6"}),
            ])
        ], color="dark", outline=True), width=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H2("–", id="kpi-success-runs", className="text-success"),
                html.P("Successful", className="text-muted mb-0"),
            ])
        ], color="dark", outline=True), width=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H2("–", id="kpi-failed-runs", className="text-danger"),
                html.P("Failed", className="text-muted mb-0"),
            ])
        ], color="dark", outline=True), width=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H2("–", id="kpi-avg-time", className="text-warning"),
                html.P([
                    "Avg Wall Time (s) ",
                    html.Small(id="kpi-max-time", className="text-muted",
                               style={"fontSize": "0.75rem"}),
                ], className="text-muted mb-0"),
                html.Small("incl. orchestration", className="text-muted",
                           style={"fontSize": "0.7rem", "opacity": "0.6"}),
            ])
        ], color="dark", outline=True), width=3),
    ], className="mb-3")


app.layout = dbc.Container([
    dcc.Location(id="url", refresh=False),
    make_header(),
    dcc.Interval(id="refresh-interval", interval=10_000, n_intervals=0),

    make_kpi_row(),

    dbc.Tabs([
        # ── Tab 1: Overview ──────────────────────────────────────────────────
        dbc.Tab(label="📊 Overview", tab_id="tab-overview", children=[

            # ── Row 1: Recent runs table + Inspect panel ─────────────────────
            dbc.Row([
                # Recent runs table (auto-populated, no selection required)
                dbc.Col([
                    html.Div([
                        html.Span("Recent Simulation Runs",
                                  style={"fontWeight": "600", "fontSize": "0.95rem"}),
                        html.Span(id="recent-runs-count",
                                  className="text-muted ms-2",
                                  style={"fontSize": "0.78rem"}),
                    ], className="mt-3 mb-2"),
                    html.Div(id="recent-runs-table"),
                ], width=8),

                # Inspect panel — select a run for full details
                dbc.Col([
                    html.Div([
                        html.P("Inspect Run", className="mt-3 mb-1",
                               style={"fontWeight": "600", "fontSize": "0.95rem"}),
                    dcc.Dropdown(
                        id="run-selector",
                        options=[],
                            placeholder="Select a run…",
                        style={"color": "#000"},
                    ),
                        html.Div(id="run-detail-card", className="mt-2"),
                    ]),

                    # System health
                    html.Hr(style={"borderColor": "#2a2a3e", "margin": "14px 0"}),
                    html.P("System Health", className="mb-2",
                           style={"fontWeight": "600", "fontSize": "0.95rem"}),
                    html.Div(id="system-health-panel"),
                ], width=4),
            ]),

            # ── Row 2: Scaling chart + T_max distribution ────────────────────
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="status-pie-chart",
                              style={"height": "300px"}),
                ], width=6),
                dbc.Col([
                    dcc.Graph(id="wall-time-bar-chart",
                              style={"height": "300px"}),
                ], width=6),
            ], className="mt-2"),
        ]),

        # ── Tab 2: Field Viewer ───────────────────────────────────────────────
        dbc.Tab(label="🌡️ Field Viewer", tab_id="tab-field", children=[
            # Animation state & interval
            dcc.Store(id="field-anim-store", data={"step": 0, "playing": False}),
            dcc.Interval(id="field-anim-interval", interval=600, disabled=True),

            dbc.Row([
                # ── Controls column ────────────────────────────────────────
                dbc.Col([
                    html.H5("Field Viewer", className="mt-3"),
                    dcc.Dropdown(id="field-run-selector", options=[],
                                 placeholder="Select a run...",
                                 style={"color": "#000"}, className="mb-2"),

                    html.Label("View mode", className="text-muted",
                               style={"fontSize": "0.8rem"}),
                    dbc.RadioItems(
                        id="field-view-mode",
                        options=[
                            {"label": "🌡 Heatmap + Isolines", "value": "heatmap"},
                            {"label": "🏔 3D Surface",          "value": "surface"},
                            {"label": "∇ Heat Flux",           "value": "gradient"},
                            {"label": "〰 Profiles",            "value": "profiles"},
                            {"label": "🍕 Z-Slice (3D)",       "value": "z_slice"},
                            {"label": "📦 Volume (3D)",         "value": "volume"},
                            {"label": "🎬 Animation",           "value": "animation"},
                        ],
                        value="heatmap",
                        className="mt-1",
                        inputStyle={"marginRight": "6px"},
                        labelStyle={"fontSize": "0.82rem", "display": "block",
                                    "marginBottom": "3px"},
                    ),

                    # Slice controls (shown for 3D modes)
                    html.Div(id="field-slice-panel", children=[
                        html.Hr(style={"borderColor": "#333", "marginTop": "10px"}),
                        html.Label("Slice position", style={"fontSize": "0.8rem",
                                                             "color": "#aaa"}),
                        dcc.Slider(id="z-slice-slider", min=0, max=1, step=0.05,
                                   value=0.5, tooltip={"placement": "right"},
                                   marks={0: "0", 0.5: "½", 1: "1"}),
                    ], style={"display": "none"}),

                    # Animation controls
                    html.Div(id="field-anim-panel", children=[
                        html.Hr(style={"borderColor": "#333", "marginTop": "10px"}),
                        html.Label("Time step", style={"fontSize": "0.8rem",
                                                        "color": "#aaa"}),
                        dcc.Slider(id="field-step-slider", min=0, max=0, step=1,
                                   value=0, tooltip={"placement": "right"},
                                   marks={0: "0"}),
                        html.Div(id="field-time-display",
                                 style={"fontSize": "0.78rem", "color": "#aaa",
                                        "textAlign": "center", "marginTop": "4px"}),
                        dbc.Button("▶ Play", id="field-play-btn", color="secondary",
                                   size="sm", className="mt-2 w-100"),
                    ], style={"display": "none"}),

                    html.Div(id="field-info", className="mt-3"),
                ], width=3),

                # ── Main plot ──────────────────────────────────────────────
                dbc.Col([
                    dcc.Graph(id="field-main-plot",
                              style={"height": "550px"},
                              config={"scrollZoom": True,
                                      "toImageButtonOptions": {"format": "png",
                                                               "scale": 2}}),
                ], width=9),
            ]),

            # ── Secondary row ──────────────────────────────────────────────
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="field-secondary-plot", style={"height": "280px"}),
                ], width=7),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Thermal Statistics",
                                       style={"fontSize": "0.85rem", "padding": "6px 12px"}),
                        dbc.CardBody(id="field-stats-card",
                                     style={"padding": "8px 12px"}),
                    ], color="dark", outline=True, className="mt-2"),
                ], width=5),
            ], className="mt-2"),
        ]),

        # ── Tab 3: Convergence ────────────────────────────────────────────────
        dbc.Tab(label="📈 Convergence", tab_id="tab-convergence", children=[
            dbc.Row([
                dbc.Col([
                    html.H5("Select Runs", className="mt-3"),
                    dcc.Dropdown(
                        id="conv-run-selector",
                        options=[], multi=True,
                        placeholder="Select runs to compare...",
                        style={"color": "#000"},
                    ),
                ], width=12),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="convergence-plot"), width=8),
                dbc.Col([
                    html.Div(id="convergence-stats", className="mt-3"),
                ], width=4),
            ]),
        ]),

        # ── Tab 4: Parametric Studies ────────────────────────────────────────
        dbc.Tab(label="🔬 Parametric", tab_id="tab-parametric", children=[
            dbc.Row([
                dbc.Col([
                    html.H5("Parametric Analysis", className="mt-3"),
                    html.P("Compare runs across a swept parameter.", className="text-muted"),
                    dcc.Dropdown(
                        id="param-runs-selector",
                        options=[], multi=True,
                        placeholder="Select runs for comparison...",
                        style={"color": "#000"},
                    ),
                    html.Label("X-axis parameter:", className="mt-2"),
                    dcc.Dropdown(
                        id="param-x-selector",
                        options=[
                            {"label": "k (conductivity)", "value": "k"},
                            {"label": "nx (mesh res)", "value": "nx"},
                            {"label": "dt (time step)", "value": "dt"},
                            {"label": "source (heat gen)", "value": "source"},
                        ],
                        value="k",
                        style={"color": "#000"},
                    ),
                ], width=3),
                dbc.Col([
                    dcc.Graph(id="parametric-scatter"),
                ], width=9),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="parametric-wall-time"), width=6),
                dbc.Col(dcc.Graph(id="parametric-t-range"), width=6),
            ]),
        ]),

        # ── Tab 5: Agent Chat ────────────────────────────────────────────────
        dbc.Tab(label="🤖 Agent Chat", tab_id="tab-chat", children=[
            dcc.Store(id="chat-job-store", data={}),
            dcc.Interval(id="chat-poll-interval", interval=3000,
                         n_intervals=0, disabled=True),
            dbc.Row([
                # ── Left: agent selector + quick prompts ─────────────────────
                dbc.Col([
                    html.H5("Agent", className="mt-3 mb-1"),
                    dbc.Select(
                        id="agent-selector",
                        options=[
                            {"label": "🎯 Orchestrator (All Agents)", "value": "orchestrator"},
                            {"label": "⚙️ Simulation Agent",          "value": "simulation"},
                            {"label": "📊 Analytics Agent",           "value": "analytics"},
                            {"label": "🗄️ Database Agent",            "value": "database"},
                        ],
                        value="orchestrator",
                    ),

                    html.Hr(style={"borderColor": "#2a2a3e", "marginTop": "12px"}),
                    html.P("Quick prompts", className="text-muted mb-2",
                           style={"fontSize": "0.78rem", "textTransform": "uppercase",
                                  "letterSpacing": "0.05em"}),

                    # ── History & analysis quick prompts ──────────────────────
                    html.P("🔍 History & Analysis",
                           style={"fontSize": "0.75rem", "color": "#00b4d8",
                                  "marginBottom": "4px", "fontWeight": "bold"}),
                    *[
                        dbc.Button(label, id={"type": "quick-prompt", "index": idx},
                                   n_clicks=0, size="sm",
                                   className="mb-1 w-100 text-start",
                                   style={
                                       "fontSize": "0.74rem",
                                       "textAlign": "left",
                                       "backgroundColor": "#1e2a3a",
                                       "color": "#d0e8ff",
                                       "border": "1px solid #2e4a6a",
                                   })
                        for idx, (label, _) in enumerate([
                            ("📋 List all past runs",
                             "Show me all simulation runs in the database with their status, "
                             "material, T_max, and wall time."),
                            ("❌ Show failed runs",
                             "List all failed simulation runs. For each, explain what went wrong "
                             "and suggest how to fix the configuration."),
                            ("🏆 Best convergence",
                             "Which run had the best convergence (lowest final L2 norm)? "
                             "Show its full configuration and results."),
                            ("🌡 Highest temperature",
                             "Which run recorded the highest peak temperature? "
                             "Show its config and explain why it reached that temperature."),
                            ("📊 Analyze recent 5 runs",
                             "Analyze my last 5 simulation runs. Summarize T_max, T_min, "
                             "convergence, and wall time. Highlight any trends."),
                            ("🔬 Compare by material",
                             "Compare all successful runs grouped by material. "
                             "Which material gave the most uniform temperature distribution "
                             "and fastest solve time?"),
                            ("💡 Suggest next run",
                             "Based on all past runs and knowledge graph patterns, "
                             "what is the most informative next simulation to run? "
                             "Identify gaps in BC pattern or material coverage and propose "
                             "a specific configuration."),
                        ])
                    ],

                    html.Hr(style={"borderColor": "#2a2a3e", "marginTop": "8px"}),
                    # ── Knowledge Graph quick prompts ─────────────────────────
                    html.P("🧠 Knowledge Graph",
                           style={"fontSize": "0.75rem", "color": "#b48eff",
                                  "marginBottom": "4px", "fontWeight": "bold"}),
                    *[
                        dbc.Button(label, id={"type": "quick-prompt", "index": idx + 10},
                                   n_clicks=0, size="sm",
                                   className="mb-1 w-100 text-start",
                                   style={
                                       "fontSize": "0.74rem",
                                       "textAlign": "left",
                                       "backgroundColor": "#1e1a2e",
                                       "color": "#e0d4ff",
                                       "border": "1px solid #4a3a6e",
                                   })
                        for idx, (label, _) in enumerate([
                            ("🔍 BC pattern outcomes",
                             "Query the knowledge graph: compare all boundary condition patterns "
                             "(dirichlet+neumann, dirichlet+robin, robin). "
                             "For each pattern, what T_max range, average solve time, and "
                             "materials were used? Which pattern is best for uniform heat distribution?"),
                            ("📏 Domain size impact",
                             "Query the knowledge graph: how does domain size affect simulation outcomes? "
                             "Compare micro, component, panel, and structural scale domains — "
                             "show average solve time, DOF count, and T_max for each class."),
                            ("⚡ Thermal class comparison",
                             "Query the knowledge graph: compare high-conductors (copper, aluminium, silicon) "
                             "vs medium-conductors (steel, titanium) vs low-conductors (concrete, glass) vs "
                             "thermal insulators (water, air). What T_max ranges and solve times do each class produce?"),
                            ("🔗 Find similar to steel+robin",
                             "Using the knowledge graph, find past runs most similar to: "
                             "steel (k=50), 2D, dirichlet+robin BCs, component-scale domain (4cm × 2cm). "
                             "Show the top 5 matches with their outcomes and relevance scores."),
                            ("📖 Physics references for copper",
                             "Retrieve the physics references for a copper simulation "
                             "(k=385, rho=8960, cp=385) with robin BC (convective cooling). "
                             "What are the validity limits and recommended h coefficients?"),
                            ("⚙️ Solver guidance",
                             "What are the recommended mesh resolution, time-step stability criteria, "
                             "and element degree choices for heat conduction simulations? "
                             "Use the knowledge graph physics references to answer."),
                        ])
                    ],

                    html.Hr(style={"borderColor": "#2a2a3e", "marginTop": "8px"}),
                    # ── Simulation quick prompts ──────────────────────────────
                    html.P("⚙️ New Simulations",
                           style={"fontSize": "0.75rem", "color": "#e9a440",
                                  "marginBottom": "4px", "fontWeight": "bold"}),
                    *[
                        dbc.Button(label, id={"type": "quick-prompt", "index": idx + 20},
                                   n_clicks=0, size="sm",
                                   className="mb-1 w-100 text-start",
                                   style={
                                       "fontSize": "0.74rem",
                                       "textAlign": "left",
                                       "backgroundColor": "#1e2a1e",
                                       "color": "#d4f0d4",
                                       "border": "1px solid #2e5a2e",
                                   })
                        for idx, (label, _) in enumerate([
                            ("🔲 2D steel plate",
                             "Run a 2D heat equation on a steel plate 4cm × 2cm. "
                             "k=50, rho=7850, cp=490. Left edge T=800K, right edge T=300K, "
                             "top/bottom insulated (Neumann). t_end=10, dt=0.1, nx=40, ny=20."),
                            ("🧊 3D aluminium block",
                             "Run a 3D heat equation on a 2cm × 2cm × 2cm aluminium block. "
                             "k=205, rho=2700, cp=900. "
                             "One face at 500K, opposite face at 300K, all other faces insulated. "
                             "t_end=5, dt=0.05, nx=ny=nz=20."),
                            ("🌊 Convective cooling (copper)",
                             "Run a 2D heat simulation on copper (k=400, rho=8960, cp=385), "
                             "4cm × 2cm domain. Left edge fixed at T=800K, right edge Robin BC "
                             "with h=500 W/m²K and T_inf=300K (convective cooling), "
                             "top and bottom insulated. t_end=2, dt=0.05, nx=40, ny=20."),
                            ("🔥 Internal heat source",
                             "Run a 2D heat equation with a strong internal heat source of "
                             "1e6 W/m³ on a 1cm × 1cm silicon domain (k=148, rho=2330, cp=710). "
                             "All boundaries convective Robin with h=200, T_inf=300K. "
                             "Mesh 40×40, t_end=1, dt=0.01."),
                            ("🏗️ Concrete panel (large)",
                             "Run a 2D heat equation on a concrete wall 50cm × 20cm "
                             "(k=1.7, rho=2300, cp=880, Lx=0.5, Ly=0.2). "
                             "Left face T=600K (fire side), right face T=293K (ambient), "
                             "top/bottom insulated. nx=50, ny=20, t_end=3600, dt=30."),
                        ])
                    ],
                ], width=3, style={"borderRight": "1px solid #2a2a3e",
                                   "paddingRight": "12px"}),

                # ── Right: chat window ────────────────────────────────────────
                dbc.Col([
                    html.H5("Chat", className="mt-3 mb-1"),
                    html.P(
                        "Tasks run in the background — no timeouts. "
                        "Use the quick prompts on the left or type your own.",
                        className="text-muted", style={"fontSize": "0.8rem"},
                    ),
                    html.Div(
                        id="chat-history",
                        style={
                            "height": "430px", "overflowY": "auto",
                            "backgroundColor": "#0f0f1a", "padding": "10px",
                            "borderRadius": "8px", "marginTop": "4px",
                            "fontFamily": "monospace", "fontSize": "0.85rem",
                        },
                    ),
                    html.Div(
                        id="chat-status-bar",
                        style={"marginTop": "6px", "minHeight": "22px",
                               "fontFamily": "monospace", "fontSize": "0.82rem"},
                    ),
                    dbc.InputGroup([
                        dbc.Input(
                            id="chat-input",
                            placeholder="e.g. Which run had the highest temperature? "
                                        "Or: Run a 2D heat equation on a copper plate...",
                            type="text",
                            n_submit=0,
                            debounce=False,
                        ),
                        dbc.Button("Send", id="chat-send-btn", color="danger", n_clicks=0),
                    ], className="mt-2"),
                ], width=9),
            ]),
        ]),
        # ── Tab 6: Knowledge Graph ───────────────────────────────────────────
        dbc.Tab(label="🧠 Knowledge Graph", tab_id="tab-kg", children=[
            dcc.Store(id="kg-search-results", data=[]),
            dbc.Row([
                # ── Left: KG stats + semantic search ───────────────────────
                dbc.Col([
                    html.H5("Graph Statistics", className="mt-3 mb-2"),
                    html.Div(id="kg-stats-cards"),

                    html.Hr(style={"borderColor": "#2a2a3e", "margin": "12px 0"}),

                    html.H5("Semantic Run Search", className="mb-2"),
                    html.P(
                        "Describe a simulation in natural language — the KG will "
                        "find the most similar past runs using vector embeddings.",
                        className="text-muted mb-2", style={"fontSize": "0.8rem"},
                    ),
                    dbc.Textarea(
                        id="kg-search-input",
                        placeholder="e.g. 2D steel domain with convective cooling on one side, "
                                    "component-scale, medium conductor...",
                        rows=3,
                        style={"backgroundColor": "#1a1a2e", "color": TEXT_COLOR,
                               "borderColor": "#2e4a6a", "fontSize": "0.85rem"},
                    ),
                    dbc.Button(
                        "🔍 Find Similar Runs", id="kg-search-btn",
                        color="primary", className="mt-2 mb-3 w-100",
                    ),
                    html.Div(id="kg-search-results-panel"),

                    html.Hr(style={"borderColor": "#2a2a3e", "margin": "12px 0"}),

                    html.H5("Open in NeoDash", className="mb-1"),
                    html.P(
                        "NeoDash is an open-source graph exploration tool by Neo4j Labs "
                        "(Apache 2.0). Launch it to visualize and navigate the full "
                        "knowledge graph with interactive Cypher-powered dashboards.",
                        className="text-muted mb-2", style={"fontSize": "0.8rem"},
                    ),
                    html.A(
                        dbc.Button("Launch NeoDash 🚀", color="secondary",
                                   className="w-100"),
                        id="kg-neodash-btn",
                        href="#",
                        target="_blank",
                    ),
                    html.P(
                        "Note: NeoDash runs on port 9001 (reuses MinIO console slot). "
                        "Start with: docker compose up neodash",
                        className="text-muted mt-1",
                        style={"fontSize": "0.72rem", "opacity": "0.7"},
                    ),
                ], width=4, style={"borderRight": "1px solid #2a2a3e",
                                   "paddingRight": "16px"}),

                # ── Right: Upload + Reference browser ──────────────────────
                dbc.Col([
                    # ── Unified "Add Knowledge" panel ────────────────────────
                    dbc.Card([
                        dbc.CardHeader(
                            html.Span("➕ Add to Knowledge Graph",
                                      style={"fontWeight": "600", "fontSize": "0.95rem"}),
                            style={"backgroundColor": "#141428", "padding": "8px 14px"},
                        ),
                        dbc.CardBody([
                            dbc.Tabs([
                                # ── Tab 1: Upload file ─────────────────────
                                dbc.Tab(label="📎 Upload File", tab_id="add-kg-file", children=[
                                    html.Div([
                                        html.P(
                                            "Upload a PDF, TXT, or Markdown document. The text will be "
                                            "embedded and automatically linked to the most similar "
                                            "simulation runs in the knowledge graph.",
                                            className="text-muted mt-2 mb-2",
                                            style={"fontSize": "0.8rem"},
                                        ),
                                        # ── DOI quick-fill ─────────────────
                                        dbc.InputGroup([
                                            dbc.InputGroupText(
                                                "DOI", style={"backgroundColor": "#1a1a2e",
                                                              "color": "#90e0ef",
                                                              "borderColor": "#2e4a6a",
                                                              "fontSize": "0.8rem",
                                                              "fontWeight": "600"},
                                            ),
                                            dbc.Input(
                                                id="ref-doi-input",
                                                placeholder="10.xxxx/…  — paste a DOI and press Enter",
                                                type="text",
                                                debounce=False,
                                                style={"backgroundColor": "#1a1a2e",
                                                       "color": TEXT_COLOR,
                                                       "borderColor": "#2e4a6a",
                                                       "fontSize": "0.82rem"},
                                            ),
                                            dbc.Button(
                                                "Fetch", id="ref-doi-fetch-btn",
                                                color="info", size="sm", n_clicks=0,
                                                style={"fontSize": "0.8rem"},
                                            ),
                                        ], className="mb-1"),
                                        html.Div(id="ref-doi-status",
                                                 style={"fontSize": "0.77rem", "minHeight": "18px"},
                                                 className="mb-2 text-muted"),
                                        html.Hr(style={"borderColor": "#2a2a3e",
                                                        "margin": "8px 0 10px"}),
                                        dcc.Upload(
                                            id="ref-upload-file",
                                            children=html.Div([
                                                html.Span("📄 ", style={"fontSize": "1.4rem"}),
                                                html.Span("Drag & drop or "),
                                                html.A("browse", style={"color": ACCENT,
                                                                         "cursor": "pointer"}),
                                                html.Br(),
                                                html.Small("PDF · TXT · Markdown",
                                                           style={"color": "#888",
                                                                  "fontSize": "0.75rem"}),
                                            ]),
                                            style={
                                                "width": "100%", "height": "80px",
                                                "borderWidth": "1px", "borderStyle": "dashed",
                                                "borderRadius": "6px", "borderColor": "#2e4a6a",
                                                "textAlign": "center", "paddingTop": "16px",
                                                "backgroundColor": "#11112a", "cursor": "pointer",
                                            },
                                            multiple=False,
                                        ),
                                        html.Div(id="ref-upload-filename-label",
                                                 className="text-muted mt-1 mb-2",
                                                 style={"fontSize": "0.78rem"}),
                                        dbc.Row([
                                            dbc.Col(dbc.Input(
                                                id="ref-upload-title",
                                                placeholder="Title *  (required)",
                                                type="text",
                                                style={"backgroundColor": "#1a1a2e",
                                                       "color": TEXT_COLOR,
                                                       "borderColor": "#2e4a6a",
                                                       "fontSize": "0.82rem"},
                                            ), width=12, className="mb-2"),
                                            dbc.Col(dbc.Input(
                                                id="ref-upload-source",
                                                placeholder="Citation  (journal, year, pages…)",
                                                type="text",
                                                style={"backgroundColor": "#1a1a2e",
                                                       "color": TEXT_COLOR,
                                                       "borderColor": "#2e4a6a",
                                                       "fontSize": "0.82rem"},
                                            ), width=12, className="mb-2"),
                                            dbc.Col(dbc.Input(
                                                id="ref-upload-url",
                                                placeholder="URL / DOI  (optional)",
                                                type="text",
                                                style={"backgroundColor": "#1a1a2e",
                                                       "color": TEXT_COLOR,
                                                       "borderColor": "#2e4a6a",
                                                       "fontSize": "0.82rem"},
                                            ), width=12, className="mb-2"),
                                            dbc.Col(dbc.Input(
                                                id="ref-upload-subject",
                                                placeholder="Subject / keywords",
                                                type="text",
                                                style={"backgroundColor": "#1a1a2e",
                                                       "color": TEXT_COLOR,
                                                       "borderColor": "#2e4a6a",
                                                       "fontSize": "0.82rem"},
                                            ), width=8, className="mb-2"),
                                            dbc.Col(dbc.Select(
                                                id="ref-upload-type",
                                                options=[
                                                    {"label": "📄 Paper",    "value": "paper"},
                                                    {"label": "📋 Report",   "value": "report"},
                                                    {"label": "📚 Handbook", "value": "handbook"},
                                                    {"label": "🏛️ Standard", "value": "standard"},
                                                ],
                                                value="paper",
                                                style={"backgroundColor": "#1a1a2e",
                                                       "color": TEXT_COLOR,
                                                       "borderColor": "#2e4a6a",
                                                       "fontSize": "0.82rem"},
                                            ), width=4, className="mb-2"),
                                            dbc.Col(dbc.Input(
                                                id="ref-upload-run-ids",
                                                placeholder="Pin to run IDs  (comma-separated, optional)",
                                                type="text",
                                                style={"backgroundColor": "#1a1a2e",
                                                       "color": TEXT_COLOR,
                                                       "borderColor": "#2e4a6a",
                                                       "fontSize": "0.82rem"},
                                            ), width=9, className="mb-2"),
                                            dbc.Col(dbc.Input(
                                                id="ref-upload-top-k",
                                                placeholder="Auto-link top",
                                                type="number", min=0, max=50, value=10,
                                                style={"backgroundColor": "#1a1a2e",
                                                       "color": TEXT_COLOR,
                                                       "borderColor": "#2e4a6a",
                                                       "fontSize": "0.82rem"},
                                            ), width=3, className="mb-2"),
                                        ]),
                                        dbc.Button(
                                            "📤 Upload & Link to Knowledge Graph",
                                            id="ref-upload-submit-btn",
                                            color="success", className="w-100 mt-1",
                                            disabled=False,
                                        ),
                                        html.Div(id="ref-upload-status", className="mt-2"),
                                    ]),
                                ]),

                                # ── Tab 2: Fetch web resource ───────────────
                                dbc.Tab(label="🌐 Web Resource URL", tab_id="add-kg-web",
                                        children=[
                                    html.Div([
                                        html.P(
                                            "Paste a URL to a tutorial, ebook, or documentation site. "
                                            "Pages are crawled automatically, parsed with Docling, and "
                                            "each section is cross-referenced to simulation runs.",
                                            className="text-muted mt-2 mb-2",
                                            style={"fontSize": "0.8rem"},
                                        ),
                                        dbc.InputGroup([
                                            dbc.InputGroupText(
                                                "URL",
                                                style={"backgroundColor": "#1a1a2e",
                                                       "color": "#90e0ef",
                                                       "borderColor": "#2e4a6a",
                                                       "fontSize": "0.8rem",
                                                       "fontWeight": "600"},
                                            ),
                                            dbc.Input(
                                                id="web-fetch-url",
                                                placeholder="https://jsdokken.com/dolfinx-tutorial/  — press Enter or click Fetch",
                                                type="url",
                                                debounce=False,
                                                style={"backgroundColor": "#1a1a2e",
                                                       "color": TEXT_COLOR,
                                                       "borderColor": "#2e4a6a",
                                                       "fontSize": "0.82rem"},
                                            ),
                                            dbc.Button(
                                                "Fetch", id="web-fetch-btn",
                                                color="info", size="sm", n_clicks=0,
                                                style={"fontSize": "0.8rem"},
                                            ),
                                        ], className="mb-2"),
                                        dbc.Row([
                                            dbc.Col(dbc.Input(
                                                id="web-fetch-title",
                                                placeholder="Title  (e.g. FEniCSx Tutorial) — optional",
                                                type="text",
                                                style={"backgroundColor": "#1a1a2e",
                                                       "color": TEXT_COLOR,
                                                       "borderColor": "#2e4a6a",
                                                       "fontSize": "0.82rem"},
                                            ), width=7, className="mb-2"),
                                            dbc.Col(dbc.Input(
                                                id="web-fetch-subject",
                                                placeholder="Subject / keywords",
                                                type="text",
                                                style={"backgroundColor": "#1a1a2e",
                                                       "color": TEXT_COLOR,
                                                       "borderColor": "#2e4a6a",
                                                       "fontSize": "0.82rem"},
                                            ), width=3, className="mb-2"),
                                            dbc.Col(dbc.Input(
                                                id="web-fetch-max-pages",
                                                placeholder="Pages",
                                                type="number", min=1, max=100, value=50,
                                                style={"backgroundColor": "#1a1a2e",
                                                       "color": TEXT_COLOR,
                                                       "borderColor": "#2e4a6a",
                                                       "fontSize": "0.82rem"},
                                            ), width=2, className="mb-2"),
                                        ]),
                                        html.Div(id="web-fetch-status", className="mt-1"),
                                        html.P(
                                            "💡 Examples: FEniCSx tutorial · ASHRAE Handbook · "
                                            "FEniCS Book · OpenFOAM docs · any HTML-based ebook",
                                            className="text-muted mt-2 mb-0",
                                            style={"fontSize": "0.75rem"},
                                        ),
                                    ]),
                                ]),
                            ], id="add-kg-tabs", active_tab="add-kg-file",
                               style={"marginBottom": "4px"},
                            ),
                        ], style={"backgroundColor": "#0d0d1f", "padding": "8px 14px 14px"}),
                    ], style={"border": "1px solid #1e2a3a", "marginTop": "12px",
                              "marginBottom": "16px"}),

                    # ── Uploaded documents list ──────────────────────────────
                    dbc.Row([
                        dbc.Col(html.H6("📚 Uploaded Documents",
                                        className="mb-2",
                                        style={"color": ACCENT}), width=9),
                        dbc.Col(dbc.Button("↻", id="ref-docs-refresh-btn",
                                           color="secondary", size="sm",
                                           className="mb-2 w-100"), width=3),
                    ]),
                    html.Div(id="ref-uploaded-docs-list",
                             style={"maxHeight": "200px", "overflowY": "auto",
                                    "marginBottom": "12px"}),

                    html.Hr(style={"borderColor": "#2a2a3e", "margin": "8px 0"}),

                    dbc.Row([
                        dbc.Col(html.H5("Physics Reference Browser",
                                        className="mt-3 mb-2"), width=8),
                        dbc.Col(
                            dbc.Select(
                                id="kg-ref-type-filter",
                                options=[
                                    {"label": "All types", "value": "all"},
                                    {"label": "📐 Material Properties", "value": "material_property"},
                                    {"label": "🔲 BC Practice",          "value": "bc_practice"},
                                    {"label": "⚙️ Solver Guidance",      "value": "solver_guidance"},
                                    {"label": "🌐 Domain Physics",       "value": "domain_physics"},
                                ],
                                value="all",
                                style={"backgroundColor": "#1a1a2e", "color": TEXT_COLOR,
                                       "borderColor": "#2e4a6a", "fontSize": "0.82rem"},
                            ), width=4, className="mt-3",
                        ),
                    ]),
                    html.Div(id="kg-reference-browser",
                             style={"maxHeight": "400px", "overflowY": "auto"}),
                ], width=8),
            ], className="mt-2"),

            # ── Chunk-level search (full width row below) ─────────────────────
            html.Hr(style={"borderColor": "#2a2a3e", "margin": "16px 0 8px"}),
            dbc.Row([
                dbc.Col([
                    html.H6("🔬 Semantic Chunk Search",
                             style={"color": ACCENT, "marginBottom": "6px"}),
                    html.P(
                        "Search across all extracted document chunks. "
                        "Each chunk is independently embedded — this finds the most "
                        "relevant paragraphs, tables, and sections across all uploaded documents.",
                        className="text-muted", style={"fontSize": "0.78rem"},
                    ),
                    dbc.InputGroup([
                        dbc.Input(
                            id="chunk-search-input",
                            placeholder="e.g. thermal conductivity of steel at high temperature — press Enter or click Search",
                            type="text",
                            debounce=False,
                            style={"backgroundColor": "#1a1a2e", "color": TEXT_COLOR,
                                   "borderColor": "#2e4a6a", "fontSize": "0.82rem"},
                        ),
                        dbc.Button("🔍 Search Chunks", id="chunk-search-btn",
                                   color="primary", n_clicks=0),
                    ], className="mb-2"),
                ], width=12),
            ]),
            html.Div(id="chunk-search-results",
                     style={"maxHeight": "350px", "overflowY": "auto",
                            "marginBottom": "12px"}),
        ]),
        # ── Tab 7: Run Explorer ──────────────────────────────────────────────
        dbc.Tab(label="🔎 Run Explorer", tab_id="tab-explorer", children=[
            dcc.Store(id="explorer-selected-run", data=None),
            dbc.Row([
                # ── Left panel: run list ────────────────────────────────────
                dbc.Col([
                    html.H5("Run Explorer", className="mt-3"),
                    dbc.InputGroup([
                        dbc.Input(id="explorer-search-input",
                                  placeholder="Search run ID…",
                                  type="text", debounce=True),
                        dbc.Button("🔍", id="explorer-search-btn",
                                   color="secondary", n_clicks=0),
                    ], className="mb-2"),
                    dbc.Row([
                        dbc.Col(dbc.Select(
                            id="explorer-status-filter",
                            options=[
                                {"label": "All statuses", "value": ""},
                                {"label": "✅ Success",   "value": "success"},
                                {"label": "❌ Failed",    "value": "failed"},
                                {"label": "⏳ Running",  "value": "running"},
                            ],
                            value="",
                        ), width=6),
                        dbc.Col(dbc.Select(
                            id="explorer-dim-filter",
                            options=[
                                {"label": "All dims", "value": "0"},
                                {"label": "2D",       "value": "2"},
                                {"label": "3D",       "value": "3"},
                            ],
                            value="0",
                        ), width=6),
                    ], className="mb-2"),
                    dbc.Button("↺ Refresh", id="explorer-refresh-btn",
                               color="dark", outline=True, size="sm",
                               n_clicks=0, className="mb-2 w-100"),
                    html.Div(id="explorer-run-list",
                             style={"height": "72vh", "overflowY": "auto"}),
                ], width=3, style={"borderRight": "1px solid #333",
                                   "paddingRight": "12px"}),

                # ── Right panel: detail view ────────────────────────────────
                dbc.Col([
                    html.Div(id="explorer-detail-panel",
                             style={"height": "80vh", "overflowY": "auto"}),
                ], width=9),
            ], className="mt-2"),
        ]),
    ], id="main-tabs", active_tab="tab-overview"),

    html.Footer(
        "PDE Agents v1.0  |  FEniCSx + LangGraph + Ollama",
        style={"textAlign": "center", "color": "#666", "marginTop": "30px", "padding": "10px"},
    ),
], fluid=True, style={"backgroundColor": BG_COLOR, "minHeight": "100vh", "color": TEXT_COLOR})


# ─── Callbacks ────────────────────────────────────────────────────────────────


app.clientside_callback(
    """
    function(href) {
        var host = window.location.hostname;

        var docs    = '/agents/docs';
        var minio   = 'http://' + host + ':9002';  // MinIO console on 9002 (9001 is NeoDash)

        var boltUrl = encodeURIComponent('bolt://' + host + ':7687');
        var neo4j   = 'http://' + host + ':7474/browser/?connectURL=' + boltUrl;

        var neodash = 'http://' + host + ':9001';

        return [docs, minio, neo4j, neodash];
    }
    """,
    Output("nav-link-docs",    "href"),
    Output("nav-link-minio",   "href"),
    Output("nav-link-neo4j",   "href"),
    Output("nav-link-neodash", "href"),
    Input("url", "href"),
)


@app.callback(
    Output("run-selector",       "options"),
    Output("field-run-selector", "options"),
    Output("conv-run-selector",  "options"),
    Output("param-runs-selector","options"),
    Output("kpi-total-runs",     "children"),
    Output("kpi-success-runs",   "children"),
    Output("kpi-failed-runs",    "children"),
    Output("kpi-avg-time",       "children"),
    Output("kpi-max-time",       "children"),
    Input("refresh-interval",    "n_intervals"),
)
def refresh_run_list(n):
    run_ids = _get_run_ids()
    options = [{"label": rid, "value": rid} for rid in run_ids]

    # KPI values come from PostgreSQL (authoritative source).
    # result.json files only store the FEniCS solver outcome/time, not the
    # full orchestration status or wall time (which can be 100× larger).
    db = _get_db_kpi_stats()
    if db["total"]:
        total    = db["total"]
        success  = db["success"]
        failed   = db["failed"]
        avg_wall = db["avg_wall"]
        max_wall = db["max_wall"]
    else:
        # Fallback to file-based counts when DB is unreachable
        results  = [_load_result(rid) for rid in run_ids]
        total    = len(run_ids)
        success  = sum(1 for r in results if r.get("status") == "success")
        failed   = sum(1 for r in results if r.get("status") == "failed")
        times    = [r.get("wall_time", 0) for r in results if r.get("wall_time")]
        avg_wall = sum(times) / len(times) if times else 0.0
        max_wall = max(times) if times else 0.0

    avg_t = f"{avg_wall:.1f}" if avg_wall else "–"
    max_t = f"max {max_wall:.0f}s" if max_wall else ""

    return options, options, options, options, total, success, failed, avg_t, max_t


@app.callback(
    Output("status-pie-chart",    "figure"),
    Output("wall-time-bar-chart", "figure"),
    Input("refresh-interval",     "n_intervals"),
)
def update_overview_charts(n):
    run_ids = _get_run_ids()

    dofs_list, time_list, tmax_list, labels, dims, geo_types = [], [], [], [], [], []
    k_vals, tmax_all = [], []

    for rid in run_ids:
        r = _load_result(rid)
        c = _load_config(rid)
        if not r or not c:
            continue
        dofs = r.get("n_dofs")
        wt   = r.get("wall_time")
        tmax = r.get("max_temperature")
        dim  = c.get("dim", 2)
        k    = c.get("k", 1.0)
        geo  = (c.get("geometry") or {}).get("type", f"{dim}D-rect")

        if tmax is not None:
            tmax_all.append(tmax)
            k_vals.append(k)

        if dofs and wt:
            dofs_list.append(dofs)
            time_list.append(wt)
            labels.append(rid[-12:])
            dims.append(dim)
            geo_types.append(geo)

    # ── Chart 1: DOFs vs Wall Time (computational scaling) ────────────────────
    if dofs_list:
        color_map = {"2": "#00b4d8", "3": ACCENT}
        colors = [color_map.get(str(d), "#90e0ef") for d in dims]
        scatter_fig = go.Figure(go.Scatter(
            x=dofs_list, y=time_list,
            mode="markers+text",
            text=labels,
            textposition="top center",
            textfont=dict(size=8, color=TEXT_COLOR),
            marker=dict(
                color=colors,
                size=9,
                opacity=0.85,
                line=dict(color="#fff", width=0.5),
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "DOFs: %{x:,}<br>"
                "Wall time: %{y:.2f}s<br>"
                "<extra></extra>"
            ),
        ))
        # Reference O(N) trendline
        if len(dofs_list) > 2:
            d_arr = np.array(sorted(dofs_list))
            t_arr = np.array(sorted(time_list))
            slope = float(np.median(t_arr / np.maximum(d_arr, 1)))
            scatter_fig.add_trace(go.Scatter(
                x=d_arr.tolist(),
                y=(slope * d_arr).tolist(),
                mode="lines",
                line=dict(color="#555", dash="dot", width=1),
                name="O(N) ref",
                hoverinfo="skip",
            ))
        scatter_fig.update_layout(
            title="Solver Scaling: DOFs vs Wall Time",
            xaxis_title="DOFs", yaxis_title="Wall Time (s)",
            xaxis_type="log", yaxis_type="log",
            showlegend=False,
            **PLOT_LAYOUT,
        )
    else:
        scatter_fig = _make_empty_fig("No run data yet")
        scatter_fig.update_layout(title="Solver Scaling: DOFs vs Wall Time",
                                  **PLOT_LAYOUT)

    # ── Chart 2: T_max distribution per run (sorted by k) ────────────────────
    if tmax_all:
        sorted_pairs = sorted(zip(k_vals, tmax_all))
        k_sorted, t_sorted = zip(*sorted_pairs)
        tmax_fig = go.Figure(go.Bar(
            x=[f"k={k:.3g}" for k in k_sorted],
            y=list(t_sorted),
            marker=dict(
                color=list(t_sorted),
                colorscale="Plasma",
                showscale=True,
                colorbar=dict(
                    title="T_max (K)",
                    titlefont=dict(color=TEXT_COLOR),
                    tickfont=dict(color=TEXT_COLOR),
                    thickness=12,
                ),
            ),
            hovertemplate="k=%{x}<br>T_max=%{y:.1f} K<extra></extra>",
        ))
        tmax_fig.update_layout(
            title="Peak Temperature by Conductivity",
            xaxis_title="Thermal conductivity k",
            yaxis_title="T_max [K]",
            xaxis_tickangle=-35,
        **PLOT_LAYOUT,
    )
    else:
        tmax_fig = _make_empty_fig("No temperature data yet")
        tmax_fig.update_layout(title="Peak Temperature by Conductivity",
                               **PLOT_LAYOUT)

    return scatter_fig, tmax_fig


@app.callback(
    Output("run-detail-card", "children"),
    Input("run-selector", "value"),
)
def show_run_detail(run_id):
    if not run_id:
        return html.P("Pick a run above to inspect it.",
                      className="text-muted", style={"fontSize": "0.82rem"})

    result = _load_result(run_id)
    config = _load_config(run_id)
    if not result:
        return dbc.Alert(f"No result.json for {run_id}", color="warning", className="mt-2")

    status = result.get("status", "unknown")
    status_color = "success" if status == "success" else "danger"

    # Geometry string
    geo = config.get("geometry") or {}
    if geo:
        geo_str = geo.get("type", "gmsh")
        ms = geo.get("mesh_size")
        geo_str += f" (h={ms})" if ms else ""
    else:
        dim = config.get("dim", 2)
        geo_str = (f"{dim}D rect  "
                   f"{config.get('nx')}×{config.get('ny')}"
                   + (f"×{config.get('nz')}" if dim == 3 else ""))

    # BC types summary
    bcs = config.get("bcs", [])
    bc_summary = ", ".join(sorted({b.get("type", "?") for b in bcs})) or "–"

    # Material
    k   = config.get("k",   1.0)
    rho = config.get("rho", 1.0)
    cp  = config.get("cp",  1.0)
    alpha = k / (rho * cp) if rho * cp else 0.0

    tmax = result.get("max_temperature")
    tmin = result.get("min_temperature")
    tmean = result.get("mean_temperature")

    rows = [
        ("Status",    dbc.Badge(status.upper(), color=status_color, className="me-1")),
        ("Geometry",  geo_str),
        ("DOFs",      f"{result.get('n_dofs', 0):,}"),
        ("Steps",     f"{result.get('n_timesteps', '–')}  (dt={config.get('dt', '?')}s)"),
        ("Wall time", f"{result.get('wall_time', 0):.2f} s"),
        ("T range",   f"{tmin:.1f} – {tmax:.1f} K" if tmax is not None else "–"),
        ("T mean",    f"{tmean:.1f} K" if tmean is not None else "–"),
        ("k",         f"{k} W/(m·K)"),
        ("α = k/ρcₚ", f"{alpha:.3e} m²/s"),
        ("BCs",       bc_summary),
    ]

    td_label = {"color": "#888", "fontSize": "0.78rem", "paddingRight": "8px",
                "whiteSpace": "nowrap"}
    td_value = {"fontSize": "0.8rem"}

    # ── SIMILAR_TO neighbours from KG ────────────────────────────────────────
    similar_section = []
    try:
        kg = _get_kg()
        if kg:
            sim_rows = kg._run(
                """
                MATCH (src:Run {run_id: $run_id})-[rel:SIMILAR_TO]->(nb:Run)
                OPTIONAL MATCH (nb)-[:USES_BC_CONFIG]->(b:BCConfig)
                RETURN nb.run_id    AS run_id,
                       nb.k         AS k,
                       nb.t_max     AS t_max,
                       nb.n_dofs    AS n_dofs,
                       nb.wall_time AS wall_time,
                       b.pattern    AS bc_pattern,
                       rel.score    AS score
                ORDER BY rel.score DESC LIMIT 5
                """,
                run_id=run_id,
            )
            if sim_rows:
                sim_items = []
                for nb in sim_rows:
                    score_pct = int((nb.get("score") or 0) * 100)
                    sim_items.append(html.Tr([
                        html.Td(html.Span(nb["run_id"][:14],
                                          style={"fontFamily": "monospace",
                                                 "fontSize": "0.72rem"})),
                        html.Td(f"{score_pct}%",
                                style={"color": "#6fd672", "fontWeight": "600",
                                       "fontSize": "0.78rem"}),
                        html.Td(f"k={nb.get('k')}",
                                style={"fontSize": "0.72rem"}),
                        html.Td(nb.get("bc_pattern", "–"),
                                style={"fontSize": "0.7rem", "color": "#aaa"}),
                        html.Td(f"T={nb.get('t_max', 0):.0f}K" if nb.get("t_max") else "–",
                                style={"fontSize": "0.72rem"}),
                    ]))
                similar_section = [
                    html.Hr(style={"borderColor": "#2a2a3e", "margin": "6px 0"}),
                    html.P("🔗 Similar Runs (SIMILAR_TO edges)",
                           style={"fontSize": "0.72rem", "color": "#b48eff",
                                  "marginBottom": "4px", "fontWeight": "600"}),
                    dbc.Table(
                        [html.Thead(html.Tr([
                             html.Th("Run ID", style={"fontSize": "0.7rem"}),
                             html.Th("Score",  style={"fontSize": "0.7rem"}),
                             html.Th("k",      style={"fontSize": "0.7rem"}),
                             html.Th("BCs",    style={"fontSize": "0.7rem"}),
                             html.Th("T_max",  style={"fontSize": "0.7rem"}),
                         ])),
                         html.Tbody(sim_items)],
                        size="sm", bordered=False, dark=True,
                        style={"marginBottom": 0},
                    ),
                ]
    except Exception:
        pass

    return dbc.Card([
        dbc.CardHeader(
            html.Span(run_id, style={"fontSize": "0.82rem", "fontWeight": "600"}),
            style={"padding": "6px 10px"},
        ),
        dbc.CardBody([
            dbc.Table(
                html.Tbody([
                    html.Tr([html.Td(label, style=td_label),
                             html.Td(value, style=td_value)])
                    for label, value in rows
                ]),
                size="sm", bordered=False,
                style={"marginBottom": 0},
            ),
            *similar_section,
        ], style={"padding": "6px 10px"}),
    ], color="dark", outline=True, className="mt-2")


# ─── Recent runs table ────────────────────────────────────────────────────────

@app.callback(
    Output("recent-runs-table", "children"),
    Output("recent-runs-count", "children"),
    Input("refresh-interval",   "n_intervals"),
)
def refresh_recent_runs_table(n):
    run_ids = _get_run_ids()  # already sorted newest-first

    if not run_ids:
        return (
            html.P("No simulation runs found yet.", className="text-muted mt-2",
                   style={"fontSize": "0.85rem"}),
            "",
        )

    rows = []
    for rid in run_ids[:30]:
        r = _load_result(rid)
        c = _load_config(rid)
        if not r:
            continue

        status  = r.get("status", "unknown")
        s_color = "success" if status == "success" else ("danger" if status == "failed" else "secondary")

        geo     = c.get("geometry") or {}
        if geo:
            geo_str = geo.get("type", "gmsh")
        else:
            dim = c.get("dim", 2)
            geo_str = (f"{dim}D  {c.get('nx')}×{c.get('ny')}"
                       + (f"×{c.get('nz')}" if dim == 3 else ""))

        k_val   = c.get("k", "–")
        tmax    = r.get("max_temperature")
        dofs    = r.get("n_dofs", 0)
        wt      = r.get("wall_time", 0)

        rows.append(html.Tr([
            html.Td(rid[-16:], style={"fontSize": "0.76rem", "fontFamily": "monospace",
                                      "color": "#90e0ef", "whiteSpace": "nowrap"}),
            html.Td(geo_str,   style={"fontSize": "0.76rem"}),
            html.Td(f"{k_val} W/(m·K)", style={"fontSize": "0.76rem"}),
            html.Td(f"{tmax:.0f} K" if tmax is not None else "–",
                    style={"fontSize": "0.76rem"}),
            html.Td(f"{dofs:,}", style={"fontSize": "0.76rem", "textAlign": "right"}),
            html.Td(f"{wt:.1f}s", style={"fontSize": "0.76rem", "textAlign": "right"}),
            html.Td(dbc.Badge(status, color=s_color, pill=True,
                              style={"fontSize": "0.68rem"})),
        ]))

    if not rows:
        return html.P("No result data found.", className="text-muted"), ""

    th_style = {"fontSize": "0.74rem", "color": "#aaa", "fontWeight": "500",
                "borderBottom": "1px solid #2a2a3e", "padding": "4px 8px"}
    td_style = {"padding": "3px 8px"}

    table = dbc.Table([
        html.Thead(html.Tr([
            html.Th("Run ID",       style=th_style),
            html.Th("Geometry",     style=th_style),
            html.Th("k",            style=th_style),
            html.Th("T_max",        style=th_style),
            html.Th("DOFs",         style={**th_style, "textAlign": "right"}),
            html.Th("Wall Time",    style={**th_style, "textAlign": "right"}),
            html.Th("Status",       style=th_style),
        ])),
        html.Tbody(rows, style=td_style),
    ], size="sm", bordered=False, hover=True,
       style={"marginBottom": 0, "width": "100%"})

    count_label = f"showing {len(rows)} of {len(run_ids)}"
    return table, count_label


# ─── System health panel ──────────────────────────────────────────────────────

def _check_service(name: str, check_fn) -> html.Div:
    """Run a health check and return a coloured badge row."""
    try:
        ok, detail = check_fn()
    except Exception as exc:
        ok, detail = False, str(exc)[:50]
    color  = "#2dc653" if ok else "#e94560"
    label  = "online" if ok else "offline"
    return html.Div([
        html.Span("●", style={"color": color, "fontSize": "0.95rem",
                               "marginRight": "6px"}),
        html.Span(name, style={"fontSize": "0.8rem", "fontWeight": "500"}),
        html.Span(f" — {detail}", style={"fontSize": "0.74rem", "color": "#888"}),
    ], className="mb-1")


@app.callback(
    Output("system-health-panel", "children"),
    Input("refresh-interval",     "n_intervals"),
)
def refresh_system_health(n):
    OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    SIM_MODEL  = os.getenv("SIMULATION_AGENT_MODEL", "qwen2.5-coder:32b")
    DB_MODEL   = os.getenv("DATABASE_AGENT_MODEL",   "qwen2.5-coder:14b")

    def check_postgres():
        import psycopg2
        conn = psycopg2.connect(
            host=POSTGRES_HOST, port=POSTGRES_PORT,
            dbname=POSTGRES_DB, user=POSTGRES_USER, password=POSTGRES_PASS,
            connect_timeout=2,
        )
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM simulation_runs")
            count = cur.fetchone()[0]
        conn.close()
        return True, f"{count} runs"

    def check_neo4j():
        import requests as _req
        r = _req.get("http://neo4j:7474/", timeout=2)
        return r.status_code < 400, f"HTTP {r.status_code}"

    def check_agents_api():
        import requests as _req
        r = _req.get(f"{AGENTS_API}/health", timeout=2)
        return r.status_code < 400, "responding"

    def check_ollama():
        import requests as _req
        r = _req.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        data = r.json()
        n_models = len(data.get("models", []))
        return True, f"{n_models} model(s) loaded"

    items = [
        _check_service("PostgreSQL",  check_postgres),
        _check_service("Neo4j",       check_neo4j),
        _check_service("Agents API",  check_agents_api),
        _check_service("Ollama",      check_ollama),
    ]

    model_info = html.Div([
        html.Hr(style={"borderColor": "#2a2a3e", "margin": "8px 0"}),
        html.Div([
            html.Span("SIM", style={"fontSize": "0.68rem", "color": "#00b4d8",
                                    "fontWeight": "700", "marginRight": "4px"}),
            html.Span(SIM_MODEL, style={"fontSize": "0.74rem", "fontFamily": "monospace"}),
        ], className="mb-1"),
        html.Div([
            html.Span("DB", style={"fontSize": "0.68rem", "color": "#a8dadc",
                                   "fontWeight": "700", "marginRight": "4px"}),
            html.Span(DB_MODEL, style={"fontSize": "0.74rem", "fontFamily": "monospace"}),
        ]),
    ])

    return html.Div(items + [model_info])


# ─── Knowledge Graph tab callbacks ───────────────────────────────────────────

def _get_kg():
    """Lazy-load the KG singleton for dashboard callbacks."""
    try:
        import sys
        sys.path.insert(0, "/app")
        from knowledge_graph.graph import get_kg
        kg = get_kg()
        return kg if kg.available else None
    except Exception:
        return None


# ── Reference upload callbacks ────────────────────────────────────────────────

@app.callback(
    Output("ref-upload-filename-label", "children"),
    Input("ref-upload-file", "filename"),
    prevent_initial_call=True,
)
def show_uploaded_filename(filename):
    if not filename:
        return ""
    return f"📎 {filename}"


@app.callback(
    Output("ref-upload-status",      "children"),
    Output("ref-uploaded-docs-list", "children"),
    Input("ref-upload-submit-btn",   "n_clicks"),
    Input("ref-docs-refresh-btn",    "n_clicks"),
    State("ref-upload-file",         "contents"),
    State("ref-upload-file",         "filename"),
    State("ref-upload-title",        "value"),
    State("ref-upload-source",       "value"),
    State("ref-upload-url",          "value"),
    State("ref-upload-subject",      "value"),
    State("ref-upload-type",         "value"),
    State("ref-upload-run-ids",      "value"),
    State("ref-upload-top-k",        "value"),
    prevent_initial_call=True,
)
def handle_reference_upload(
    upload_clicks, refresh_clicks,
    file_contents, file_name,
    title, source, url, subject, ref_type, run_ids, top_k,
):
    import base64
    import io
    import requests as _req
    from dash import ctx

    # ── Refresh-only: just reload the list ─────────────────────────────────
    docs_panel = _build_uploaded_docs_list()
    if ctx.triggered_id == "ref-docs-refresh-btn":
        return no_update, docs_panel

    # ── Upload triggered ────────────────────────────────────────────────────
    if not file_contents:
        return (
            dbc.Alert("Please select a file first.", color="warning",
                      className="py-2", style={"fontSize": "0.82rem"}),
            docs_panel,
        )
    if not title or not title.strip():
        return (
            dbc.Alert("Title is required.", color="warning",
                      className="py-2", style={"fontSize": "0.82rem"}),
            docs_panel,
        )

    try:
        # Decode base64 file content from Dash
        _ctype, b64 = file_contents.split(",", 1)
        file_bytes = base64.b64decode(b64)

        agents_url = os.environ.get("AGENTS_API_URL", "http://agents:8000")
        resp = _req.post(
            f"{agents_url}/references/upload",
            files={"file": (file_name or "document", io.BytesIO(file_bytes),
                            "application/octet-stream")},
            data={
                "title":           title.strip(),
                "source":          source or "",
                "url":             url or "",
                "subject":         subject or "",
                "ref_type":        ref_type or "paper",
                "run_ids":         run_ids or "",
                "auto_link_top_k": int(top_k) if top_k else 10,
            },
            timeout=180,
        )
        resp.raise_for_status()
        result = resp.json()

        n_linked  = result.get("runs_linked", 0)
        method    = result.get("link_method", "none")
        embedded  = result.get("embedded", False)
        method_label = {"manual": "manually pinned", "auto": "auto-linked by similarity",
                        "none": "no runs linked"}.get(method, method)

        status_el = dbc.Alert(
            [
                html.Strong(f"✅  '{title}' uploaded successfully."),
                html.Br(),
                html.Span(
                    f"{n_linked} run(s) {method_label}. "
                    f"{'Embedding stored ✓' if embedded else 'Embedding not available'}",
                    style={"fontSize": "0.8rem"},
                ),
            ],
            color="success", className="py-2 mt-1",
            style={"fontSize": "0.85rem"},
        )
        return status_el, _build_uploaded_docs_list()

    except Exception as exc:
        return (
            dbc.Alert(f"Upload failed: {exc}", color="danger",
                      className="py-2", style={"fontSize": "0.82rem"}),
            docs_panel,
        )


@app.callback(
    Output("ref-upload-title",   "value"),
    Output("ref-upload-source",  "value"),
    Output("ref-upload-url",     "value"),
    Output("ref-upload-subject", "value"),
    Output("ref-upload-type",    "value"),
    Output("ref-doi-status",     "children"),
    Input("ref-doi-fetch-btn", "n_clicks"),
    Input("ref-doi-input",     "n_submit"),
    State("ref-doi-input",     "value"),
    prevent_initial_call=True,
)
def fetch_doi_metadata(n_clicks, n_submit, doi):
    """Look up a DOI via CrossRef and auto-fill the upload form fields."""
    import requests as _req
    from dash.exceptions import PreventUpdate

    if not doi or not doi.strip():
        raise PreventUpdate

    doi = doi.strip().lstrip("https://doi.org/").lstrip("http://doi.org/")
    try:
        resp = _req.get(
            f"https://api.crossref.org/works/{doi}",
            headers={"User-Agent": "pde-agents/2.0 (mailto:admin@localhost)"},
            timeout=12,
        )
        if resp.status_code == 404:
            return (no_update, no_update, no_update, no_update, no_update,
                    "❌ DOI not found in CrossRef.")
        resp.raise_for_status()
        msg = resp.json().get("message", {})
    except Exception as exc:
        return (no_update, no_update, no_update, no_update, no_update,
                f"❌ Lookup failed: {exc}")

    # ── Extract fields ──────────────────────────────────────────────────────
    titles   = msg.get("title") or []
    title    = titles[0] if titles else ""

    container = msg.get("container-title") or []
    journal   = container[0] if container else ""

    authors = msg.get("author") or []
    author_str = "; ".join(
        f"{a.get('family', '')} {a.get('given', '')}".strip() for a in authors[:4]
    )
    if len(authors) > 4:
        author_str += " et al."

    year_parts = (msg.get("published") or msg.get("issued") or {}).get("date-parts", [])
    year = str(year_parts[0][0]) if year_parts and year_parts[0] else ""

    source   = f"{author_str}. {journal}. {year}".strip(". ")
    doc_url  = msg.get("URL") or f"https://doi.org/{doi}"

    # ── Guess ref type ──────────────────────────────────────────────────────
    rtype_raw = (msg.get("type") or "").lower()
    ref_type_map = {
        "journal-article": "paper",
        "proceedings-article": "paper",
        "book-chapter": "handbook",
        "book": "handbook",
        "report": "report",
        "standard": "standard",
    }
    ref_type = ref_type_map.get(rtype_raw, "paper")

    subject_list = msg.get("subject") or []
    subject = ", ".join(subject_list[:5])

    status_msg = f"✅ Fetched: {journal or rtype_raw} ({year})"
    return title, source, doc_url, subject, ref_type, status_msg


def _build_uploaded_docs_list():
    """Fetch uploaded documents with processing status and chunk details."""
    import requests as _req
    agents_url = os.environ.get("AGENTS_API_URL", "http://agents:8000")

    try:
        resp = _req.get(f"{agents_url}/references/uploaded?limit=30", timeout=10)
        resp.raise_for_status()
        docs = resp.json()
    except Exception:
        docs = []

    if not docs:
        return html.P("No uploaded documents yet.",
                      className="text-muted", style={"fontSize": "0.8rem"})

    rows = []
    for d in docs:
        ref_id     = d.get("ref_id", "")
        title      = d.get("title") or d.get("subject") or ref_id or "—"
        n_runs     = d.get("linked_runs", 0)
        ref_type   = d.get("type", "uploaded")
        uploaded   = (d.get("uploaded_at") or "")[:10]
        doc_url    = d.get("url") or ""

        type_badge_color = {
            "paper": "primary", "report": "info",
            "handbook": "warning", "standard": "secondary",
        }.get(ref_type, "light")

        # Fetch processing status for this ref
        proc_status = ""
        n_chunks = 0
        n_xrefs  = 0
        try:
            sr = _req.get(f"{agents_url}/references/{ref_id}/status", timeout=5)
            if sr.status_code == 200:
                sd = sr.json()
                proc_status = sd.get("status") or ""
                n_chunks    = sd.get("chunks_stored") or 0
                n_xrefs     = sd.get("cross_refs") or 0
        except Exception:
            pass

        status_badge_map = {
            "completed": ("success", "✓ processed"),
            "embedding": ("warning", "⏳ embedding…"),
            "queued":    ("info",    "⏳ queued"),
            "failed":    ("danger",  "✗ failed"),
        }
        sb_color, sb_text = status_badge_map.get(proc_status, ("secondary", proc_status or "pending"))

        title_el = (
            html.A(title, href=doc_url, target="_blank",
                   style={"color": ACCENT, "fontSize": "0.8rem",
                          "textDecoration": "none"})
            if doc_url else html.Span(title, style={"fontSize": "0.8rem"})
        )

        chunks_detail = ""
        if n_chunks:
            chunks_detail = f"📑 {n_chunks} chunks · 🔗 {n_xrefs} cross-refs"

        rows.append(dbc.ListGroupItem([
            dbc.Row([
                dbc.Col([
                    title_el,
                    html.Br(),
                    html.Small(chunks_detail or uploaded,
                               className="text-muted",
                               style={"fontSize": "0.73rem"}),
                ], width=5),
                dbc.Col(
                    dbc.Badge(ref_type, color=type_badge_color, pill=True,
                              style={"fontSize": "0.68rem"}),
                    width=2, className="d-flex align-items-center",
                ),
                dbc.Col(
                    html.Small(f"🔗 {n_runs} runs",
                               style={"fontSize": "0.73rem", "color": "#90e0ef"}),
                    width=2,
                ),
                dbc.Col(
                    dbc.Badge(sb_text, color=sb_color, pill=True,
                              style={"fontSize": "0.65rem"}),
                    width=3, className="d-flex align-items-center justify-content-end",
                ),
            ], align="center"),
        ], style={"backgroundColor": "#11112a", "borderColor": "#1e2a3a",
                  "padding": "6px 10px", "cursor": "default"})
        )

    return dbc.ListGroup(rows, flush=True)


@app.callback(
    Output("web-fetch-status", "children"),
    Input("web-fetch-btn", "n_clicks"),
    Input("web-fetch-url", "n_submit"),
    State("web-fetch-url",     "value"),
    State("web-fetch-title",   "value"),
    State("web-fetch-subject", "value"),
    State("web-fetch-max-pages", "value"),
    prevent_initial_call=True,
)
def handle_web_fetch(n_clicks, n_submit, url, title, subject, max_pages):
    """Queue a web resource for crawling and indexing."""
    import requests as _req

    if not url or not url.strip().startswith("http"):
        return dbc.Alert("Please enter a valid URL.", color="warning",
                        style={"fontSize": "0.82rem"})

    try:
        agents_url = os.environ.get("AGENTS_API_URL", "http://agents:8000")
        resp = _req.post(
            f"{agents_url}/references/fetch-url",
            json={
                "url": url.strip(),
                "title": (title or "").strip() or url.strip(),
                "subject": (subject or "").strip(),
                "max_pages": int(max_pages) if max_pages else 50,
                "ref_type": "web_resource",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        return dbc.Alert([
            html.Strong(f"✅ Queued: {data.get('title', url)}"),
            html.Br(),
            html.Span(
                f"Crawling up to {data.get('max_pages', '?')} pages. "
                f"Processing: {data.get('processing', '?')}. "
                f"Refresh the docs list to track progress.",
                style={"fontSize": "0.78rem"},
            ),
        ], color="success", className="py-2 mt-1",
            style={"fontSize": "0.85rem"})

    except Exception as exc:
        return dbc.Alert(f"Failed: {exc}", color="danger",
                        style={"fontSize": "0.82rem"})


@app.callback(
    Output("chunk-search-results", "children"),
    Input("chunk-search-btn",   "n_clicks"),
    Input("chunk-search-input", "n_submit"),
    State("chunk-search-input", "value"),
    prevent_initial_call=True,
)
def search_chunks(n_clicks, n_submit, query):
    """Semantic search across all document chunks via the agents API."""
    import requests as _req
    from dash.exceptions import PreventUpdate
    if not query or not query.strip():
        return html.P("Enter a search query above.", className="text-muted",
                      style={"fontSize": "0.8rem"})
    try:
        agents_url = os.environ.get("AGENTS_API_URL", "http://agents:8000")
        resp = _req.get(
            f"{agents_url}/references/search-chunks",
            params={"query": query.strip(), "top_k": 8},
            timeout=30,
        )
        resp.raise_for_status()
        results = resp.json()
    except Exception as exc:
        return dbc.Alert(f"Search failed: {exc}", color="danger",
                        style={"fontSize": "0.82rem"})

    if not results:
        return html.P("No matching chunks found.", className="text-muted",
                      style={"fontSize": "0.8rem"})

    cls_colors = {
        "material": "#4ecdc4", "bc": "#ff6b6b",
        "solver": "#45b7d1", "domain": "#96ceb4", "general": "#888",
    }

    def _resolve_href(url: str) -> str:
        if url.startswith("10.") or ("doi.org" in url and not url.startswith("http")):
            return f"https://doi.org/{url.lstrip('/')}"
        return url

    cards = []
    for r in results:
        cls          = r.get("classification", "general")
        chunk_type   = r.get("chunk_type", "text")
        heading      = r.get("heading") or ""
        text         = (r.get("text") or "")[:400]
        score        = r.get("score", 0)
        ref_title    = r.get("ref_title") or r.get("ref_id", "")
        parent_title = r.get("parent_title") or ref_title
        page         = r.get("page", 0)
        ref_url      = (r.get("ref_url") or "").strip()
        is_web       = bool(r.get("is_web")) or r.get("ref_type") in ("web_resource", "web_page")

        cls_color = cls_colors.get(cls, "#888")
        href      = _resolve_href(ref_url) if ref_url else None

        # ── Header: clickable title when URL exists ──────────────────
        display_title = parent_title if parent_title and parent_title != ref_title else ref_title
        if href:
            title_node = html.A(
                display_title,
                href=href,
                target="_blank",
                rel="noopener noreferrer",
                style={
                    "fontSize": "0.78rem",
                    "fontWeight": "600",
                    "color": "#90e0ef",
                    "textDecoration": "underline",
                    "textUnderlineOffset": "3px",
                    "cursor": "pointer",
                },
            )
        else:
            title_node = html.Span(
                display_title,
                style={"fontSize": "0.78rem", "fontWeight": "600", "color": "#b0b0c0"},
            )

        header_children = []
        if heading and heading != display_title:
            header_children.append(
                html.Span(heading + "  ", style={"fontSize": "0.8rem", "color": "#e0e0e0"})
            )
        header_children.append(title_node)

        # ── Footer: badges + optional open-link button ───────────────
        footer_children = [
            dbc.Badge(cls, style={"backgroundColor": cls_color,
                                   "fontSize": "0.65rem", "marginRight": "4px"}),
            dbc.Badge(chunk_type, color="dark",
                      style={"fontSize": "0.65rem", "marginRight": "8px"}),
            html.Small(f"score: {score:.3f}",
                       style={"fontSize": "0.7rem", "color": "#90e0ef",
                              "marginRight": "10px"}),
        ]
        if page and int(page) > 0:
            footer_children.append(
                html.Small(f"p. {page}",
                           style={"fontSize": "0.7rem", "color": "#888",
                                  "marginRight": "10px"})
            )
        if href:
            icon  = "🌐" if is_web else "📄"
            label = "Open page" if is_web else "View source"
            footer_children.append(
                html.A(
                    [icon, f" {label}"],
                    href=href,
                    target="_blank",
                    rel="noopener noreferrer",
                    style={
                        "fontSize": "0.72rem",
                        "color": "#90e0ef",
                        "textDecoration": "none",
                        "border": "1px solid #2e4a6a",
                        "borderRadius": "4px",
                        "padding": "1px 7px",
                        "cursor": "pointer",
                    },
                )
            )
        else:
            footer_children.append(
                dbc.Badge("no source URL", color="secondary",
                          style={"fontSize": "0.62rem", "opacity": "0.5"}),
            )

        cards.append(dbc.Card([
            dbc.CardBody([
                html.Div(header_children, className="mb-1"),
                html.P(text, style={"fontSize": "0.78rem", "color": "#ccc",
                                    "marginBottom": "6px", "lineHeight": "1.4"}),
                html.Div(footer_children,
                         style={"display": "flex", "alignItems": "center",
                                "flexWrap": "wrap", "gap": "4px"}),
            ], style={"padding": "8px 12px"}),
        ], style={"backgroundColor": "#11112a", "border": "1px solid #1e2a3a",
                  "marginBottom": "6px"}))

    return html.Div(cards)


@app.callback(
    Output("kg-stats-cards",   "children"),
    Output("kg-neodash-btn",   "href"),
    Input("refresh-interval",  "n_intervals"),
    Input("url",               "href"),
)
def refresh_kg_stats(n, href):
    """Populate KG stats cards and set NeoDash href dynamically."""
    # Derive host from the dashboard's own URL for NeoDash link
    neodash_href = "#"
    if href:
        try:
            from urllib.parse import urlparse
            host = urlparse(href).hostname or "localhost"
            neodash_href = f"http://{host}:9001"
        except Exception:
            pass

    kg = _get_kg()
    if not kg:
        return html.P("Knowledge graph unavailable", className="text-danger"), neodash_href

    s = kg.stats()

    # Count SIMILAR_TO edges
    try:
        edge_rows = kg._run("MATCH ()-[r:SIMILAR_TO]->() RETURN count(r) AS n")
        n_edges = edge_rows[0]["n"] if edge_rows else 0
    except Exception:
        n_edges = 0

    pct_embedded = (
        round(s.get("embedded_runs", 0) / max(s.get("total_runs", 1), 1) * 100)
        if s.get("total_runs") else 0
    )

    def stat_card(value, label, color, sublabel=""):
        return dbc.Card(dbc.CardBody([
            html.H3(str(value), className=f"text-{color} mb-0",
                    style={"fontWeight": "700"}),
            html.P(label, className="text-muted mb-0",
                   style={"fontSize": "0.78rem"}),
            html.Small(sublabel, className="text-muted",
                       style={"fontSize": "0.68rem", "opacity": "0.7"}) if sublabel else None,
        ]), color="dark", outline=True, className="mb-2")

    cards = [
        stat_card(s.get("total_runs", 0),      "Total Run Nodes",       "info"),
        stat_card(
            f"{s.get('embedded_runs', 0)} / {s.get('total_runs', 0)}",
            "Runs with Embeddings",  "success",
            f"{pct_embedded}% coverage · nomic-embed-text 768-dim",
        ),
        stat_card(n_edges,                      "SIMILAR_TO Edges",      "warning",
                  "top-5 KNN · cosine ≥ 0.85"),
        stat_card(s.get("references", 0),       "Reference Nodes",       "secondary",
                  "curated physics knowledge"),
        stat_card(s.get("materials", 0),        "Material Nodes",        "light"),
        stat_card(s.get("bc_configs", 0),       "BCConfig Nodes",        "light"),
        stat_card(s.get("domains", 0),          "Domain Nodes",          "light"),
        stat_card(s.get("thermal_classes", 0),  "ThermalClass Nodes",    "light"),
    ]
    return html.Div(cards), neodash_href


@app.callback(
    Output("kg-search-results-panel", "children"),
    Input("kg-search-btn",            "n_clicks"),
    State("kg-search-input",          "value"),
    prevent_initial_call=True,
)
def kg_semantic_search(n_clicks, query_text):
    """Run semantic similarity search from free-text query via embeddings."""
    if not query_text or not query_text.strip():
        return html.P("Enter a description above and click Search.",
                      className="text-muted", style={"fontSize": "0.82rem"})

    kg = _get_kg()
    if not kg:
        return html.P("Knowledge graph unavailable.", className="text-danger")

    try:
        from knowledge_graph.embeddings import get_embedder
        emb = get_embedder()
        vec = emb.embed_text(query_text.strip())
        if not vec:
            return html.P("Embedding unavailable (is nomic-embed-text pulled?)",
                          className="text-warning")

        rows = kg._run(
            """
            CALL db.index.vector.queryNodes('run_embedding_index', 8, $vec)
            YIELD node AS r, score
            WHERE r.status = 'success'
            OPTIONAL MATCH (r)-[:USES_BC_CONFIG]->(b:BCConfig)
            OPTIONAL MATCH (r)-[:ON_DOMAIN]->(d:Domain)
            OPTIONAL MATCH (r)-[:USES_MATERIAL]->(m:Material)
            RETURN r.run_id    AS run_id,
                   r.k         AS k,
                   r.dim       AS dim,
                   r.t_max     AS t_max,
                   r.wall_time AS wall_time,
                   r.n_dofs    AS n_dofs,
                   b.pattern   AS bc_pattern,
                   d.label     AS domain_label,
                   m.name      AS material,
                   round(score, 4) AS score
            ORDER BY score DESC LIMIT 5
            """,
            vec=vec,
        )
    except Exception as exc:
        return html.P(f"Search failed: {exc}", className="text-danger",
                      style={"fontSize": "0.8rem"})

    if not rows:
        return html.P("No similar runs found.", className="text-muted")

    items = []
    for r in rows:
        score    = r.get("score", 0.0)
        score_pct = int(score * 100)
        bar_color = "success" if score_pct >= 95 else "info" if score_pct >= 88 else "warning"
        items.append(dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Span(r.get("run_id", "")[:16],
                              style={"fontFamily": "monospace", "fontWeight": "700",
                                     "fontSize": "0.8rem", "color": "#00b4d8"}),
                    html.Br(),
                    html.Small(
                        f"k={r.get('k')} · {r.get('dim')}D · {r.get('bc_pattern','')} · "
                        f"{r.get('domain_label','')} · {r.get('material','')}",
                        className="text-muted",
                    ),
                ], width=9),
                dbc.Col([
                    html.Div(f"{score_pct}%",
                             style={"fontWeight": "700", "fontSize": "1.0rem",
                                    "color": "#6fd672", "textAlign": "right"}),
                    html.Small("similarity", className="text-muted",
                               style={"fontSize": "0.68rem", "float": "right"}),
                ], width=3),
            ]),
            dbc.Progress(value=score_pct, color=bar_color, className="mt-1",
                         style={"height": "4px"}),
            html.Small(
                f"T_max={r.get('t_max', 0):.0f}K  "
                f"DOFs={r.get('n_dofs', '?')}  "
                f"wall={r.get('wall_time', 0):.2f}s",
                className="text-muted",
                style={"fontSize": "0.72rem"},
            ),
        ]), color="dark", outline=True, className="mb-1"))

    return html.Div([
        html.P(f"Top {len(rows)} semantically similar runs:",
               className="text-muted mb-1", style={"fontSize": "0.78rem"}),
        html.Div(items),
    ])


@app.callback(
    Output("kg-reference-browser", "children"),
    Input("kg-ref-type-filter",    "value"),
    Input("refresh-interval",      "n_intervals"),
)
def refresh_reference_browser(ref_type, n):
    """Display all Reference nodes, filtered by type."""
    kg = _get_kg()
    if not kg:
        return html.P("Knowledge graph unavailable.", className="text-danger mt-3")

    cypher = """
        MATCH (ref:Reference)
        WHERE $type = 'all' OR ref.type = $type
        OPTIONAL MATCH (m:Material)-[:HAS_REFERENCE]->(ref)
        OPTIONAL MATCH (b:BCConfig)-[:HAS_REFERENCE]->(ref)
        OPTIONAL MATCH (d:Domain)-[:HAS_REFERENCE]->(ref)
        RETURN ref.ref_id  AS ref_id,
               ref.type    AS type,
               ref.subject AS subject,
               ref.text    AS text,
               ref.source  AS source,
               ref.url     AS url,
               ref.tags    AS tags,
               collect(DISTINCT m.name)    AS materials,
               collect(DISTINCT b.pattern) AS bc_patterns,
               collect(DISTINCT d.label)   AS domains
        ORDER BY ref.type, ref.subject
    """
    try:
        rows = kg._run(cypher, type=ref_type)
    except Exception as exc:
        return html.P(f"Query failed: {exc}", className="text-danger")

    if not rows:
        return html.P("No references found.", className="text-muted mt-3")

    TYPE_COLORS = {
        "material_property": ("#1e2a3a", "#00b4d8", "📐"),
        "bc_practice":       ("#1e2a1e", "#6fd672", "🔲"),
        "solver_guidance":   ("#2a1e1e", "#ff8c69", "⚙️"),
        "domain_physics":    ("#1e1a2e", "#b48eff", "🌐"),
    }

    cards = []
    for r in rows:
        bg, color, icon = TYPE_COLORS.get(r["type"], ("#1a1a2e", "#fff", "📌"))

        linked_to = []
        if r.get("materials"):  linked_to.append("Materials: " + ", ".join(r["materials"]))
        if r.get("bc_patterns"):linked_to.append("BCs: " + ", ".join(r["bc_patterns"]))
        if r.get("domains"):    linked_to.append("Domains: " + ", ".join(r["domains"]))
        linked_str = " · ".join(linked_to) if linked_to else "General (no direct links)"

        tags = r.get("tags") or []

        cards.append(dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col(
                    dbc.Badge(f"{icon} {r['type'].replace('_', ' ').title()}",
                              color="dark",
                              style={"backgroundColor": bg, "color": color,
                                     "border": f"1px solid {color}",
                                     "fontSize": "0.68rem"}),
                    width="auto",
                ),
                dbc.Col(
                    html.Small(linked_str, className="text-muted",
                               style={"fontSize": "0.7rem"}),
                    className="d-flex align-items-center",
                ),
            ], className="mb-1"),
            html.Strong(r["subject"], style={"fontSize": "0.85rem", "color": color}),
            html.P(r["text"], className="mb-1 mt-1",
                   style={"fontSize": "0.8rem", "color": "#c8d8e8", "lineHeight": "1.4"}),
            html.Small([
                "📚 ",
                html.A(
                    r["source"],
                    href=r.get("url") or "#",
                    target="_blank",
                    rel="noopener noreferrer",
                    style={
                        "color": "#8ab4c8",
                        "textDecoration": "underline",
                        "cursor": "pointer",
                    },
                ) if r.get("url") else r["source"],
            ], className="text-muted",
               style={"fontSize": "0.7rem", "fontStyle": "italic"}),
            html.Div([
                dbc.Badge(tag, color="secondary", className="me-1",
                          style={"fontSize": "0.62rem", "opacity": "0.7"})
                for tag in (tags[:6] if tags else [])
            ], className="mt-1"),
        ]), color="dark", outline=True, className="mb-2",
            style={"borderColor": color + "44"}))

    return html.Div(cards)


@app.callback(
    Output("field-main-plot",      "figure"),
    Output("field-secondary-plot", "figure"),
    Output("field-info",           "children"),
    Output("field-stats-card",     "children"),
    Output("field-slice-panel",    "style"),
    Output("field-anim-panel",     "style"),
    Output("field-step-slider",    "max"),
    Output("field-step-slider",    "marks"),
    Input("field-run-selector",    "value"),
    Input("field-view-mode",       "value"),
    Input("z-slice-slider",        "value"),
    Input("field-step-slider",     "value"),
    Input("field-anim-store",      "data"),
)
def update_field_view(run_id, view, z_frac, step_idx, anim_store):
    empty_fig  = _make_empty_fig("Select a run and view mode")
    empty_sec  = _make_empty_fig()
    no_stats   = [html.P("—", className="text-muted")]
    hidden     = {"display": "none"}
    visible    = {"display": "block"}

    if not run_id:
        return empty_fig, empty_sec, "", no_stats, hidden, hidden, 0, {0: "0"}

    # Load field data
    u = _load_final_field(run_id)
    config = _load_config(run_id)

    if u is None:
        alert = dbc.Alert("No field data found. Run the simulation first.", color="warning")
        return empty_fig, empty_sec, alert, no_stats, hidden, hidden, 0, {0: "0"}

    coords = _load_dof_coords(run_id)
    dim    = config.get("dim", 2)

    # Fallback: if dof_coords.npy absent, synthesise coords on a regular grid
    if coords is None:
        nx = config.get("nx", 32); ny = config.get("ny", 32)
        nz = config.get("nz", 1)
        if dim == 2:
            xs = np.linspace(0, 1, nx + 1); ys = np.linspace(0, 1, ny + 1)
            Xg, Yg = np.meshgrid(xs, ys)
            coords  = np.column_stack([Xg.ravel(), Yg.ravel(),
                                       np.zeros((nx+1)*(ny+1))])
            u = u[: len(coords)]  # trim if necessary
        else:
            xs = np.linspace(0, 1, nx+1); ys = np.linspace(0, 1, ny+1)
            zs = np.linspace(0, 1, nz+1)
            Xg, Yg, Zg = np.meshgrid(xs, ys, zs)
            coords = np.column_stack([Xg.ravel(), Yg.ravel(), Zg.ravel()])
            u = u[: len(coords)]

    # For animation mode, possibly override u with a snapshot
    snaps = _list_snapshots(run_id)
    snap_times = _load_snapshot_times(run_id)
    n_snaps = len(snaps)
    anim_step = anim_store.get("step", 0) if anim_store else 0

    if view == "animation" and n_snaps > 0:
        idx = max(0, min(anim_step, n_snaps - 1))
        u_snap = _load_snapshot(run_id, snaps[idx])
        if u_snap is not None:
            u = u_snap

    # Slider for animation
    if n_snaps > 1:
        step_max = n_snaps - 1
        mark_step = max(1, step_max // 8)
        marks = {}
        for i in range(0, n_snaps, mark_step):
            t_label = f"{snap_times[i]:.2f}s" if snap_times is not None else str(i)
            marks[i] = {"label": t_label, "style": {"color": TEXT_COLOR,
                                                      "fontSize": "9px"}}
    else:
        step_max = 0
        marks = {0: "0"}

    # Panel visibility
    slice_panel_style = visible if view in ("z_slice", "x_slice", "y_slice", "volume") else hidden
    anim_panel_style  = visible if view == "animation" else hidden

    # Build main figure via factory; pass frac for slice modes
    effective_view = view if view != "animation" else "heatmap"
    frac = z_frac if z_frac is not None else 0.5
    try:
        main_fig = _build_field_figure(run_id, u, coords, config,
                                        effective_view, frac=frac)
    except Exception as exc:
        import traceback
        main_fig = _make_empty_fig(f"Error: {exc}")

    # Secondary plot
    effective_view2 = view if view != "animation" else "heatmap"
    try:
        sec_fig = _build_secondary_figure(run_id, u, coords, config, effective_view2)
    except Exception:
        sec_fig = _make_empty_fig()

    # Info badge
    t_label = ""
    if view == "animation" and snap_times is not None and anim_step < len(snap_times):
        t_label = f"  |  t = {snap_times[anim_step]:.3f} s"
    info = dbc.Badge(
        f"T ∈ [{u.min():.2f}, {u.max():.2f}] K  |  DOFs: {len(u):,}{t_label}",
        color="info", pill=True,
    )

    stats = _build_stats_panel(run_id, u, config)
    return (main_fig, sec_fig, info, stats,
            slice_panel_style, anim_panel_style,
            step_max, marks)


@app.callback(
    Output("field-anim-store",    "data",     allow_duplicate=True),
    Output("field-anim-interval", "disabled", allow_duplicate=True),
    Output("field-play-btn",      "children", allow_duplicate=True),
    Input("field-play-btn",       "n_clicks"),
    Input("field-anim-interval",  "n_intervals"),
    State("field-anim-store",     "data"),
    State("field-step-slider",    "max"),
    prevent_initial_call=True,
)
def handle_animation(play_clicks, n_intervals, store, step_max):
    store = store or {"step": 0, "playing": False}
    triggered = ctx.triggered_id

    if triggered == "field-play-btn":
        playing = not store["playing"]
        store["playing"] = playing
        label = "⏸ Pause" if playing else "▶ Play"
        return store, not playing, label

    elif triggered == "field-anim-interval":
        if store["playing"]:
            nxt = (store["step"] + 1) % (step_max + 1) if step_max else 0
            store["step"] = nxt
        return store, not store["playing"], no_update

    return store, True, "▶ Play"


@app.callback(
    Output("field-step-slider", "value", allow_duplicate=True),
    Input("field-anim-store",   "data"),
    prevent_initial_call=True,
)
def sync_slider_from_store(store):
    return (store or {}).get("step", 0)


@app.callback(
    Output("convergence-plot", "figure"),
    Output("convergence-stats", "children"),
    Input("conv-run-selector", "value"),
)
def update_convergence(run_ids):
    if not run_ids:
        fig = go.Figure()
        fig.update_layout(title="Select runs to view convergence", **PLOT_LAYOUT)
        return fig, ""

    fig = go.Figure()
    stats = []

    for rid in run_ids:
        result = _load_result(rid)
        config = _load_config(rid)
        history = result.get("convergence_history", [])
        if not history:
            continue
        dt = config.get("dt", 0.01)
        times = [i * dt for i in range(len(history))]
        fig.add_trace(go.Scatter(
            x=times, y=history, mode="lines", name=rid,
            line=dict(width=2),
        ))
        if history:
            stats.append(dbc.ListGroupItem(
                f"{rid}: L2 {history[0]:.4g} → {history[-1]:.4g}",
                color="dark",
            ))

    fig.update_layout(
        title="Convergence History (L2 Norm vs Time)",
        xaxis_title="Time [s]",
        yaxis_title="L2 Norm",
        yaxis_type="log",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        **PLOT_LAYOUT,
    )

    stats_widget = dbc.ListGroup(stats, flush=True) if stats else html.P("No data")
    return fig, stats_widget


@app.callback(
    Output("parametric-scatter",   "figure"),
    Output("parametric-wall-time", "figure"),
    Output("parametric-t-range",   "figure"),
    Input("param-runs-selector",   "value"),
    Input("param-x-selector",      "value"),
)
def update_parametric(run_ids, x_param):
    empty = go.Figure()
    empty.update_layout(**PLOT_LAYOUT)

    if not run_ids or not x_param:
        return empty, empty, empty

    rows = []
    for rid in run_ids:
        r = _load_result(rid)
        c = _load_config(rid)
        if not r or not c:
            continue
        rows.append({
            "run_id": rid,
            x_param: c.get(x_param),
            "t_max": r.get("max_temperature"),
            "t_min": r.get("min_temperature"),
            "t_mean": r.get("mean_temperature"),
            "wall_time": r.get("wall_time"),
        })

    if not rows:
        return empty, empty, empty

    xs    = [row[x_param] for row in rows]
    t_max = [row["t_max"] for row in rows]
    t_min = [row["t_min"] for row in rows]
    t_mean = [row["t_mean"] for row in rows]
    times = [row["wall_time"] for row in rows]
    rids  = [row["run_id"] for row in rows]

    # T_mean vs parameter
    scatter = go.Figure([
        go.Scatter(x=xs, y=t_mean, mode="lines+markers", name="T_mean",
                   marker=dict(color="#00b4d8", size=8)),
        go.Scatter(x=xs, y=t_max, mode="lines+markers", name="T_max",
                   marker=dict(color=ACCENT, size=8)),
    ])
    scatter.update_layout(
        title=f"Temperature vs {x_param}",
        xaxis_title=x_param, yaxis_title="Temperature [K]",
        **PLOT_LAYOUT,
    )

    # Wall time vs parameter
    wt_fig = go.Figure(go.Bar(x=xs, y=times, marker_color="#90e0ef"))
    wt_fig.update_layout(
        title=f"Wall Time vs {x_param}",
        xaxis_title=x_param, yaxis_title="Wall Time [s]",
        **PLOT_LAYOUT,
    )

    # T range (box-like: max-min spread)
    spread = [(mx or 0) - (mn or 0) for mx, mn in zip(t_max, t_min)]
    tr_fig = go.Figure(go.Scatter(
        x=xs, y=spread, mode="lines+markers",
        fill="tozeroy", line=dict(color="#f72585"),
        marker=dict(size=8),
    ))
    tr_fig.update_layout(
        title=f"Temperature Spread (T_max – T_min) vs {x_param}",
        xaxis_title=x_param, yaxis_title="ΔT [K]",
        **PLOT_LAYOUT,
    )

    return scatter, wt_fig, tr_fig


_QUICK_PROMPTS = [
    # ── History & Analysis (idx 0-6) → database (0-3) / analytics (4-6) ──────
    ("📋 List all past runs",
     "Show me all simulation runs in the database with their status, "
     "material, T_max, and wall time."),
    ("❌ Show failed runs",
     "List all failed simulation runs. For each, explain what went wrong "
     "and suggest how to fix the configuration."),
    ("🏆 Best convergence",
     "Which run had the best convergence (lowest final L2 norm)? "
     "Show its full configuration and results."),
    ("🌡 Highest temperature",
     "Which run recorded the highest peak temperature? "
     "Show its config and explain why it reached that temperature."),
    ("📊 Analyze recent 5 runs",
     "Analyze my last 5 simulation runs. Summarize T_max, T_min, "
     "convergence, and wall time. Highlight any trends."),
    ("🔬 Compare by material",
     "Compare all successful runs grouped by material. "
     "Which material gave the most uniform temperature distribution "
     "and fastest solve time?"),
    ("💡 Suggest next run",
     "Based on all past runs and knowledge graph patterns, "
     "what is the most informative next simulation to run? "
     "Identify gaps in BC pattern or material coverage and propose "
     "a specific configuration."),
    # spacers 7-9
    ("", ""), ("", ""), ("", ""),
    # ── Knowledge Graph (idx 10-13) → analytics agent ────────────────────────
    ("🔍 BC pattern outcomes",
     "Query the knowledge graph: compare all boundary condition patterns "
     "(dirichlet+neumann, dirichlet+robin, robin). "
     "For each pattern, what T_max range, average solve time, and "
     "materials were used? Which pattern is best for uniform heat distribution?"),
    ("📏 Domain size impact",
     "Query the knowledge graph: how does domain size affect simulation outcomes? "
     "Compare micro, component, panel, and structural scale domains — "
     "show average solve time, DOF count, and T_max for each class."),
    ("⚡ Thermal class comparison",
     "Query the knowledge graph: compare high-conductors (copper, aluminium, silicon) "
     "vs medium-conductors (steel, titanium) vs low-conductors (concrete, glass) vs "
     "thermal insulators (water, air). What T_max ranges and solve times do each class produce?"),
    ("🔗 Find similar to steel+robin",
     "Using the knowledge graph, find past runs most similar to: "
     "steel (k=50), 2D, dirichlet+robin BCs, component-scale domain (4cm × 2cm). "
     "Show the top 5 matches with their outcomes and relevance scores."),
    ("📖 Physics references for copper",
     "Retrieve the physics references for a copper simulation "
     "(k=385, rho=8960, cp=385) with robin BC (convective cooling). "
     "What are the validity limits and recommended h coefficients?"),
    ("⚙️ Solver guidance",
     "What are the recommended mesh resolution, time-step stability criteria, "
     "and element degree choices for heat conduction simulations? "
     "Use the knowledge graph physics references to answer."),
    # spacers 16-19
    ("", ""), ("", ""), ("", ""), ("", ""),
    # ── New Simulations (idx 20-24) → orchestrator ───────────────────────────
    ("🔲 2D steel plate",
     "Run a 2D heat equation on a steel plate 4cm × 2cm. "
     "k=50, rho=7850, cp=490. Left edge T=800K, right edge T=300K, "
     "top/bottom insulated (Neumann). t_end=10, dt=0.1, nx=40, ny=20."),
    ("🧊 3D aluminium block",
     "Run a 3D heat equation on a 2cm × 2cm × 2cm aluminium block. "
     "k=205, rho=2700, cp=900. "
     "One face at 500K, opposite face at 300K, all other faces insulated. "
     "t_end=5, dt=0.05, nx=ny=nz=20."),
    ("🌊 Convective cooling (copper)",
     "Run a 2D heat simulation on copper (k=400, rho=8960, cp=385), "
     "4cm × 2cm domain. Left edge fixed at T=800K, right edge Robin BC "
     "with h=500 W/m²K and T_inf=300K (convective cooling), "
     "top and bottom insulated. t_end=2, dt=0.05, nx=40, ny=20."),
    ("🔥 Internal heat source",
     "Run a 2D heat equation with a strong internal heat source of "
     "1e6 W/m³ on a 1cm × 1cm silicon domain (k=148, rho=2330, cp=710). "
     "All boundaries convective Robin with h=200, T_inf=300K. "
     "Mesh 40×40, t_end=1, dt=0.01."),
    ("🏗️ Concrete panel (large)",
     "Run a 2D heat equation on a concrete wall 50cm × 20cm "
     "(k=1.7, rho=2300, cp=880, Lx=0.5, Ly=0.2). "
     "Left face T=600K (fire side), right face T=293K (ambient), "
     "top/bottom insulated. nx=50, ny=20, t_end=3600, dt=30."),
]


@app.callback(
    Output("chat-input", "value"),
    Output("agent-selector", "value"),
    Input({"type": "quick-prompt", "index": ALL}, "n_clicks"),
    State("agent-selector", "value"),
    prevent_initial_call=True,
)
def fill_quick_prompt(n_clicks_list, current_agent):
    """When a quick-prompt button is clicked, pre-fill the chat input."""
    if not any(n_clicks_list):
        return no_update, no_update
    triggered = ctx.triggered_id
    if not triggered or not isinstance(triggered, dict):
        return no_update, no_update
    idx = triggered.get("index", -1)
    if idx < 0 or idx >= len(_QUICK_PROMPTS):
        return no_update, no_update
    _, prompt_text = _QUICK_PROMPTS[idx]
    if not prompt_text:
        return no_update, no_update
    # Route to the most capable agent for each prompt category
    if idx <= 3:        # list / search queries → database agent
        agent = "database"
    elif idx <= 15:     # analysis + knowledge graph + reference queries → analytics agent
        agent = "analytics"
    else:               # new simulation requests → orchestrator
        agent = "orchestrator"
    return prompt_text, agent


@app.callback(
    Output("chat-history",       "children"),
    Output("chat-job-store",     "data"),
    Output("chat-poll-interval", "disabled"),
    Output("chat-status-bar",    "children"),
    Input("chat-send-btn",       "n_clicks"),
    Input("chat-input",          "n_submit"),
    State("chat-input",          "value"),
    State("agent-selector",      "value"),
    State("chat-history",        "children"),
    prevent_initial_call=True,
)
def handle_chat_submit(n_clicks, n_submit, user_input, agent, history):
    """Submit the task to the async API and start the polling interval."""
    if not user_input:
        return history or [], {}, True, ""

    history = list(history or [])

    # User bubble
    history.append(html.Div([
        html.Span("You: ", style={"color": "#aaa", "fontWeight": "bold"}),
        html.Span(user_input),
    ], style={"marginBottom": "8px", "color": TEXT_COLOR}))

    # Submit to async endpoint
    if agent == "orchestrator":
        endpoint = "run/async"
        payload  = {"task": user_input}
    else:
        endpoint = f"agent/{agent}/async"
        payload  = {"task": user_input}

    response = _api_call("POST", endpoint, payload)

    job_id = response.get("job_id")
    if not job_id:
        # Submission itself failed
        err = response.get("error", json.dumps(response)[:300])
        history.append(html.Div([
            html.Span("⚠️ Error: ", style={"color": "#ff6b6b", "fontWeight": "bold"}),
            html.Span(err, style={"color": "#ff6b6b"}),
        ], style={"marginBottom": "12px"}))
        return history, {}, True, ""

    import time
    job_store = {"job_id": job_id, "agent": agent, "started_at": time.time()}
    status_msg = html.Span(
        f"⏳  Job {job_id} submitted — waiting for response...",
        style={"color": "#f0a500"},
    )
    return history, job_store, False, status_msg


@app.callback(
    Output("chat-history",       "children", allow_duplicate=True),
    Output("chat-job-store",     "data",     allow_duplicate=True),
    Output("chat-poll-interval", "disabled", allow_duplicate=True),
    Output("chat-status-bar",    "children", allow_duplicate=True),
    Input("chat-poll-interval",  "n_intervals"),
    State("chat-job-store",      "data"),
    State("chat-history",        "children"),
    prevent_initial_call=True,
)
def poll_job_result(n_intervals, job_store, history):
    """Poll the background job and update the chat when done."""
    import time

    if not job_store or not job_store.get("job_id"):
        return history or [], {}, True, ""

    job_id  = job_store["job_id"]
    agent   = job_store.get("agent", "orchestrator")
    t_start = job_store.get("started_at", time.time())
    elapsed = int(time.time() - t_start)

    job = _api_call("GET", f"jobs/{job_id}")

    if job.get("status") == "running":
        status_msg = html.Span(
            f"⏳  Running... {elapsed}s elapsed  (job {job_id})",
            style={"color": "#f0a500"},
        )
        return history or [], job_store, False, status_msg

    # Job finished (success or error)
    history = list(history or [])
    result  = job.get("result") or {}
    error   = job.get("error")

    if error:
        answer = f"Error: {error}"
        color  = "#ff6b6b"
    else:
        answer = (
            result.get("final_report") or
            result.get("answer") or
            json.dumps(result, indent=2)[:2000]
        )
        color = "#c8e6c9"

    history.append(html.Div([
        html.Span(
            f"🤖 {agent} ({job['elapsed_s']:.0f}s): ",
            style={"color": "#00b4d8", "fontWeight": "bold"},
        ),
        html.Pre(str(answer), style={
            "whiteSpace": "pre-wrap", "wordBreak": "break-word",
            "color": color, "fontSize": "0.8rem", "marginTop": "4px",
            "maxHeight": "300px", "overflowY": "auto",
        }),
    ], style={"marginBottom": "12px"}))

    return history, {}, True, ""


# ─── Run Explorer callbacks ───────────────────────────────────────────────────

def _explorer_status_badge(status: str) -> dbc.Badge:
    color = {"success": "success", "failed": "danger",
             "running": "warning", "pending": "secondary"}.get(status, "light")
    return dbc.Badge(status.upper(), color=color, className="ms-1")


def _explorer_run_item(run: dict) -> dbc.ListGroupItem:
    """
    Clickable list item for one simulation run.
    Uses dbc.ListGroupItem(action=True) so n_clicks works reliably in Dash.
    """
    t_end  = run.get("t_end") or "?"
    k_val  = run.get("k") or "?"
    nx, ny, nz = run.get("nx"), run.get("ny"), run.get("nz")
    mesh   = f"{nx}×{ny}" + (f"×{nz}" if nz else "")
    steps  = run.get("log_steps", 0)
    wt     = run.get("wall_time")
    wt_str = f"{wt:.1f}s" if wt else "–"
    status = run.get("status", "")
    s_color = {"success": "success", "failed": "danger",
               "running": "warning", "pending": "secondary"}.get(status, "light")

    return dbc.ListGroupItem(
        [
            html.Div([
                html.Span(run["run_id"],
                          style={"fontFamily": "monospace", "fontSize": "0.78rem",
                                 "fontWeight": "bold", "wordBreak": "break-all"}),
                dbc.Badge(status.upper(), color=s_color, className="ms-1 float-end"),
            ]),
            html.Div([
                dbc.Badge(f"{run.get('dim','?')}D", color="info",
                          pill=True, className="me-1"),
                html.Small(f"k={k_val}  {mesh}  t={t_end}  ⏱{wt_str}",
                           className="text-muted", style={"fontSize": "0.72rem"}),
            ], className="mt-1"),
            html.Div([
                html.Small(f"🧠 {steps} steps",
                           style={"fontSize": "0.70rem", "color": "#9e9e9e"}),
                html.Small(run.get("created_at", "")[:16],
                           style={"fontSize": "0.70rem", "color": "#666",
                                  "float": "right"}),
            ]),
        ],
        id={"type": "explorer-run-item", "index": run["run_id"]},
        action=True,          # makes it clickable & fires n_clicks
        n_clicks=0,
        style={"backgroundColor": CARD_COLOR, "border": "none",
               "borderBottom": "1px solid #2a2a3e",
               "padding": "8px 10px", "cursor": "pointer"},
        className="explorer-run-item",
    )


@app.callback(
    Output("explorer-run-list", "children"),
    Input("explorer-search-btn",   "n_clicks"),
    Input("explorer-refresh-btn",  "n_clicks"),
    Input("explorer-search-input", "value"),
    Input("explorer-status-filter","value"),
    Input("explorer-dim-filter",   "value"),
    prevent_initial_call=False,
)
def explorer_load_runs(n_search, n_refresh, search, status, dim):
    params = {}
    if search:
        params["search"] = search
    if status:
        params["status"] = status
    if dim and dim != "0":
        params["dim"] = dim
    params["limit"] = 200

    url = f"{AGENTS_API}/explorer/runs"
    try:
        r = requests.get(url, params=params, timeout=15)
        runs = r.json() if r.status_code == 200 else []
    except Exception:
        runs = []

    if not runs:
        return html.P("No runs found.", className="text-muted mt-2",
                      style={"padding": "8px"})

    return dbc.ListGroup(
        [_explorer_run_item(run) for run in runs],
        flush=True,
        style={"borderRadius": "6px", "overflow": "hidden",
               "border": "1px solid #2a2a3e"},
    )


# Track which list item was clicked — ListGroupItem(action=True) fires n_clicks reliably
@app.callback(
    Output("explorer-selected-run", "data"),
    Input({"type": "explorer-run-item", "index": ALL}, "n_clicks"),
    prevent_initial_call=False,   # must be False: prevent_initial_call=True suppresses
                                  # the first click after a list re-render in Dash
)
def explorer_select_run(n_clicks_list):
    # All zeros = initial render or re-render with reset counters, not a real click
    if not any(n_clicks_list):
        return no_update
    triggered = ctx.triggered_id
    if not triggered or not isinstance(triggered, dict):
        return no_update
    return triggered.get("index")


@app.callback(
    Output("explorer-detail-panel", "children"),
    Input("explorer-selected-run", "data"),
    prevent_initial_call=True,
)
def explorer_show_detail(run_id):
    if not run_id:
        return html.P("Select a run from the list.", className="text-muted mt-4")

    url = f"{AGENTS_API}/explorer/runs/{run_id}/detail"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return dbc.Alert(f"API error {r.status_code}: {r.text[:300]}", color="danger")
        detail = r.json()
    except Exception as exc:
        return dbc.Alert(f"Could not fetch detail: {exc}", color="danger")

    # ── Header strip ──────────────────────────────────────────────────────────
    header = dbc.Card(dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.H5(detail["run_id"],
                        style={"fontFamily": "monospace", "marginBottom": "2px"}),
                html.Small(
                    f"Created: {detail['created_at'][:19]}  │  "
                    f"Started: {(detail.get('started_at') or '–')[:19]}  │  "
                    f"Finished: {(detail.get('finished_at') or '–')[:19]}",
                    className="text-muted",
                ),
            ], width=8),
            dbc.Col([
                _explorer_status_badge(detail["status"]),
                dbc.Badge(f"{detail.get('dim','?')}D", color="info",
                          className="ms-2"),
                dbc.Badge(f"⏱ {detail.get('wall_time') or '?'}s",
                          color="secondary", className="ms-2"),
                dbc.Badge(f"{detail.get('n_dofs') or '?'} DOFs",
                          color="dark", className="ms-2"),
            ], width=4, className="text-end"),
        ]),
        html.Div(detail.get("error_msg") or "", style={"color": "#ff6b6b",
                                                        "fontSize": "0.8rem",
                                                        "marginTop": "4px"}),
    ]), color="dark", outline=True, className="mb-2")

    # ── KPI bar ───────────────────────────────────────────────────────────────
    res = detail.get("results", {})
    kpis = dbc.Row([
        dbc.Col(_kpi_mini("T_max", f"{res.get('t_max') or '–':.1f} K"
                          if isinstance(res.get('t_max'), float) else "–"), width=2),
        dbc.Col(_kpi_mini("T_min", f"{res.get('t_min') or '–':.1f} K"
                          if isinstance(res.get('t_min'), float) else "–"), width=2),
        dbc.Col(_kpi_mini("T_mean", f"{res.get('t_mean') or '–':.1f} K"
                          if isinstance(res.get('t_mean'), float) else "–"), width=2),
        dbc.Col(_kpi_mini("L2 norm", f"{res.get('final_l2_norm'):.2e}"
                          if isinstance(res.get('final_l2_norm'), float) else "–"), width=2),
        dbc.Col(_kpi_mini("Converged", "✅" if res.get("converged") else "❌"), width=2),
        dbc.Col(_kpi_mini("Agent logs",
                          str(len(detail.get("agent_logs", [])))), width=2),
    ], className="mb-2 g-1")

    # ── Sub-tabs ──────────────────────────────────────────────────────────────
    tabs = dbc.Tabs([

        # Overview
        dbc.Tab(label="📋 Overview", tab_id="exp-overview", children=[
            _build_explorer_overview(detail)
        ]),

        # Agent timeline
        dbc.Tab(label="🧠 Agent Timeline", tab_id="exp-logs", children=[
            _build_agent_timeline(detail.get("agent_logs", []))
        ]),

        # Reproducible config
        dbc.Tab(label="⚙️ Config", tab_id="exp-config", children=[
            _build_config_panel(detail.get("config", {}), detail["run_id"])
        ]),

        # Files
        dbc.Tab(label="📁 Files (MinIO)", tab_id="exp-files", children=[
            _build_files_panel(detail.get("minio_files", []))
        ]),

        # Recommendations
        dbc.Tab(label="💡 Recommendations", tab_id="exp-recs", children=[
            _build_recommendations_panel(detail.get("suggestions", []))
        ]),

    ], active_tab="exp-overview")

    return html.Div([header, kpis, tabs])


# ── Explorer sub-panel builders ───────────────────────────────────────────────

def _kpi_mini(label: str, value: str) -> dbc.Card:
    return dbc.Card(dbc.CardBody([
        html.Div(value, style={"fontSize": "1rem", "fontWeight": "bold",
                               "color": "#e0e0e0"}),
        html.Div(label, style={"fontSize": "0.7rem", "color": "#888"}),
    ], style={"padding": "6px 8px"}), color="dark", outline=True)


def _build_explorer_overview(detail: dict) -> html.Div:
    cfg   = detail.get("config", {})
    bcs   = cfg.get("bcs", [])

    def _row(label, value):
        return html.Tr([
            html.Td(label, style={"color": "#aaa", "paddingRight": "16px",
                                  "fontSize": "0.82rem", "whiteSpace": "nowrap"}),
            html.Td(str(value), style={"fontFamily": "monospace",
                                       "fontSize": "0.82rem"}),
        ])

    # Physics table
    physics_rows = [
        _row("PDE type",    cfg.get("pde_type", "heat_equation")),
        _row("Dimension",   cfg.get("dim")),
        _row("Mesh",        f"{cfg.get('nx')} × {cfg.get('ny')}"
                            + (f" × {cfg.get('nz')}" if cfg.get("nz") else "")),
        _row("Element deg", cfg.get("element_degree", 1)),
        _row("k (W/m·K)",   cfg.get("k")),
        _row("ρ (kg/m³)",   cfg.get("rho")),
        _row("cₚ (J/kg·K)", cfg.get("cp")),
        _row("Source (W/m³)", cfg.get("source", 0.0)),
        _row("t_end (s)",   cfg.get("t_end")),
        _row("dt (s)",      cfg.get("dt")),
        _row("θ-scheme",    cfg.get("theta", 1.0)),
        _row("n_dofs",      detail.get("n_dofs")),
        _row("n_timesteps", detail.get("n_timesteps")),
        _row("Wall time",   f"{detail.get('wall_time') or '?'}s"),
    ]

    # Boundary conditions table
    bc_rows = []
    for bc in (bcs if isinstance(bcs, list) else []):
        if isinstance(bc, dict):
            bc_rows.append(_row(bc.get("type", "BC"),
                                f"mark={bc.get('mark','?')} "
                                f"val={bc.get('value','?')} "
                                f"{'h='+str(bc.get('h','')) if bc.get('h') else ''}"))

    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H6("Physics Parameters", className="text-info mt-2 mb-1"),
                html.Table(physics_rows,
                           style={"width": "100%", "borderSpacing": "0 4px"}),
            ], width=6),
            dbc.Col([
                html.H6("Boundary Conditions", className="text-info mt-2 mb-1"),
                (html.Table(bc_rows,
                            style={"width": "100%", "borderSpacing": "0 4px"})
                 if bc_rows else html.P("No BCs recorded.", className="text-muted")),
            ], width=6),
        ]),
    ], className="mt-2 ps-1")


def _step_icon(step_type: str) -> str:
    return {
        "reasoning":    "🧠",
        "tool_call":    "🔧",
        "tool_result":  "✅",
        "final_answer": "🏁",
    }.get(step_type, "•")


def _build_agent_timeline(logs: list[dict]) -> html.Div:
    if not logs:
        return html.P(
            "No agent logs found for this run. "
            "Logs are captured for runs executed via the Agent Chat.",
            className="text-muted mt-3",
        )

    # Group by task_id so we can show separate agent sessions
    sessions: dict[str, list] = {}
    for log in logs:
        tid = log["task_id"]
        sessions.setdefault(tid, []).append(log)

    accordion_items = []
    for tid, steps in sessions.items():
        agent = steps[0].get("agent_name", "?") if steps else "?"
        n_tool = sum(1 for s in steps if s["step_type"] == "tool_call")

        # Build timeline items for this session
        timeline_items = []
        for step in sorted(steps, key=lambda s: s["step_index"]):
            icon  = _step_icon(step["step_type"])
            stype = step["step_type"]
            content_data = step.get("content", {})
            ts = step.get("created_at", "")[:19]
            elapsed = step.get("elapsed_ms")
            el_str = f"  ({elapsed}ms)" if elapsed else ""

            if stype == "reasoning":
                body_text = content_data.get("content", "")
            elif stype == "tool_call":
                args = content_data.get("args", {})
                try:
                    body_text = (f"Tool: **{content_data.get('tool')}**\n"
                                 + json.dumps(args, indent=2)[:1200])
                except Exception:
                    body_text = str(content_data)[:600]
            elif stype == "tool_result":
                result = content_data.get("result", content_data)
                try:
                    body_text = json.dumps(result, indent=2)[:2000]
                except Exception:
                    body_text = str(result)[:1000]
            elif stype == "final_answer":
                body_text = content_data.get("answer", "")
            else:
                body_text = str(content_data)[:500]

            header_text = (
                f"{icon} [{step['step_index']}] {stype}"
                f"  —  {agent}  {ts}{el_str}"
            )

            timeline_items.append(
                dbc.AccordionItem(
                    html.Pre(body_text,
                             style={"whiteSpace": "pre-wrap", "wordBreak": "break-word",
                                    "fontSize": "0.78rem", "color": "#ddd",
                                    "maxHeight": "300px", "overflowY": "auto",
                                    "margin": "0"}),
                    title=header_text,
                    item_id=f"step-{tid[:8]}-{step['step_index']}",
                    style={"backgroundColor": CARD_COLOR},
                )
            )

        accordion_items.append(
            dbc.AccordionItem(
                dbc.Accordion(timeline_items, flush=True, always_open=True,
                              start_collapsed=True),
                title=(f"Task {tid[:12]}…  │  {agent}  │  "
                       f"{len(steps)} steps  │  {n_tool} tool calls"),
                item_id=f"session-{tid[:8]}",
            )
        )

    return html.Div([
        html.P(f"{len(logs)} log entries across {len(sessions)} agent session(s).",
               className="text-muted mt-2 mb-1", style={"fontSize": "0.8rem"}),
        dbc.Accordion(accordion_items, flush=True, start_collapsed=False),
    ], className="mt-2")


def _build_config_panel(config: dict, run_id: str) -> html.Div:
    try:
        config_str = json.dumps(config, indent=2)
    except Exception:
        config_str = str(config)

    return html.Div([
        html.P(
            "Copy this JSON and paste it into the Agent Chat or the REST API "
            "to reproduce this run exactly:",
            className="text-muted mt-2", style={"fontSize": "0.82rem"},
        ),
        dbc.Alert([
            html.Code(
                f'curl -s http://localhost:8000/agent/simulation/async \\\n'
                f'  -H "Content-Type: application/json" \\\n'
                f'  -d \'{{"task": "Run this config: {config_str[:120]}…"}}\''
                , style={"fontSize": "0.75rem", "whiteSpace": "pre-wrap"}
            ),
        ], color="dark"),
        html.Label("Full config.json:", className="text-info",
                   style={"fontSize": "0.82rem"}),
        html.Pre(
            config_str,
            style={
                "backgroundColor": "#0a0a16",
                "color": "#a8e6cf",
                "padding": "12px",
                "borderRadius": "6px",
                "fontSize": "0.78rem",
                "overflowX": "auto",
                "maxHeight": "60vh",
                "whiteSpace": "pre",
            },
        ),
    ])


def _build_files_panel(files: list[dict]) -> html.Div:
    if not files:
        return html.P("No files stored in MinIO for this run.",
                      className="text-muted mt-3")

    # Check for errors
    if len(files) == 1 and "error" in files[0]:
        return dbc.Alert(f"MinIO error: {files[0]['error']}", color="warning")

    def _size_str(sz):
        if sz is None:
            return "?"
        if sz >= 1_048_576:
            return f"{sz/1_048_576:.1f} MB"
        if sz >= 1_024:
            return f"{sz/1_024:.1f} KB"
        return f"{sz} B"

    def _file_icon(name: str) -> str:
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        return {"npy": "🔢", "json": "📄", "xdmf": "🗂️", "h5": "📦",
                "vtk": "🗂️", "vtu": "🗂️", "csv": "📊"}.get(ext, "📎")

    rows = [
        html.Tr([
            html.Td(_file_icon(f["name"]),
                    style={"fontSize": "1rem", "paddingRight": "8px"}),
            html.Td(f["name"],
                    style={"fontFamily": "monospace", "fontSize": "0.78rem"}),
            html.Td(_size_str(f.get("size")),
                    style={"fontSize": "0.75rem", "color": "#aaa",
                           "paddingLeft": "12px"}),
            html.Td((f.get("last_modified") or "")[:16],
                    style={"fontSize": "0.72rem", "color": "#666",
                           "paddingLeft": "12px"}),
        ])
        for f in sorted(files, key=lambda x: x.get("name", ""))
    ]

    return html.Div([
        html.P(f"{len(files)} file(s) stored in MinIO.",
               className="text-muted mt-2", style={"fontSize": "0.82rem"}),
        html.Table(
            [html.Thead(html.Tr([html.Th(""), html.Th("File"), html.Th("Size"),
                                 html.Th("Modified")])),
             html.Tbody(rows)],
            style={"width": "100%", "borderSpacing": "0 3px"},
        ),
    ], className="mt-2")


def _build_recommendations_panel(suggestions: list[dict]) -> html.Div:
    if not suggestions:
        return html.P(
            "No recommendations yet. Run the Analytics Agent on this run "
            "to generate suggestions for follow-up simulations.",
            className="text-muted mt-3",
        )

    cards = []
    for s in suggestions:
        priority = s.get("priority", 5)
        accepted = s.get("accepted")
        color = "success" if accepted else ("warning" if priority <= 3 else "secondary")
        try:
            cfg_str = json.dumps(s.get("suggested_config", {}), indent=2)
        except Exception:
            cfg_str = str(s.get("suggested_config", ""))

        cards.append(dbc.Card([
            dbc.CardHeader([
                dbc.Badge(f"Priority {priority}", color=color, className="me-2"),
                dbc.Badge("Accepted ✅" if accepted else "Pending",
                          color="success" if accepted else "secondary"),
                html.Small(f"  {s.get('created_at','')[:16]}",
                           className="text-muted ms-2"),
            ]),
            dbc.CardBody([
                html.H6("Rationale", className="text-info"),
                html.P(s.get("rationale", ""), style={"fontSize": "0.82rem"}),
                html.H6("Suggested Config", className="text-info mt-2"),
                html.Pre(cfg_str,
                         style={"backgroundColor": "#0a0a16", "color": "#a8e6cf",
                                "padding": "8px", "borderRadius": "4px",
                                "fontSize": "0.75rem", "overflowX": "auto",
                                "maxHeight": "200px"}),
            ]),
        ], color="dark", outline=True, className="mb-2"))

    return html.Div(cards, className="mt-2")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("DASHBOARD_PORT", 8050)),
        debug=os.getenv("DASH_DEBUG", "false").lower() == "true",
    )
