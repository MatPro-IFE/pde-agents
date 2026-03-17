# PDE Agents

A multi-agent ecosystem built on open-source LLMs running locally to solve PDEs with the Finite Element Method, enhanced with a GraphRAG knowledge graph for physics-informed reasoning.

**Hardware:** 2× NVIDIA RTX PRO 6000 Blackwell Server Edition (~98 GB VRAM each · ~196 GB total) · CUDA 13.1  
**FEM Solver:** DOLFINx (FEniCSx) `0.10.0.post2`

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR (LangGraph)                           │
│              LLM: llama3.3:70b  (supervisor)                         │
└───────────┬───────────────┬────────────────┬─────────────────────────┘
            │               │                │
            ▼               ▼                ▼
  ┌─────────────────┐ ┌──────────────┐ ┌───────────────┐
  │  AGENT-1        │ │  AGENT-2     │ │  AGENT-3      │
  │  Simulation     │ │  Analytics   │ │  Database     │
  │  qwen2.5-coder  │ │  llama3.3    │ │  qwen2.5-coder│
  │  :32b           │ │  :70b        │ │  :14b         │
  └────────┬────────┘ └──────┬───────┘ └──────┬────────┘
           │                 │                │
           │    ┌────────────┴────────────┐   │
           ▼    ▼                         ▼   ▼
  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────────────┐
  │  FEniCSx     │  │  NumPy       │  │  Neo4j Knowledge Graph (GraphRAG)│
  │  DOLFINx     │  │  Plotly Dash │  │  ─────────────────────────────│
  │  2D/3D FEM   │  │  PostgreSQL  │  │  Run nodes + 768-dim embeddings│
  │  Gmsh meshes │  │  FastAPI     │  │  SIMILAR_TO KNN edges          │
  └──────────────┘  └──────────────┘  │  Reference nodes (physics KB)  │
                                       │  Semantic vector search (HNSW) │
                                       └───────────────────────────────┘
```

### Agents

| Agent | Model | Role |
|-------|-------|------|
| Simulation Agent | `qwen2.5-coder:32b` | Set up, validate (KG-aware), run, and debug FEM simulations |
| Analytics Agent  | `llama3.3:70b`      | Analyze results, compare runs, query the knowledge graph |
| Database Agent   | `qwen2.5-coder:14b` | Store results, run SQL queries, catalog studies, search history |
| Orchestrator     | `llama3.3:70b`      | Coordinate all agents, synthesize final report |

### Services

All browser-facing UIs are accessed through a single **nginx reverse proxy on port 8050**.

| Service | Host access | Purpose |
|---------|-------------|---------|
| **nginx** | `:8050` | Reverse proxy — single entry point for all web UIs |
| Dashboard | `:8050/` (via nginx) | Plotly Dash visualization, agent chat, KG explorer |
| Agents API | `:8050/agents/` or `:8000` | FastAPI REST + Swagger UI |
| Neo4j Browser | `:7474` (direct) | Knowledge graph browser |
| **NeoDash** | `:9001` (direct) | Open-source graph dashboard — reuses ex-MinIO console port |
| MinIO Console | `:9002` → SSH tunnel | Object storage browser (moved to port 9002) |
| MinIO API | `:9000` | S3-compatible object storage API |
| FEniCSx Runner | `:8080` | DOLFINx simulation job REST API |
| Ollama | `:11434` | Local LLM server (GPU-accelerated) |
| PostgreSQL | `:5432` | Simulation metadata database |
| Redis | `:6379` | Message broker for agent coordination |
| Neo4j Bolt | `:7687` | Graph database driver protocol |

> **NeoDash** reuses port 9001 so no new firewall rule is needed.
> The MinIO **S3 API on port 9000** is unaffected — all agent file uploads and
> downloads continue normally. The MinIO web console is remapped to **port 9002**
> (internal only — no firewall change required). Access it via SSH tunnel:
> ```bash
> ssh -L 19002:localhost:9002 -N <server>   # then open http://localhost:19002
> ```

---

## Quick Start

### Prerequisites

**Required:** Docker 29+, NVIDIA drivers (tested: 590.48.01), NVIDIA Container Toolkit.

Install NVIDIA Container Toolkit if not present:
```bash
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 1. One-time setup
```bash
cd ~/pde-agents
cp env.example .env        # edit passwords/model names if desired
chmod +x setup.sh
./setup.sh
```

This will:
- Build the three custom Docker images (agents, fenics-runner, dashboard)
- Start all services including Neo4j and NeoDash
- Initialize the PostgreSQL schema
- Auto-seed the knowledge graph with 10 engineering materials, 6 known failure patterns, and 20 physics reference nodes
- Pull LLM models in the background (~70 GB total, 30–90 min first time)

### 2. Pull the embedding model

The knowledge graph uses `nomic-embed-text` (274 MB) for semantic similarity:
```bash
docker exec pde-ollama ollama pull nomic-embed-text
```

### 3. Confirm all models are ready
```bash
docker exec pde-ollama ollama list
# Expected output once all pulls complete:
# NAME                    SIZE
# llama3.3:70b            42 GB
# qwen2.5-coder:32b       19 GB
# qwen2.5-coder:14b       9.0 GB
# nomic-embed-text:latest 274 MB
```

### 4. Check all services are healthy
```bash
make health
```

### 5. Open the dashboard
Navigate to **http://localhost:8050** in your browser.

All services are accessible from the dashboard navbar or directly:

| URL | What you get |
|-----|-------------|
| http://localhost:8050/ | Dashboard (all tabs) |
| http://localhost:8050/agents/docs | API Swagger UI |
| http://localhost:7474 | Neo4j Browser |
| http://localhost:9001 | NeoDash (graph explorer — reuses MinIO console port) |
| via SSH tunnel | MinIO console (see note below) |

---

## Use Cases

All examples below can be sent to the Agents REST API at `http://localhost:8000`
(direct) or `http://localhost:8050/agents/` (via nginx), or run via individual
agent Python scripts inside the container.

---

### Use Case 1 — 2D Heat Equation (Steel Plate)

A steel plate heated on the right wall, fixed temperature on the left, insulated
top and bottom. Classic Dirichlet + Neumann boundary conditions.

**Via Agents API (recommended):**
```bash
curl -s -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Run a 2D heat equation on a 64x64 steel plate. Left wall 300K, right wall 500K, top and bottom insulated. Use u_init=300.0, t_end=100.0, dt=0.5. Store the results and report the temperature range and wall time."
  }' | python3 -m json.tool
```

**Direct FEniCSx REST API:**
```bash
curl -s -X POST http://localhost:8080/run \
  -H "Content-Type: application/json" \
  -d '{
    "dim": 2, "nx": 64, "ny": 64,
    "k": 50.0, "rho": 7800.0, "cp": 500.0,
    "u_init": 300.0,
    "bcs": [
      {"type": "dirichlet", "value": 300.0, "location": "left"},
      {"type": "dirichlet", "value": 500.0, "location": "right"},
      {"type": "neumann",   "value": 0.0,   "location": "top"},
      {"type": "neumann",   "value": 0.0,   "location": "bottom"}
    ],
    "t_end": 100.0, "dt": 0.5,
    "run_id": "steel_plate_64x64",
    "output_dir": "/workspace/results"
  }' | python3 -m json.tool
```

After every successful run the system automatically:
1. Stores metadata in **PostgreSQL**
2. Uploads output files to **MinIO** (`simulation-results/runs/<run_id>/`)
3. Adds a **Run node** to the **Neo4j knowledge graph** (material inference + rule warnings)
4. Embeds the run summary with **nomic-embed-text** (768-dim vector stored on the Run node)
5. Creates **SIMILAR_TO** edges to the top-5 most similar past runs (cosine ≥ 0.85)

---

### Use Case 2 — Complex Geometry with Gmsh

FEniCSx supports 9 built-in Gmsh geometry types for non-rectangular domains.
Specify `"geometry"` in the config with a `"type"` key:

```bash
curl -s -X POST http://localhost:8080/run \
  -H "Content-Type: application/json" \
  -d '{
    "dim": 2,
    "k": 50.0, "rho": 7800.0, "cp": 500.0,
    "u_init": 300.0,
    "geometry": {"type": "l_shape", "Lx": 0.08, "Ly": 0.08, "mesh_size": 0.005},
    "bcs": [
      {"type": "dirichlet", "value": 800.0, "boundary": "left"},
      {"type": "robin",     "h": 25.0,  "T_inf": 300.0, "boundary": "top"},
      {"type": "robin",     "h": 25.0,  "T_inf": 300.0, "boundary": "right"},
      {"type": "neumann",   "value": 0.0,               "boundary": "bottom"}
    ],
    "t_end": 0.5, "dt": 0.02,
    "run_id": "steel_l_shape_robin"
  }' | python3 -m json.tool
```

**Available geometry types:**

| Type | Description | Boundaries |
|------|-------------|------------|
| `rectangle` | Standard rectangle (default) | `left`, `right`, `top`, `bottom` |
| `l_shape` | L-shaped domain (re-entrant corner) | `left`, `right`, `top`, `bottom`, `inner_h`, `inner_v` |
| `circle` | Full circular disk | `boundary` |
| `annulus` | Ring / annular domain | `inner_wall`, `outer_wall` |
| `hollow_rectangle` | Rectangle with rectangular hole | `outer_*`, `inner_*` |
| `t_shape` | T-shaped cross-section | `left`, `right`, `top`, `bottom`, `stem_left`, `stem_right` |
| `stepped_notch` | Stepped notch (stress concentrator) | `left`, `right`, `top`, `bottom`, `notch_*` |
| `box` | 3D rectangular box | `left`, `right`, `top`, `bottom`, `front`, `back` |
| `cylinder` | 3D cylinder | `bottom_face`, `top_face`, `lateral` |

For Gmsh geometries use `"boundary"` key in BCs (physical group name) instead of `"location"`.

---

### Use Case 3 — 3D Heat Equation (Aluminum Block with Convection)

```bash
curl -s -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Run a 3D heat equation on a 16x16x16 aluminum block. k=237, rho=2700, cp=900. Left wall 273K, right wall 373K. Front and back surfaces have Robin convection: h=25, T_inf=293K. Top and bottom insulated. t_end=300.0, dt=2.0. Report min/max temperature."
  }' | python3 -m json.tool
```

---

### Use Case 4 — Parametric Sweep (Thermal Conductivity Study)

```bash
curl -s -X POST http://localhost:8000/agent/simulation \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Run a parametric sweep on a 2D 32x32 domain varying thermal conductivity k over [1, 10, 50, 100, 200] W/(mK). Keep rho=7800, cp=500, left wall 300K, right wall 500K, insulated top and bottom, u_init=300.0, t_end=20.0, dt=0.5."
  }' | python3 -m json.tool
```

---

### Use Case 5 — Analytics: Compare Runs

```bash
curl -s -X POST http://localhost:8000/agent/analytics \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Compare the runs steel_plate_64x64 and steel_agent_final. What is the difference in temperature distribution? Which mesh resolution is sufficient and why?"
  }' | python3 -m json.tool
```

---

### Use Case 6 — Database Queries in Natural Language

```bash
# Find the hottest simulations
curl -s -X POST http://localhost:8000/agent/database \
  -H "Content-Type: application/json" \
  -d '{"task": "Show me all runs where the maximum temperature exceeded 450K, ordered by wall time."}' \
  | python3 -m json.tool

# Material lookup via knowledge graph
curl -s -X POST http://localhost:8000/agent/database \
  -H "Content-Type: application/json" \
  -d '{"task": "What material properties does copper have? Find similar past runs that used copper-like conductivity."}' \
  | python3 -m json.tool
```

---

### Use Case 7 — Knowledge Graph Queries

The knowledge graph accumulates information with every run and stores curated physics knowledge. Query it via the API or ask any agent.

```bash
# Graph statistics (includes embedding coverage and reference count)
curl -s http://localhost:8000/kg/stats | python3 -m json.tool

# Semantic similarity search — finds past runs similar to a natural language description
curl -s -X POST http://localhost:8000/kg/search \
  -H "Content-Type: application/json" \
  -d '{"query": "steel l-shape with robin convection, component-scale", "top_k": 5}' \
  | python3 -m json.tool

# Pre-run context: warnings + similar runs + physics references
curl -s -X POST http://localhost:8000/kg/check \
  -H "Content-Type: application/json" \
  -d '{"k": 50, "dim": 2, "Lx": 0.08, "Ly": 0.08,
       "bcs": [{"type": "dirichlet", "value": 800},
               {"type": "robin", "h": 25, "T_inf": 300}]}' \
  | python3 -m json.tool

# Re-seed static knowledge (safe to call repeatedly)
curl -s -X POST http://localhost:8000/kg/seed | python3 -m json.tool
```

**Browse the graph visually:**

- **Neo4j Browser** at `http://localhost:7474` — raw Cypher queries, graph visualization
- **NeoDash** at `http://localhost:5005` — no-code graph dashboards and visualizations (open-source, Apache 2.0)

Example Cypher queries:
```cypher
-- Semantically similar run pairs (via KNN edges)
MATCH (r:Run)-[rel:SIMILAR_TO]->(s:Run)
RETURN r.run_id, s.run_id, rel.score AS similarity
ORDER BY rel.score DESC LIMIT 10

-- Runs grouped by BC pattern with average outcomes
MATCH (r:Run)-[:USES_BC_CONFIG]->(b:BCConfig)
WHERE r.status = 'success'
RETURN b.pattern, count(r) AS runs, avg(r.t_max) AS avg_t_max
ORDER BY runs DESC

-- Physics references for a material
MATCH (m:Material {name: 'steel'})-[:HAS_REFERENCE]->(ref:Reference)
RETURN ref.subject, ref.text, ref.source

-- Full knowledge graph overview
MATCH (n) RETURN labels(n)[0] AS type, count(n) AS count ORDER BY count DESC
```

---

### Use Case 8 — Pre-Run Physics Check

Ask the Simulation Agent to validate a configuration using the knowledge graph before running:

```bash
curl -s -X POST http://localhost:8000/agent/simulation \
  -H "Content-Type: application/json" \
  -d '{
    "task": "What issues might arise if I run a 2D simulation with k=50, nx=8, ny=8, theta=0.0, dt=0.5, u_init=0, and Dirichlet BCs at 300 K and 800 K? Use check_config_warnings and get_physics_references."
  }' | python3 -m json.tool
```

The response includes:
- Rule violations (mesh too coarse, explicit instability, inconsistent IC)
- Semantic similar runs with outcomes
- Physics reference facts (h coefficients, k(T) validity, solver guidance)
- An overall `recommendation` string

---

### Use Case 9 — Run Explorer (Dashboard)

The **🔎 Run Explorer** tab provides a visual interface to browse every simulation run:

- **Left panel:** searchable, filterable run list (status, dimension, keyword)
- **Right panel:** detailed view with sub-tabs for:
  - **Overview** — KPIs (T_max, T_min, wall time, DOFs) + SIMILAR_TO neighbour table
  - **Agent Timeline** — step-by-step trace of every agent reasoning and tool-call
  - **Config** — full simulation configuration used (for reproducibility)
  - **Files** — MinIO file listing with object names and sizes
  - **Recommendations** — agent suggestions from the run

---

### Use Case 10 — Knowledge Graph Tab (Dashboard)

The **🧠 Knowledge Graph** tab in the dashboard exposes the GraphRAG features visually:

**Graph Statistics panel:** Real-time counts of all node types, embedding coverage percentage, SIMILAR_TO edge count, and Reference node count.

**Semantic Run Search:** Type a free-text description (e.g. "2D steel with convective cooling") and the KG embeds it with nomic-embed-text and finds the top-5 similar past runs using the HNSW vector index. Results show similarity scores with progress bars.

**Physics Reference Browser:** Browse all 20 curated reference facts filtered by type:
- 📐 Material Properties — k(T) and cp(T) dependence, phase transition limits
- 🔲 BC Practice — realistic h coefficients, heat flux magnitudes, temperature validity
- ⚙️ Solver Guidance — mesh resolution rules, Fourier number criterion, P2 vs P1
- 🌐 Domain Physics — radiation at micro-scale, buoyancy at structural scale, thermal time constants

**NeoDash launcher:** Opens the open-source graph visualization tool at port 5005.

---

### Use Case 11 — Direct Python (inside containers)

#### Run the solver directly
```python
# docker exec -it pde-fenics python3
import sys; sys.path.insert(0, '/workspace')
from simulations.solvers.heat_equation import HeatConfig, HeatEquationSolver

# Gmsh L-shape geometry with named boundary conditions
cfg = HeatConfig(
    dim=2,
    k=50.0, rho=7800.0, cp=500.0,
    u_init=300.0,
    geometry={"type": "l_shape", "Lx": 0.08, "Ly": 0.08, "mesh_size": 0.005},
    bcs=[
        {"type": "dirichlet", "value": 800.0, "boundary": "left"},
        {"type": "robin",     "h": 25.0, "T_inf": 300.0, "boundary": "top"},
        {"type": "neumann",   "value": 0.0,              "boundary": "right"},
    ],
    t_end=0.5, dt=0.02,
    run_id="l_shape_robin",
    output_dir="/workspace/results",
)
result = HeatEquationSolver(cfg).solve()
print(result)
```

#### Query the knowledge graph with semantic search
```python
# docker exec -it pde-agents python3
import sys; sys.path.insert(0, '/app')
from knowledge_graph.graph import get_kg

kg = get_kg()
print("Stats:", kg.stats())
# Stats: {'total_runs': 313, 'embedded_runs': 313, 'references': 20,
#          'materials': 10, 'bc_configs': 4, 'domains': 4, ...}

# Semantic similarity search using vector embeddings
similar = kg.get_similar_runs_semantic({
    "k": 50.0, "dim": 2, "Lx": 0.08, "Ly": 0.08,
    "bcs": [{"type": "dirichlet"}, {"type": "robin", "h": 25}],
    "geometry": {"type": "l_shape"},
}, top_k=5)
for r in similar:
    print(f"  {r['run_id'][:16]}  score={r['similarity_score']:.4f}  T_max={r['t_max']:.0f}K")

# Get curated physics references for a config
refs = kg.get_references_for_config({"k": 50.0, "Lx": 0.1, "Ly": 0.1,
    "bcs": [{"type": "robin", "h": 20}]})
for r in refs:
    print(f"  [{r['type']}] {r['subject']} — {r['source']}")

# Full pre-run context (warnings + similar + material + references)
ctx = kg.get_pre_run_context({"k": 50, "dim": 2, "Lx": 0.08, "Ly": 0.08,
    "bcs": [{"type": "dirichlet", "value": 300},
            {"type": "robin", "h": 25, "T_inf": 300}]})
print("Recommendation:", ctx["recommendation"])
print("Physics refs:", len(ctx["physics_references"]))
```

#### Backfill embeddings and build KNN edges
```python
# docker exec -it pde-agents python3
import sys; sys.path.insert(0, '/app')
from knowledge_graph.graph import get_kg
kg = get_kg()

# Embed all runs that don't yet have a vector
result = kg.backfill_embeddings(batch_size=50)
print(result)  # {'newly_embedded': 0, 'failed': 0, ...}

# Build / refresh SIMILAR_TO KNN edges for all embedded runs
result = kg.build_all_similar_to_edges(k=5, min_score=0.85)
print(result)  # {'processed': 313, 'total_similar_to_in_graph': 1566}
```

#### Drive the Simulation Agent directly
```python
# docker exec -it pde-agents python3
import sys; sys.path.insert(0, '/app')
from agents.simulation_agent import SimulationAgent

agent = SimulationAgent()
result = agent.run(
    "Run a 2D heat equation for a copper plate: "
    "k=385, rho=8960, cp=385, 32x32 mesh. "
    "Left 273K, right 373K, insulated top/bottom. "
    "u_init=273.0, t_end=5.0, dt=0.1, run_id='copper_test'."
)
print(result["answer"])
```

---

## Physics: Heat Equation

### Mathematical Formulation

**Strong form** (transient heat conduction with source):
```
ρ c_p ∂u/∂t − ∇·(k ∇u) = f    in Ω × (0, T]
```

**Boundary conditions:**
- Dirichlet: `u = g` on Γ_D  (prescribed temperature)
- Neumann (flux): `k ∂u/∂n = h` on Γ_N  (insulated: h=0)
- Robin (convective): `k ∂u/∂n = α(u_∞ − u)` on Γ_R

**Weak form — θ-scheme (Backward Euler θ=1, Crank-Nicolson θ=0.5):**
```
(ρc_p/dt)(u^{n+1} − u^n, v)_Ω
  + θ   [ k(∇u^{n+1}, ∇v)_Ω + α(u^{n+1}, v)_ΓR ]
  + (1−θ)[ k(∇u^n, ∇v)_Ω    + α(u^n, v)_ΓR     ]
= ∫_Ω f v dx + ∫_ΓN h v ds + α(u_∞, v)_ΓR
```

### Configuration Reference

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `dim` | int | Spatial dimension (2 or 3) | `2` |
| `nx`, `ny`, `nz` | int | Mesh divisions (built-in meshes) | `64, 64` |
| `k` | float | Thermal conductivity [W/(m·K)] | `50.0` (steel) |
| `rho` | float | Density [kg/m³] | `7800.0` |
| `cp` | float | Specific heat [J/(kg·K)] | `500.0` |
| `source` | float | Volumetric heat source [W/m³] | `0.0` |
| `u_init` | float | Initial temperature [K] — set close to BCs | `300.0` |
| `t_end` | float | Simulation end time [s] | `100.0` |
| `dt` | float | Time step [s] | `0.5` |
| `theta` | float | Time integration: 1.0=BE, 0.5=CN | `1.0` |
| `geometry` | dict | Gmsh geometry spec — see table above | `{"type": "l_shape"}` |
| `bcs` | list | Boundary condition specs | — |
| `run_id` | str | Unique run identifier | `"steel_plate_01"` |
| `output_dir` | str | Directory for XDMF/VTK output | `"/workspace/results"` |

**Boundary condition spec:**
```json
// Built-in mesh (location key)
{"type": "dirichlet", "value": 300.0, "location": "left"}
{"type": "neumann",   "value": 0.0,   "location": "top"}
{"type": "robin",     "alpha": 10.0,  "u_inf": 293.0, "location": "front"}

// Gmsh mesh (boundary key — matches physical group name)
{"type": "dirichlet", "value": 800.0, "boundary": "left"}
{"type": "robin",     "h": 25.0, "T_inf": 300.0, "boundary": "outer_wall"}
```

> **Important:** Always set `u_init` close to the Dirichlet boundary values.
> The knowledge graph will automatically warn you with `INCONSISTENT_IC` if
> a large temperature jump is detected.

### Material Parameters (seeded in Knowledge Graph)

| Material | k [W/(m·K)] | ρ [kg/m³] | c_p [J/(kg·K)] | α [m²/s] |
|----------|-------------|-----------|----------------|----------|
| Steel (carbon) | 50 | 7800 | 500 | 1.28e-5 |
| Stainless Steel 316 | 16 | 8000 | 500 | 4.00e-6 |
| Aluminium 6061 | 200 | 2700 | 900 | 8.23e-5 |
| Copper | 385 | 8960 | 385 | 1.12e-4 |
| Titanium Ti-6Al-4V | 6.7 | 4430 | 526 | 2.88e-6 |
| Silicon | 150 | 2330 | 700 | 9.22e-5 |
| Concrete | 1.7 | 2300 | 880 | 8.40e-7 |
| Glass (borosilicate) | 1.0 | 2230 | 830 | 5.40e-7 |
| Water | 0.6 | 1000 | 4182 | 1.43e-7 |
| Air | 0.026 | 1.2 | 1005 | 2.16e-5 |

---

## Knowledge Graph — GraphRAG Features

The Neo4j knowledge graph is the system's long-term physics memory. Each simulation run is stored as a graph node and connected to typed context nodes, enabling agents to reason with both accumulated experience and curated domain knowledge.

### Graph Schema

```
(:Run {
    run_id, name, dim, status, k, rho, cp,
    nx, ny, nz, Lx, Ly, Lz, bc_types,
    t_end, dt, theta, source, u_init,
    t_max, t_min, t_mean, l2_norm,
    wall_time, n_dofs, created_at,
    embedding   ← 768-dim nomic-embed-text vector
})

(:Material { name, k, rho, cp, alpha, k_min, k_max, description, typical_uses })
(:KnownIssue { code, severity, condition, description, recommendation, observed_in })
(:BCConfig { pattern, description, has_dirichlet, has_neumann, has_robin, has_source })
(:Domain { label, description, Lx_ref, Ly_ref, char_len })
(:ThermalClass { name, description, k_threshold })
(:Reference { ref_id, type, subject, text, source, tags })

Relationships:
  (:Run)-[:USES_MATERIAL {confidence}]->(:Material)
  (:Run)-[:TRIGGERED {detected_at}]->(:KnownIssue)
  (:Run)-[:USES_BC_CONFIG]->(:BCConfig)
  (:Run)-[:ON_DOMAIN]->(:Domain)
  (:Material)-[:HAS_THERMAL_CLASS]->(:ThermalClass)
  (:Run)-[:SPAWNED_FROM]->(:Run)            — created from an agent suggestion
  (:Run)-[:SIMILAR_TO {score, updated_at}]->(:Run)  — KNN semantic edges
  (:Material)-[:HAS_REFERENCE]->(:Reference)
  (:BCConfig)-[:HAS_REFERENCE]->(:Reference)
  (:Domain)-[:HAS_REFERENCE]->(:Reference)
```

### Feature 1 — Vector Embeddings + Semantic Search

Every Run node stores a 768-dimensional `embedding` property computed by `nomic-embed-text` (via Ollama). The embedding text summarises the simulation's physics: geometry, material class, BC types, thermal diffusivity, temperature range, and DOFs.

A **Neo4j HNSW vector index** (`run_embedding_index`) enables sub-millisecond semantic similarity search.

`get_similar_runs()` automatically uses vector search as the primary strategy and falls back to Cypher parameter-distance search when Ollama is unavailable.

```python
# Semantic search from a natural-language query
kg.get_similar_runs_semantic(config, top_k=5)
# → [{run_id, k, bc_pattern, domain_label, t_max, similarity_score: 0.9656}, ...]

# Backfill all existing runs
kg.backfill_embeddings(batch_size=50)
# → {total_unembedded: 0, newly_embedded: 313, failed: 0}
```

### Feature 2 — SIMILAR_TO KNN Edges

After every run, the system creates `SIMILAR_TO` edges to the top-5 nearest embedded neighbours (cosine ≥ 0.85). This precomputes the neighbourhood graph so agents can traverse it with a simple Cypher hop rather than a vector query:

```cypher
MATCH (r:Run {run_id: $id})-[rel:SIMILAR_TO]->(nb:Run)
RETURN nb.run_id, rel.score ORDER BY rel.score DESC
```

These edges are also shown in the **Run Inspector** panel on the Overview tab.

### Feature 3 — Physics Reference Nodes

20 curated reference facts are stored as `Reference` nodes, linked to Material, BCConfig, and Domain nodes:

| Type | Count | Examples |
|------|-------|---------|
| `material_property` | 8 | Steel k(T) dependence, Curie point anomaly in cp, copper melting, silicon k drop, water convection limit |
| `bc_practice` | 5 | h = 5–25 W/(m²·K) natural air convection, h = 500–10,000 for water cooling, typical heat flux ranges |
| `solver_guidance` | 4 | 10 elements minimum across thinnest dimension, Fourier accuracy criterion, P2 vs P1 trade-offs, CG vs GMRES |
| `domain_physics` | 3 | Radiation negligible at micro-scale, natural convection at structural scale, thermal time constant formula |

References appear in `check_config_warnings()`, `get_pre_run_context()`, and the `get_physics_references` agent tool. Agents cite the `source` field when presenting reference data.

---

## Project Structure

```
pde-agents/
├── agents/
│   ├── base_agent.py         # LangGraph ReAct base: reason → act loop + tool-call parser
│   ├── simulation_agent.py   # Agent-1: KG-aware setup, run, debug FEM simulations
│   ├── analytics_agent.py    # Agent-2: analysis, comparison, KG pattern queries
│   └── database_agent.py     # Agent-3: DB storage, SQL queries, history search
│
├── tools/                    # LangChain @tool functions (agent "hands")
│   ├── simulation_tools.py   # run_simulation, validate_config, run_parametric_sweep
│   ├── analytics_tools.py    # analyze_run, compare_runs, list_runs_for_analysis
│   ├── database_tools.py     # search_history, get_run_summary, query_runs
│   └── knowledge_tools.py    # check_config_warnings, query_knowledge_graph,
│                             #   get_physics_references (NEW)
│
├── knowledge_graph/
│   ├── graph.py              # SimulationKnowledgeGraph: full schema, vector index,
│   │                         #   semantic search, SIMILAR_TO edges, Reference queries,
│   │                         #   backfill_embeddings, build_all_similar_to_edges
│   ├── embeddings.py         # OllamaEmbedder: nomic-embed-text 768-dim, run_to_text()
│   ├── references.py         # 20 curated physics reference entries (material_property,
│   │                         #   bc_practice, solver_guidance, domain_physics)
│   ├── rules.py              # Pure-Python rule engine (9 rules: IC, CFL, mesh, BCs…)
│   └── seeder.py             # Static knowledge: 10 materials + 6 failure patterns
│
├── orchestrator/
│   ├── graph.py              # Multi-agent supervisor graph (LangGraph)
│   └── api.py                # FastAPI REST interface + /kg/* endpoints
│
├── simulations/
│   ├── solvers/
│   │   └── heat_equation.py  # DOLFINx 0.10 FEM solver (2D/3D, all BC types,
│   │                         #   built-in + Gmsh mesh dispatch)
│   ├── geometry/
│   │   ├── __init__.py
│   │   └── gmsh_geometries.py  # 9 Gmsh geometry builders (rectangle, l_shape, circle,
│   │                           #   annulus, hollow_rectangle, t_shape, stepped_notch,
│   │                           #   box, cylinder) + GmshMeshResult dataclass
│   └── configs/
│       ├── heat_2d.json      # Steel plate example config
│       └── heat_3d.json      # Aluminum block with convection config
│
├── database/
│   ├── models.py             # SQLAlchemy ORM (SimulationRun, AgentRunLog, …)
│   ├── operations.py         # CRUD, log_agent_step, search_runs, get_agent_logs
│   └── init.sql              # PostgreSQL init (extensions, permissions)
│
├── visualization/
│   └── dashboard.py          # Plotly Dash: Overview, Field Viewer, Convergence,
│                             #   Parametric, Agent Chat, 🧠 Knowledge Graph (NEW),
│                             #   🔎 Run Explorer
│
├── scripts/
│   ├── migrate_kg_schema_v2.py   # One-off: add BCConfig/Domain/ThermalClass to existing runs
│   ├── seed_knowledge_graph.py   # 23 representative simulations for KG bootstrapping
│   ├── seed_bc_geometry_study.py # BC + geometry parametric study
│   └── ollama-init.sh            # Pull all required Ollama models
│
├── docker/
│   ├── Dockerfile.fenics     # dolfinx/dolfinx:stable + Gmsh + uvicorn API
│   ├── Dockerfile.agents     # Python 3.11 + LangGraph + FastAPI + neo4j driver
│   ├── Dockerfile.dashboard  # Python 3.11 + Dash + pandas + neo4j driver
│   └── fenics_runner_api.py  # FastAPI server inside the FEniCSx container
│
├── nginx/
│   └── nginx.conf            # Nginx reverse proxy (subpath routing, WebSocket tunnelling)
├── docker-compose.yml        # Full service stack incl. NeoDash on port 5005
├── env.example               # Environment variable template (copy to .env)
├── requirements.txt          # Python dependencies for agents container
├── Makefile                  # Common commands
└── setup.sh                  # One-time setup script
```

---

## LLM Models

| Model | Params | VRAM | Role |
|-------|--------|------|------|
| `qwen2.5-coder:14b` | 14B | ~9 GB  | Database Agent — fast structured queries |
| `qwen2.5-coder:32b` | 32B | ~19 GB | Simulation Agent — code generation & debugging |
| `llama3.3:70b`      | 70B | ~42 GB | Analytics Agent + Orchestrator — reasoning |
| `nomic-embed-text`  | 137M | <1 GB | Embedding model — 768-dim semantic vectors |

All LLMs fit simultaneously (~70 GB out of 196 GB available VRAM). The embedding model adds negligible overhead.

**Tool-calling compatibility:** `qwen2.5-coder` models output tool calls as JSON text inside the `content` field. The base agent includes a `_parse_content_tool_call()` fallback that normalizes this automatically.

```bash
# Pull all required models (including embedding model)
make pull-models

# Optional: use alternative models — edit .env:
SIM_MODEL=qwen2.5-coder:72b      # ~40 GB, premium code generation
ANALYTICS_MODEL=deepseek-r1:70b  # strong chain-of-thought reasoning
EMBED_MODEL=nomic-embed-text      # default embedding model (274 MB)
```

---

## Makefile Reference

```bash
make setup           # one-time setup (build images, start services, pull models)
make run             # start all services
make stop            # stop all services
make infra           # start only postgres, redis, minio, neo4j, ollama

make simulate-2d     # run 2D steel plate benchmark via agents API
make simulate-3d     # run 3D aluminum block benchmark via agents API
make sweep           # parametric sweep: k=[1..200], analyze, find optimal
make analyze RUN_ID=<id>            # analyze a specific run
make query Q="<natural language>"   # database natural-language query

make pull-models            # pull all required LLM + embedding models
make list-models            # list available Ollama models

make db-init         # initialize database tables
make db-shell        # psql shell into pde_simulations
make db-stats        # show run count and avg wall time by status

make shell-fenics    # bash inside fenics-runner container
make shell-agents    # bash inside agents container
make logs            # tail all service logs
make health          # check all service HTTP endpoints

make test-solver     # quick FEniCSx solver smoke test (8×8 mesh)
make clean           # ⚠ stop, DELETE all volumes, wipe results/
```

---

## Troubleshooting

### Knowledge Graph tab shows "Knowledge graph unavailable"

The dashboard needs the `neo4j` Python driver and access to `knowledge_graph/`. After a fresh deploy:

```bash
# Option A: recreate the container (picks up updated docker-compose.yml)
docker compose up dashboard -d --force-recreate

# Option B: install driver into running container directly
docker exec pde-dashboard pip install neo4j

# Verify
docker exec pde-dashboard python3 -c "
import sys; sys.path.insert(0, '/app')
from knowledge_graph.graph import get_kg
kg = get_kg()
print('Available:', kg.available)
print('Stats:', kg.stats())
"
```

### A nav link opens a blank page or 502 Bad Gateway

1. Check the target container is running: `docker compose ps`
2. Reload nginx: `docker compose exec nginx nginx -s reload`
3. Check nginx logs: `docker compose logs nginx --tail=30`

### NeoDash not reachable at port 9001

```bash
docker compose up neodash -d
docker logs pde-neodash --tail=20
```
NeoDash connects to Neo4j internally on the `pde-net` network. Port 9001 on the host maps to NeoDash (the MinIO console was moved to port 9002).

### MinIO console access

The MinIO console is on host port 9002 (port 9001 is now NeoDash). Access via SSH tunnel:
```bash
ssh -L 19002:localhost:9002 -N <server>   # then open http://localhost:19002
```
The MinIO S3 API on port 9000 is unaffected — agents use `http://minio:9000` internally.

### Embedding model not available (semantic search returns empty)

```bash
# Pull nomic-embed-text into Ollama
docker exec pde-ollama ollama pull nomic-embed-text

# Backfill embeddings for existing runs
docker exec pde-agents python3 -c "
import sys; sys.path.insert(0, '/app')
from knowledge_graph.graph import get_kg
kg = get_kg()
print(kg.backfill_embeddings(batch_size=50))
print(kg.build_all_similar_to_edges(k=5, min_score=0.85))
"
```

### Knowledge graph is empty after restart

The KG is seeded automatically when the agents container starts. Trigger manually if empty:
```bash
# Seed static knowledge (materials + failure patterns + references)
curl -s -X POST http://localhost:8000/kg/seed | python3 -m json.tool

# Re-run representative simulations to rebuild Run nodes
docker exec pde-agents python3 /app/scripts/seed_knowledge_graph.py

# Backfill embeddings and KNN edges for all runs
docker exec pde-agents python3 -c "
import sys; sys.path.insert(0, '/app')
from knowledge_graph.graph import get_kg
kg = get_kg()
kg.backfill_embeddings()
kg.build_all_similar_to_edges()
print(kg.stats())
"
```

### FEniCSx API returns 500 errors
```bash
docker exec pde-fenics find /workspace/simulations -name "*.pyc" -delete
docker compose restart fenics-runner
```

### Negative temperatures in early time steps
Set `u_init` close to the lower Dirichlet BC value. The `check_config_warnings` tool will catch this with the `INCONSISTENT_IC` rule.

### Tool calls not executing (agent loops without running tools)
Check that the tool name in the LLM's JSON output matches the registered tool name exactly (case-sensitive). The parser is in `agents/base_agent.py` → `_parse_content_tool_call`.

---

## Data Persistence and Volume Management

### What persists across restarts

Normal stop/start operations preserve all data — Docker named volumes are kept on disk.

```bash
docker compose stop && docker compose start   # safe, all data preserved
docker compose down && docker compose up -d   # safe — no -v flag
```

### What a clean wipe removes

```bash
docker compose down -v   # or: make clean
```

| Volume | Contents lost |
|--------|--------------|
| `neo4j_data` | All Run nodes, embeddings, SIMILAR_TO edges, Reference links |
| `postgres_data` | All simulation records, agent logs |
| `minio_data` | All uploaded XDMF/NPY result files |
| `redis_data` | Any queued jobs |

### Rebuilding after a clean wipe

```bash
# 1. Start all services
docker compose up -d

# 2. Re-seed static knowledge + references
curl -s -X POST http://localhost:8000/kg/seed

# 3. Re-run representative simulations
docker exec pde-agents python3 /app/scripts/seed_knowledge_graph.py

# 4. Pull embedding model and backfill
docker exec pde-ollama ollama pull nomic-embed-text
docker exec pde-agents python3 -c "
import sys; sys.path.insert(0, '/app')
from knowledge_graph.graph import get_kg
kg = get_kg()
kg.backfill_embeddings()
kg.build_all_similar_to_edges()
"
```

---

## Extending to Other PDEs

To add a new PDE solver:

1. Create `simulations/solvers/my_pde.py` with the same interface as `heat_equation.py`
2. Add corresponding tools in `tools/simulation_tools.py`
3. Register the new tool in the relevant agent's tool list
4. Update the Simulation Agent's system prompt with PDE-specific guidance
5. Add a visualization tab in `visualization/dashboard.py`
6. Extend `knowledge_graph/seeder.py` and `knowledge_graph/references.py` with PDE-specific failure patterns and physics knowledge

**Planned extensions:**
- **Poisson equation** — electrostatics, groundwater pressure
- **Linear elasticity** — structural mechanics, thermal stress
- **Navier-Stokes** — incompressible laminar flow
- **Coupled thermo-mechanical** — heat + stress

---

## Integration with NVIDIA PhysicsNemo

The existing `physicsnemo-25_11` container (at `~/physicsnemo-work`) can run
alongside this system for hybrid FEM + PINN workflows:

| Use case | Approach |
|----------|----------|
| Validation | Run FEniCSx (ground truth) + PhysicsNemo (PINN), compare fields |
| Surrogate models | Generate FEM training data → train PhysicsNemo surrogate |
| Fast parametric scans | FEM at sparse reference points, PINN to interpolate |
| KG-guided training | Use KG similar-run search to find best FEM training seeds |

To connect PhysicsNemo to the same Docker network and shared storage:
```yaml
# Add to docker-compose.yml under services:
physicsnemo:
  image: nvcr.io/nvidia/physicsnemo/physicsnemo:25.11
  volumes:
    - simulation_results:/workspace/fem
  networks: [pde-net]
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]
```
