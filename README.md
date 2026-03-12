# PDE Agents

A multi-agent ecosystem built on open-source LLMs running locally to solve PDEs with the Finite Element Method.

**Hardware:** 2× NVIDIA RTX PRO 6000 Blackwell Server Edition (~98 GB VRAM each · ~196 GB total) · CUDA 13.1  
**FEM Solver:** DOLFINx (FEniCSx) `0.10.0.post2`

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR (LangGraph)                      │
│              LLM: llama3.3:70b  (supervisor)                    │
└───────────┬───────────────┬────────────────┬────────────────────┘
            │               │                │
            ▼               ▼                ▼
  ┌─────────────────┐ ┌──────────────┐ ┌───────────────┐
  │  AGENT-1        │ │  AGENT-2     │ │  AGENT-3      │
  │  Simulation     │ │  Analytics   │ │  Database     │
  │  ─────────────  │ │  ──────────  │ │  ──────────   │
  │  qwen2.5-coder  │ │  llama3.3    │ │  qwen2.5-coder│
  │  :32b           │ │  :70b        │ │  :14b         │
  └────────┬────────┘ └──────┬───────┘ └──────┬────────┘
           │                 │                │
           │    ┌────────────┴────────────┐   │
           ▼    ▼                         ▼   ▼
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │  FEniCSx     │  │  NumPy       │  │  PostgreSQL  │
  │  DOLFINx     │  │  Plotly Dash │  │  MinIO       │
  │  2D/3D FEM   │  │  MLflow      │  │  (XDMF/VTK)  │
  └──────────────┘  └──────────────┘  └──────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Neo4j           │
                    │  Knowledge Graph │
                    │  (materials +    │
                    │   learned runs)  │
                    └──────────────────┘
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
Only port 8050 needs to be open in a firewall for full dashboard access.

| Service | Host access | Purpose |
|---------|-------------|---------|
| **nginx** | `:8050` | Reverse proxy — single entry point for all web UIs |
| Dashboard | `:8050/` (via nginx) | Plotly Dash interactive visualization & chat |
| Agents API | `:8050/agents/` or `:8000` | FastAPI REST + Swagger UI |
| MLflow | `:8050/mlflow/` (via nginx) | Experiment tracking UI |
| Neo4j Browser | `:8050/browser/` (via nginx) | Knowledge graph browser |
| MinIO Console | `:9001` (direct) | Object storage browser — see note below |
| MinIO API | `:9000` | S3-compatible object storage API |
| FEniCSx Runner | `:8080` | DOLFINx simulation job REST API |
| Ollama | `:11434` | Local LLM server (GPU-accelerated) |
| PostgreSQL | `:5432` | Simulation metadata database |
| Redis | `:6379` | Message broker for agent coordination |
| Neo4j Bolt | `:7687` | Graph database driver protocol |

> **MinIO console** (`port 9001`) is accessed directly rather than through nginx
> because its SPA uses a hardcoded `<base href="/">` that prevents subpath proxying.
> If port 9001 is firewalled, use an SSH tunnel:
> ```bash
> ssh -L 19001:localhost:9001 -N <server>   # then open http://localhost:19001
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
- Start all services including Neo4j
- Initialize the PostgreSQL schema
- Auto-seed the knowledge graph with 10 engineering materials and 6 known failure patterns
- Pull LLM models in the background (~70 GB total, 30–90 min first time)

### 2. Confirm all models are ready
```bash
docker exec pde-ollama ollama list
# Expected output once all pulls complete:
# NAME                  SIZE
# llama3.3:70b          42 GB
# qwen2.5-coder:32b     19 GB
# qwen2.5-coder:14b     9.0 GB
```

### 3. Check all services are healthy
```bash
make health
```

### 4. Open the dashboard
Navigate to **http://localhost:8050** in your browser.

All services are accessible from the dashboard navbar or directly:

| URL | What you get |
|-----|-------------|
| http://localhost:8050/ | Dashboard |
| http://localhost:8050/agents/docs | API Swagger UI |
| http://localhost:8050/mlflow/ | MLflow |
| http://localhost:8050/browser/ | Neo4j Browser |
| http://localhost:9001 | MinIO console (direct) |

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

**Via Makefile:**
```bash
make simulate-2d
```

After every successful run the system automatically:
1. Stores metadata in **PostgreSQL**
2. Uploads output files to **MinIO** (`simulation-results/runs/<run_id>/`)
3. Adds a **Run node** to the **Neo4j knowledge graph** (with material inference and rule warnings)

---

### Use Case 2 — 3D Heat Equation (Aluminum Block with Convection)

A 3D aluminum block with fixed-temperature left/right walls, convective Robin
boundaries on the front and back, and insulated top/bottom. Demonstrates all three
boundary condition types simultaneously.

**Via Agents API:**
```bash
curl -s -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Run a 3D heat equation on a 16x16x16 aluminum block. k=237 W/(mK), rho=2700, cp=900. Left wall 273K, right wall 373K. Front and back surfaces have Robin convection: h=25 W/(m2K), T_ambient=293K. Top and bottom insulated. Use u_init=293.0, t_end=300.0, dt=2.0. Report min/max temperature and steady-state estimate."
  }' | python3 -m json.tool
```

**Direct FEniCSx REST API:**
```bash
curl -s -X POST http://localhost:8080/run \
  -H "Content-Type: application/json" \
  -d '{
    "dim": 3, "nx": 16, "ny": 16, "nz": 16,
    "k": 237.0, "rho": 2700.0, "cp": 900.0,
    "u_init": 293.0,
    "bcs": [
      {"type": "dirichlet", "value": 273.0, "location": "left"},
      {"type": "dirichlet", "value": 373.0, "location": "right"},
      {"type": "robin", "alpha": 25.0, "u_inf": 293.0, "location": "front"},
      {"type": "robin", "alpha": 25.0, "u_inf": 293.0, "location": "back"},
      {"type": "neumann", "value": 0.0, "location": "top"},
      {"type": "neumann", "value": 0.0, "location": "bottom"}
    ],
    "t_end": 300.0, "dt": 2.0,
    "run_id": "aluminum_block_3d",
    "output_dir": "/workspace/results"
  }' | python3 -m json.tool
```

**Via Makefile:**
```bash
make simulate-3d
```

---

### Use Case 3 — Parametric Sweep (Thermal Conductivity Study)

The Simulation Agent runs multiple simulations while varying a single parameter,
automatically cataloging each run in the database.

**Via Agents API:**
```bash
curl -s -X POST http://localhost:8000/agent/simulation \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Run a parametric sweep on a 2D 32x32 domain varying thermal conductivity k over [1, 10, 50, 100, 200] W/(mK). Keep rho=7800, cp=500, left wall 300K, right wall 500K, insulated top and bottom, u_init=300.0, t_end=20.0, dt=0.5. Use run_ids: sweep_k_001 through sweep_k_005."
  }' | python3 -m json.tool
```

**Via Makefile:**
```bash
make sweep
```

---

### Use Case 4 — Analytics: Compare Runs

After running multiple simulations, the Analytics Agent cross-references results
and suggests what to try next.

```bash
curl -s -X POST http://localhost:8000/agent/analytics \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Compare the runs steel_plate_64x64 and steel_agent_final. What is the difference in temperature distribution? Which mesh resolution is sufficient and why? Suggest the next simulation parameter set to test."
  }' | python3 -m json.tool
```

**Via Makefile:**
```bash
make analyze RUN_ID=steel_plate_64x64
```

---

### Use Case 5 — Database Queries in Natural Language

The Database Agent translates natural-language questions into SQL queries and can
also search the knowledge graph for material or pattern information.

```bash
# Find the hottest simulations
curl -s -X POST http://localhost:8000/agent/database \
  -H "Content-Type: application/json" \
  -d '{"task": "Show me all runs where the maximum temperature exceeded 450K, ordered by wall time."}' \
  | python3 -m json.tool

# History query through the knowledge graph
curl -s -X POST http://localhost:8000/agent/database \
  -H "Content-Type: application/json" \
  -d '{"task": "What material properties does copper have? Find similar past runs that used copper-like conductivity."}' \
  | python3 -m json.tool

# Get statistics
curl -s -X POST http://localhost:8000/agent/database \
  -H "Content-Type: application/json" \
  -d '{"task": "How many simulations have been run in total? What is the average wall time for 2D vs 3D runs?"}' \
  | python3 -m json.tool
```

**Via Makefile:**
```bash
make query Q="Show me all 3D runs with fewer than 10000 DOFs"
make db-stats
```

---

### Use Case 6 — Full Multi-Agent Pipeline

The orchestrator decomposes a high-level task across all three agents: Simulation
runs the FEM, Analytics interprets results, and Database stores and retrieves data.

```bash
curl -s -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Investigate the effect of mesh refinement on the 2D heat equation. Run the steel plate problem (k=50, rho=7800, cp=500, left 300K, right 500K, t_end=20.0, dt=0.5) on three meshes: 16x16, 32x32, and 64x64. Store all results, compare maximum temperatures and wall times, and summarize which mesh gives the best accuracy-to-cost trade-off."
  }' | python3 -m json.tool
```

---

### Use Case 7 — Knowledge Graph Queries

The knowledge graph accumulates information with every run. Query it directly
via the API, or ask any agent in natural language.

```bash
# Graph statistics
curl -s http://localhost:8000/kg/stats | python3 -m json.tool

# Look up a material
curl -s http://localhost:8000/kg/material/titanium | python3 -m json.tool

# Find runs similar to a given run
curl -s http://localhost:8000/kg/run/steel_plate_64x64/similar | python3 -m json.tool

# Re-seed static knowledge (safe to call repeatedly)
curl -s -X POST http://localhost:8000/kg/seed | python3 -m json.tool

# Ask the Simulation Agent to check a config before running
curl -s -X POST http://localhost:8000/agent/simulation \
  -H "Content-Type: application/json" \
  -d '{
    "task": "What issues might arise if I run a 2D simulation with k=50, nx=8, ny=8, theta=0.0, dt=0.5, u_init=0, and Dirichlet BCs at 300 K and 800 K? Use check_config_warnings."
  }' | python3 -m json.tool
```

**Browse the graph visually:** open **http://localhost:8050/browser/** in your browser
(login: `neo4j` / `pde_graph_secret`)

> Neo4j Browser is also accessible at `http://localhost:7474` if that port is open.
> When connecting through the nginx proxy at `/browser/`, the Bolt WebSocket is
> tunnelled via `/neo4j-bolt` on port 8050 — no separate port needed.

Example Cypher queries in the Neo4j Browser:
```cypher
// All runs that triggered the INCONSISTENT_IC warning
MATCH (r:Run)-[:TRIGGERED]->(i:KnownIssue {code: "INCONSISTENT_IC"})
RETURN r.run_id, r.t_max, i.recommendation

// Runs grouped by inferred material
MATCH (r:Run)-[:USES_MATERIAL]->(m:Material)
RETURN m.name, count(r) AS runs, avg(r.t_max) AS avg_t_max
ORDER BY runs DESC

// Find similar runs to a given configuration
MATCH (r:Run)
WHERE r.dim = 2 AND r.k >= 40 AND r.k <= 60
RETURN r.run_id, r.k, r.t_max, r.wall_time
ORDER BY r.k
```

---

### Use Case 8 — Run Explorer (Dashboard)

The **🔎 Run Explorer** tab in the dashboard (`http://localhost:8050/`) provides a
visual interface to browse every simulation run:

- **Left panel:** searchable, filterable run list (status, dimension, keyword)
- **Right panel:** detailed view with sub-tabs for:
  - **Overview** — KPIs (T_max, T_min, wall time, DOFs)
  - **Agent Timeline** — step-by-step trace of every agent reasoning and tool-call
  - **Config** — full simulation configuration used (for reproducibility)
  - **Files** — MinIO file listing with object names and sizes
  - **Recommendations** — agent suggestions from the run

Every run stores its agent decision trace in the `agent_run_logs` table so you
can replay exactly what the LLM reasoned and which tools it called.

---

### Use Case 9 — Direct Python (inside containers)

#### Run the solver directly
```python
# docker exec -it pde-fenics python3
import sys; sys.path.insert(0, '/workspace')
from simulations.solvers.heat_equation import HeatConfig, HeatEquationSolver

cfg = HeatConfig(
    dim=2, nx=128, ny=128,
    k=1.5, rho=2500.0, cp=800.0,
    source=5000.0,          # internal heat generation [W/m³]
    u_init=293.0,
    bcs=[
        {"type": "dirichlet", "value": 293.0, "location": "left"},
        {"type": "dirichlet", "value": 293.0, "location": "right"},
        {"type": "neumann",   "value": 0.0,   "location": "top"},
        {"type": "neumann",   "value": 0.0,   "location": "bottom"},
    ],
    t_end=200.0, dt=1.0,
    run_id="custom_run_001",
    output_dir="/workspace/results",
)
result = HeatEquationSolver(cfg).solve()
print(result)
```

#### Query the knowledge graph directly
```python
# docker exec -it pde-agents python3
import sys; sys.path.insert(0, '/app')
from knowledge_graph.graph import get_kg

kg = get_kg()
print("Stats:", kg.stats())
print("Copper:", kg.get_material_info("copper"))

# Pre-run context for a proposed config
ctx = kg.get_pre_run_context({
    "dim": 2, "k": 50, "rho": 7800, "cp": 500,
    "nx": 64, "ny": 64, "dt": 0.5, "t_end": 100,
    "theta": 1.0, "u_init": 300,
    "bcs": [{"type": "dirichlet", "value": 300, "location": "left"},
            {"type": "dirichlet", "value": 500, "location": "right"}]
})
print("Warnings:", ctx["warnings"])
print("Similar runs:", ctx["similar_runs"])
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
print(f"Iterations: {result['iterations']}")
```

#### Run the multi-agent orchestrator
```python
# docker exec -it pde-agents python3
import sys; sys.path.insert(0, '/app')
from orchestrator.graph import MultiAgentOrchestrator

orch = MultiAgentOrchestrator()
result = orch.run(
    "Investigate how thermal conductivity affects heat distribution "
    "in a 2D square domain. Sweep k from 1 to 200 W/(m·K) in 4 steps, "
    "analyze the results, and recommend the optimal k for maximum uniformity."
)
print(result["final_report"])
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

**Time integration schemes:**
- θ = 1.0 — Backward Euler: unconditionally stable, 1st-order accurate *(default)*
- θ = 0.5 — Crank-Nicolson: 2nd-order accurate, conditionally stable

### Configuration Reference

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `dim` | int | Spatial dimension (2 or 3) | `2` |
| `nx`, `ny`, `nz` | int | Mesh divisions per axis | `64, 64` |
| `k` | float | Thermal conductivity [W/(m·K)] | `50.0` (steel) |
| `rho` | float | Density [kg/m³] | `7800.0` (steel) |
| `cp` | float | Specific heat [J/(kg·K)] | `500.0` (steel) |
| `source` | float | Volumetric heat source [W/m³] | `0.0` |
| `u_init` | float | Initial temperature [K] — set close to BCs | `300.0` |
| `t_end` | float | Simulation end time [s] | `100.0` |
| `dt` | float | Time step [s] | `0.5` |
| `theta` | float | Time integration: 1.0=BE, 0.5=CN | `1.0` |
| `bcs` | list | Boundary condition specs (see below) | — |
| `run_id` | str | Unique run identifier | `"steel_plate_01"` |
| `output_dir` | str | Directory for XDMF/VTK output | `"/workspace/results"` |

**Boundary condition spec:**
```json
{"type": "dirichlet", "value": 300.0, "location": "left"}
{"type": "neumann",   "value": 0.0,   "location": "top"}
{"type": "robin",     "alpha": 10.0,  "u_inf": 293.0, "location": "front"}
```
Locations for 2D: `left`, `right`, `top`, `bottom`.  
Locations for 3D: `left`, `right`, `top`, `bottom`, `front`, `back`.

> **Important:** Always set `u_init` close to the Dirichlet boundary values.
> Starting from `u_init=0.0` with walls at 300–500 K causes numerical overshoot
> (Gibbs-like oscillation) in the first few time steps. The knowledge graph will
> automatically warn you if this inconsistency is detected.

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

Thermal diffusivity `α = k / (ρ c_p)`. Characteristic time `τ = L² / α`.  
All materials are stored in Neo4j and agents can look them up by name (e.g. "what are the properties of titanium?").

---

## Project Structure

```
pde-agents/
├── agents/
│   ├── base_agent.py         # LangGraph ReAct base: reason → act loop + tool-call parser
│   │                         #   + step-by-step agent_run_logs instrumentation
│   ├── simulation_agent.py   # Agent-1: KG-aware setup, run, debug FEM simulations
│   ├── analytics_agent.py    # Agent-2: analysis, comparison, KG pattern queries
│   └── database_agent.py     # Agent-3: DB storage, SQL queries, history search
│
├── tools/                    # LangChain @tool functions (agent "hands")
│   ├── simulation_tools.py   # run_simulation (auto MinIO + KG), validate_config,
│   │                         #   debug_simulation, run_parametric_sweep
│   ├── analytics_tools.py    # analyze_run, compare_runs, list_runs_for_analysis
│   ├── database_tools.py     # search_history, get_run_summary, query_runs,
│   │                         #   store_result, export_to_csv
│   └── knowledge_tools.py    # check_config_warnings, query_knowledge_graph
│
├── knowledge_graph/          # Phase-1 knowledge graph (Neo4j)
│   ├── __init__.py
│   ├── graph.py              # SimulationKnowledgeGraph: add_run, get_similar_runs,
│   │                         #   get_pre_run_context, get_material_info, get_run_lineage
│   ├── rules.py              # Pure-Python rule engine (8 rules: IC, CFL, mesh, BCs…)
│   └── seeder.py             # Static knowledge: 10 materials + 6 failure patterns
│
├── orchestrator/
│   ├── graph.py              # Multi-agent supervisor graph (LangGraph)
│   └── api.py                # FastAPI REST interface + WebSocket streaming
│                             #   /explorer/* endpoints (Run Explorer)
│                             #   /kg/* endpoints (knowledge graph REST)
│
├── simulations/
│   ├── solvers/
│   │   └── heat_equation.py  # DOLFINx 0.10 FEM solver (2D/3D, all BC types)
│   └── configs/
│       ├── heat_2d.json      # Steel plate example config
│       └── heat_3d.json      # Aluminum block with convection config
│
├── database/
│   ├── models.py             # SQLAlchemy ORM (SimulationRun, RunResult,
│   │                         #   AgentRunLog, AgentSuggestion, ParametricStudy…)
│   ├── operations.py         # CRUD, log_agent_step, search_runs, get_agent_logs
│   └── init.sql              # PostgreSQL init (extensions, permissions)
│
├── visualization/
│   └── dashboard.py          # Plotly Dash: field viewer, convergence, parametric,
│                             #   🔎 Run Explorer tab, 🤖 Agent Chat (Enter-key + quick prompts)
│
├── docker/
│   ├── Dockerfile.fenics     # dolfinx/dolfinx:stable + uvicorn API
│   ├── Dockerfile.agents     # Python 3.11 + LangGraph + FastAPI + neo4j driver
│   ├── Dockerfile.dashboard  # Python 3.11 + Dash + pandas
│   └── fenics_runner_api.py  # FastAPI server inside the FEniCSx container
│
├── scripts/
│   └── ollama-init.sh        # Model pull script (uses ollama CLI only, no curl)
│
├── nginx/
│   └── nginx.conf            # Nginx reverse proxy config (subpath routing for all UIs)
├── docker-compose.yml        # Full service stack orchestration (includes nginx + Neo4j)
├── env.example               # Environment variable template (copy to .env)
├── requirements.txt          # Python dependencies for agents container
├── Makefile                  # Common commands (simulate, sweep, query, logs, etc.)
└── setup.sh                  # One-time setup script
```

---

## LLM Models

All three models are pulled automatically by `setup.sh` and confirmed ready once
`docker exec pde-ollama ollama list` shows all three entries.

| Model | Params | VRAM | Role |
|-------|--------|------|------|
| `qwen2.5-coder:14b` | 14B | ~9 GB  | Database Agent — fast structured queries |
| `qwen2.5-coder:32b` | 32B | ~19 GB | Simulation Agent — code generation & debugging |
| `llama3.3:70b`      | 70B | ~42 GB | Analytics Agent + Orchestrator — reasoning |

All three fit simultaneously (~70 GB out of 196 GB available VRAM).

**Tool-calling compatibility note:** `qwen2.5-coder` models output tool calls as
JSON text inside the `content` field rather than in the structured `tool_calls`
field. The base agent (`base_agent.py`) includes a `_parse_content_tool_call()`
fallback that detects and normalizes this automatically — the parser handles plain
JSON, markdown code fences, and JSON embedded after preamble text.

```bash
# Check download progress
docker exec pde-ollama ollama list

# Pull all required models manually if needed
make pull-models

# Optional: use larger/alternative models — edit .env:
SIM_MODEL=qwen2.5-coder:72b      # ~40 GB, premium code generation
ANALYTICS_MODEL=deepseek-r1:70b  # strong chain-of-thought reasoning
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

make pull-models            # pull all required LLM models into Ollama
make list-models            # list available Ollama models
make pull-model MODEL=<name>        # pull a specific model

make db-init         # initialize database tables
make db-shell        # psql shell into pde_simulations
make db-stats        # show run count and avg wall time by status

make shell-fenics    # bash inside fenics-runner container
make shell-agents    # bash inside agents container
make logs            # tail all service logs
make logs-agents     # tail agents container logs
make logs-fenics     # tail fenics-runner logs
make health          # check all service HTTP endpoints

make test-solver     # quick FEniCSx solver smoke test (8×8 mesh)
make clean           # ⚠ stop, DELETE all volumes, wipe results/ — see Data Persistence section
```

---

## Troubleshooting

### A nav link opens a blank page or 502 Bad Gateway

All web UIs are proxied through nginx. If a link gives 502:
1. Check the target container is running: `docker compose ps`
2. Reload nginx config: `docker compose exec nginx nginx -s reload`
3. Check nginx logs: `docker compose logs nginx --tail=30`

If the **MinIO** link does not work, the server's port 9001 may be firewalled.
Use an SSH tunnel from your local machine:
```bash
ssh -L 19001:localhost:9001 -N <server>   # use any free local port
# then open http://localhost:19001
```

### NVIDIA Container Toolkit warning during setup
```bash
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```
Without it, Ollama and FEniCSx containers fall back to CPU.

### Database init fails during `./setup.sh`
Run it manually after all services are up:
```bash
make db-init
```

### Ollama shows as `(unhealthy)` after startup
The healthcheck uses `ollama list` (not curl, which is absent in the Ollama image).
If still unhealthy after ~60 seconds:
```bash
docker compose restart ollama
docker exec pde-ollama ollama list   # confirm server is responding
```

### Agents container stays in `Created` state
The agents service depends on `ollama`, `postgres`, `redis`, and `neo4j` all being
`healthy`. Wait for all healthchecks to pass (~40 s for Neo4j), then bring up agents:
```bash
docker compose up -d neo4j
sleep 45
docker compose up -d agents
```

### Neo4j shows `(unhealthy)` on first start
Neo4j takes 30–40 seconds to initialize its store on first boot. The healthcheck
runs `neo4j status` — wait for it to pass before the agents container tries to connect.
The agents service gracefully handles a temporarily unavailable Neo4j (it logs a warning
and continues without KG features until Neo4j is reachable).

### Knowledge graph is empty after restart
The KG is seeded automatically when the agents container starts (5 s after startup).
If it's empty, trigger it manually:
```bash
curl -s -X POST http://localhost:8000/kg/seed | python3 -m json.tool
```

This re-seeds the 10 engineering materials and 6 known failure patterns, but
does **not** restore simulation run history. See below for how to rebuild that.

---

## Data Persistence and Volume Management

### What persists across restarts

Normal stop/start operations (`docker compose stop` / `docker compose start`, or
a server reboot) preserve all data — Docker named volumes are kept on disk.

```bash
# Safe — all data preserved
docker compose stop
docker compose start

# Also safe — volumes untouched
docker compose down        # no -v flag
docker compose up -d
```

### What a clean wipe removes

Running `docker compose down -v` (or `make clean`) deletes all named volumes:

| Volume | Contents lost |
|--------|--------------|
| `neo4j_data` | All Run nodes, `USES_MATERIAL` and `TRIGGERED` edges |
| `postgres_data` | All simulation records, agent logs, parametric studies |
| `minio_data` | All uploaded XDMF/NPY result files |
| `redis_data` | Any queued jobs |

After a clean wipe and `docker compose up -d`, the KG auto-seeds the static
knowledge (materials + known issues), but all simulation run history is gone.

### Rebuilding the knowledge graph after a clean wipe

A seeding script is provided that bypasses the LLM orchestrator and directly
calls the FEniCSx runner to re-populate the graph with a representative set
of simulations across all 10 materials:

```bash
# Rebuild everything (23 simulations, ~2 minutes)
docker exec pde-agents python3 /app/scripts/seed_knowledge_graph.py

# Preview what it will run without executing
docker exec pde-agents python3 /app/scripts/seed_knowledge_graph.py --dry-run

# Rebuild only specific materials
docker exec pde-agents python3 /app/scripts/seed_knowledge_graph.py --filter steel
docker exec pde-agents python3 /app/scripts/seed_knowledge_graph.py --filter 3d
docker exec pde-agents python3 /app/scripts/seed_knowledge_graph.py --filter rule
```

The script runs 23 simulations covering:
- All 10 materials with auto-calculated stable `dt` and `t_end`
- Steel geometry variants (square, wide, tall domains)
- Robin (convective) boundary conditions on aluminium and steel
- Internal heat source on concrete and silicon
- 3D runs for steel, aluminium, and copper
- 3 deliberate rule violations to create `TRIGGERED` edges in the KG
  (`INCONSISTENT_IC`, `SHORT_SIMULATION`, `LARGE_DT_RELATIVE_TO_DIFFUSION`)

> **Why use this script instead of asking the orchestrator?**
> The LLM orchestrator is designed for open-ended reasoning tasks, not bulk
> data generation. For 20+ simulations it frequently drifts from the task,
> makes JSON formatting errors, and uses ~400 LLM tokens per run. The seed
> script is deterministic and completes in under 2 minutes.

### FEniCSx API returns 500 errors
Stale `.pyc` bytecode files can shadow updated solver code. Clear them and restart:
```bash
docker exec pde-fenics find /workspace/simulations -name "*.pyc" -delete
docker exec pde-fenics find /workspace/simulations -name "__pycache__" -exec rm -rf {} + 2>/dev/null
docker compose restart fenics-runner
```

### Negative temperatures in early time steps
This is a numerical artifact when `u_init` is far from the Dirichlet BC values
(e.g., `u_init=0.0` with walls at 300–500 K). The FEM lifting operation can
cause slight Gibbs-like overshoot in the first few time steps. Fix: always set
`u_init` close to the lower Dirichlet BC value. The `check_config_warnings` tool
will automatically catch this with the `INCONSISTENT_IC` rule.

### Tool calls not executing (agent loops without running tools)
If an agent completes in 1 iteration and returns raw JSON as its answer, the
tool-call parser may have failed to recognize a custom tool name. Check that the
tool name in the LLM's JSON output matches the registered tool name exactly
(case-sensitive). The parser is in `agents/base_agent.py` → `_parse_content_tool_call`.
It handles plain JSON, markdown code fences, and JSON embedded after preamble text.

### Chat agent stuck on history queries
If the agent responds with repeated apology messages without calling tools, the
most likely cause is a tool signature mismatch. The `search_history`, `query_runs`,
and `list_runs_for_analysis` tools use **flat parameters** (not a JSON string argument)
which are easier for LLMs to call correctly. Verify the tool definitions in
`tools/database_tools.py` and `tools/analytics_tools.py`.

### SQLAlchemy `DetachedInstanceError`
Occurs when an ORM object is accessed after its session has closed. The session
factory is configured with `expire_on_commit=False` and `get_run` / `list_runs`
call `.expunge()` before returning. If you write new database operations, always
either expunge objects or serialize them to dicts inside the `with get_db()` block.

### Re-running a simulation with the same `run_id`
`create_run` uses overwrite semantics: if a record with the same `run_id` already
exists it is deleted and re-created. This allows freely re-running experiments
with the same identifier during development.

---

## Extending to Other PDEs

The framework is designed for extensibility. To add a new PDE solver:

1. Create `simulations/solvers/my_pde.py` with the same interface as `heat_equation.py`
   (`HeatConfig`-style dataclass → `MyConfig`, `HeatEquationSolver`-style class → `MySolver`)
2. Add corresponding tools in `tools/simulation_tools.py`
3. Register the new tool in the relevant agent's tool list
4. Update the Simulation Agent's system prompt with PDE-specific guidance
5. Add a visualization tab in `visualization/dashboard.py`
6. Extend `knowledge_graph/seeder.py` with PDE-specific failure patterns

**Planned extensions:**
- **Poisson equation** — electrostatics, groundwater pressure
- **Linear elasticity** — structural mechanics, thermal stress
- **Navier-Stokes** — incompressible laminar flow
- **Coupled thermo-mechanical** — heat + stress (natural next step after heat equation)

---

## Integration with NVIDIA PhysicsNemo

The existing `physicsnemo-25_11` container (at `~/physicsnemo-work`) can run
alongside this system for hybrid FEM + PINN workflows:

| Use case | Approach |
|----------|----------|
| Validation | Run FEniCSx (ground truth) + PhysicsNemo (PINN), compare fields |
| Surrogate models | Generate FEM training data → train PhysicsNemo surrogate |
| Fast parametric scans | FEM at sparse reference points, PINN to interpolate |

To connect PhysicsNemo to the same Docker network and shared storage:
```yaml
# Add to docker-compose.yml under services:
physicsnemo:
  image: nvcr.io/nvidia/physicsnemo/physicsnemo:25.11
  volumes:
    - ./results:/workspace/results      # shared result directory
    - simulation_results:/workspace/fem  # shared Docker volume
  networks: [pde-net]
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]
```
