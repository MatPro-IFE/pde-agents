# PDE Agents — Architecture & Technology Stack

> A complete guide to how the system is built, how each technology works,  
> and how all the layers connect into a working agentic PDE simulation platform.

---

## Table of Contents

1. [Docker & Docker Compose — Infrastructure Glue](#1-docker--docker-compose)
2. [FEniCSx (DOLFINx) — Physics Engine](#2-fenicsxdolfinx--physics-engine)
3. [Ollama — Local LLM Server](#3-ollama--local-llm-server)
4. [LangChain & LangGraph — Agent Framework](#4-langchain--langgraph--agent-framework)
5. [The Three Agents — Specialised Workers](#5-the-three-agents--specialised-workers)
6. [The Orchestrator — Supervisor](#6-the-orchestrator--supervisor)
7. [FastAPI — REST Interface](#7-fastapi--rest-interface)
8. [PostgreSQL + SQLAlchemy — Metadata Database](#8-postgresql--sqlalchemy--metadata-database)
9. [MinIO — Object Storage](#9-minio--object-storage)
10. [Redis — Message Broker](#10-redis--message-broker)
11. [Plotly Dash — Visualization Dashboard](#11-plotly-dash--visualization-dashboard)
12. [MLflow — Experiment Tracking](#12-mlflow--experiment-tracking)
13. [Neo4j — Simulation Knowledge Graph](#13-neo4j--simulation-knowledge-graph)
14. [How Everything Works Together — End to End](#14-how-everything-works-together--end-to-end)
15. [The Four-Layer Architecture](#15-the-four-layer-architecture)

---

## 1. Docker & Docker Compose

### What it is

Docker packages each component into an **isolated container** — its own filesystem,
network stack, and process space. Docker Compose orchestrates all containers as a
single application defined in one `docker-compose.yml` file.

### How the services are arranged

```
                          docker network: pde-net
  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │  ollama  │  │ postgres │  │  redis   │  │  minio   │  │  neo4j   │
  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘
  ┌──────────────┐  ┌──────────────────────────┐  ┌──────────────┐
  │fenics-runner │  │          agents           │  │  dashboard   │
  └──────────────┘  └──────────────────────────┘  └──────────────┘
                              ▲   ▲   ▲   ▲
                              │   │   │   │   (internal only)
                     ┌────────────────────────────────────┐
                     │   nginx  (pde-nginx)  port 80      │
                     │   single entry point, subpath rout │
                     └────────────────────────────────────┘
                              │
                         port 8050 → host
```

Every service talks to every other service by **container name**
(e.g. `http://ollama:11434`, `postgres:5432`).
The host machine reaches all web UIs through a single port via **nginx**.

### Nginx subpath routing

All browser traffic enters on **port 8050** (the only port that needs to be open
in a restrictive firewall). Nginx routes by URL prefix:

| URL path | Routes to | Service |
|----------|-----------|---------|
| `/` | `dashboard:8050` | Plotly Dash (main UI) |
| `/mlflow/` | `mlflow:5000` | MLflow experiment tracking |
| `/agents/` | `agents:8000` | FastAPI REST + Swagger UI |
| `/browser/` | `neo4j:7474` | Neo4j Browser UI |
| `/neo4j-bolt` | `neo4j:7687` | Neo4j Bolt WebSocket |
| `/api/v1/` | `minio:9001` | MinIO console API calls |

MinIO console (`port 9001`) is accessed **directly** — see the note in
[Section 9](#9-minio--object-storage).

### Service port map

| Container | Internal port | Host port | What it exposes |
|-----------|-------------|-----------|-----------------|
| `pde-nginx` | 80 | **8050** | Reverse proxy — all web UIs on one port |
| `pde-ollama` | 11434 | 11434 | LLM inference API |
| `pde-fenics` | 8080 | 8080 | FEM simulation REST API |
| `pde-agents` | 8000 | 8000 | Orchestrator + agent REST API (also via `/agents/`) |
| `pde-dashboard` | 8050 | — | Plotly Dash (internal only, accessed via nginx `/`) |
| `pde-mlflow` | 5000 | — | MLflow (internal only, accessed via nginx `/mlflow/`) |
| `pde-minio` | 9000/9001 | 9000/9001 | S3 API / console (direct access on 9001) |
| `pde-postgres` | 5432 | 5432 | PostgreSQL |
| `pde-redis` | 6379 | 6379 | Redis |
| `pde-neo4j` | 7687 / 7474 | 7687 / 7474 | Bolt (driver) / Browser (also via nginx `/browser/`) |

### Why Docker?

Each service needs radically different software:

- FEniCSx requires a custom MPI + PETSc build
- Ollama needs GPU drivers and CUDA libraries
- Dash needs a Python web stack with scipy and plotly
- PostgreSQL is a compiled C database server

Isolating them in containers prevents dependency conflicts, and the whole
system starts with one command:

```bash
docker compose up -d
```

### Dependency chain

```
postgres ──▶ agents (waits for DB to be healthy)
redis    ──▶ agents (waits for Redis to be healthy)
ollama   ──▶ agents (waits for Ollama to be healthy)
neo4j    ──▶ agents (waits for Neo4j to be healthy — ~40 s on first boot)
agents   ──▶ dashboard
dashboard, mlflow, agents, fenics-runner, neo4j ──▶ nginx (soft dep)
```

This ensures agents never start before all dependencies are ready.
If Neo4j is temporarily unavailable at startup the agents continue without
knowledge graph features (graceful degradation) — KG reconnects on the next call.

---

## 2. FEniCSx/DOLFINx — Physics Engine

### What it is

FEniCSx is the leading open-source Finite Element Method (FEM) framework. It solves
Partial Differential Equations on arbitrary domains by converting them from a strong
differential form into a **weak (variational) form** and assembling a large sparse
linear system that is solved numerically.

Version: **DOLFINx 0.10.0.post2**, running inside the `fenics-runner` container.

### The heat equation being solved

**Strong form** (what you write in a physics textbook):

```
ρ c_p ∂u/∂t − ∇·(k ∇u) = f    in Ω × (0, T]
```

Where:
- `u(x,y,t)` — temperature field [K]
- `ρ` — density [kg/m³]
- `c_p` — specific heat [J/(kg·K)]
- `k` — thermal conductivity [W/(m·K)]
- `f` — volumetric heat source [W/m³]

**Boundary conditions:**

| Type | Equation | Physical meaning |
|------|----------|-----------------|
| Dirichlet | `u = g` on Γ_D | Fixed temperature (e.g. wall at 300 K) |
| Neumann | `k ∂u/∂n = h` on Γ_N | Prescribed heat flux (h=0 → insulated) |
| Robin | `k ∂u/∂n = α(u_∞ − u)` on Γ_R | Convective cooling |

### Step-by-step: how FEniCSx solves it

**Step 1 — Mesh the domain**

```python
msh = create_unit_square(comm, nx=64, ny=64, CellType.triangle)
```

The square domain [0,1]² is divided into 64×64 triangles.
Each triangle vertex is a **node** (degree of freedom, DOF).
For a 64×64 mesh: **(64+1)² = 4,225 DOFs**.

```
  y
  1 ┤ · · · · · ·
    │ · · · · · ·
    │ · · · · · ·   each dot = 1 DOF
    │ · · · · · ·   each triangle = 1 element
  0 ┤ · · · · · ·
    └────────────── x
    0               1
```

**Step 2 — Define the function space**

```python
V = functionspace(msh, ("Lagrange", 1))   # P1 Lagrange elements
```

The temperature `u(x,y,t)` is approximated as a sum of piecewise-linear
basis functions φᵢ, one per node:

```
u(x,y,t) ≈ Σᵢ uᵢ(t) · φᵢ(x,y)
```

The unknowns are the scalar coefficients `uᵢ` — the temperature at each node.

**Step 3 — Derive the weak form (θ-scheme)**

Multiply the strong form by a test function `v` and integrate over Ω.
Using the θ-scheme for time discretisation (θ=1 → Backward Euler):

```
(ρcₚ/dt)(u^{n+1} − u^n, v)_Ω
  + θ   [ k(∇u^{n+1}, ∇v)_Ω + α(u^{n+1}, v)_Γ_R ]
  + (1−θ)[ k(∇u^n, ∇v)_Ω    + α(u^n, v)_Γ_R     ]
= ∫_Ω f v dx + ∫_Γ_N h v ds + α(u_∞, v)_Γ_R
```

This is written in Python using UFL (Unified Form Language):

```python
a_ufl = (rho_cp/dt) * inner(u, v)*dx + theta * k * inner(grad(u), grad(v))*dx
L_ufl = (rho_cp/dt) * inner(u_n, v)*dx - (1-theta)*k * inner(grad(u_n), grad(v))*dx
```

**Step 4 — Assemble the linear system**

FEniCSx computes the **stiffness matrix A** and **load vector b** by integrating
the UFL forms over every element and assembling contributions into a global
sparse system:

```
A · u^{n+1} = b(u^n)
```

For 4,225 DOFs, A is a 4,225 × 4,225 sparse matrix with ~30,000 non-zero entries.

**Step 5 — Apply boundary conditions**

Dirichlet conditions modify rows of A and b to enforce `u = g` at boundary nodes.
Neumann and Robin conditions add flux terms to b through the facet integrals.

**Step 6 — Solve with PETSc**

```python
solver = PETSc.KSP()
solver.setType("cg")          # Conjugate Gradient
solver.getPC().setType("hypre")  # HYPRE AMG preconditioner
solver.solve(b, u_h.x.petsc_vec)
```

For a well-conditioned system this converges in ~20 iterations per time step.

**Step 7 — Advance in time**

```python
for step in range(n_steps):
    t += dt
    b = assemble_vector(L)      # rebuild RHS with updated u_n
    apply_lifting(b, [a], bcs=[dirichlet_bcs])
    set_bc(b, dirichlet_bcs)
    solver.solve(b, u_h)
    u_n.x.array[:] = u_h.x.array  # u^n ← u^{n+1}
```

**Step 8 — Save results**

At every `save_every` steps:

| File | Contents | Used by |
|------|----------|---------|
| `u_final.npy` | Final temperature at all DOFs | Dashboard visualisation |
| `dof_coords.npy` | (x,y) or (x,y,z) of every DOF | Dashboard interpolation |
| `snapshots/u_NNNN.npy` | Temperature at each saved time step | Dashboard animation |
| `snapshot_times.npy` | Physical times of each snapshot | Dashboard time slider |
| `temperature.xdmf/.h5` | Full time series (ParaView format) | External post-processing |
| `config.json` | All simulation parameters | Dashboard + replay |
| `result.json` | Summary statistics | Dashboard + database |

### Why FEniCSx over NumPy/SciPy finite differences?

| Feature | FEniCSx FEM | NumPy finite differences |
|---------|-------------|--------------------------|
| Geometry | Arbitrary domains, curved boundaries | Rectangular grids only |
| Boundary conditions | Dirichlet, Neumann, Robin, mixed | Dirichlet only (easily) |
| Convergence order | O(h²) with P1, O(h⁴) with P2 | O(h²) |
| 3D scaling | Tetrahedral meshes | Structured tensor grids |
| Parallel (MPI) | Built-in | Complex to implement |
| Weak form | Any PDE with a variational form | Heat equation only |

---

## 3. Ollama — Local LLM Server

### What it is

Ollama is an open-source tool that downloads, manages, and serves large language
models (LLMs) locally on your GPU. It exposes a REST API identical in design to
OpenAI's API, so switching from a cloud model to a local one requires changing only
the base URL.

### How it works

```
CPU: sends token IDs ──▶ GPU VRAM: model weights loaded ──▶ CPU: receives logits
                              ↕
                        CUDA kernels do
                        matrix multiplications
                        (transformer forward pass)
```

Ollama loads model weights in **GGUF format** (quantised to 4-8 bits), which means
a 70B parameter model that would normally require ~140 GB of full-precision memory
fits in ~42 GB of VRAM.

### The API call

```bash
POST http://ollama:11434/api/chat
{
  "model": "qwen2.5-coder:32b",
  "messages": [
    {"role": "system", "content": "You are a physics simulation expert..."},
    {"role": "user",   "content": "Run a 2D heat equation on steel..."}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "run_simulation",
        "description": "Execute a FEM simulation",
        "parameters": {
          "type": "object",
          "properties": {
            "config_json": {"type": "string"}
          }
        }
      }
    }
  ],
  "stream": false
}
```

The response contains either:
- A natural language answer in `message.content`
- A structured tool call in `message.tool_calls` (llama3.3)
- A JSON string in `message.content` that looks like a tool call (qwen2.5-coder)

### The three models and why each was chosen

| Model | Parameters | VRAM | Role | Why |
|-------|-----------|------|------|-----|
| `qwen2.5-coder:32b` | 32B | ~19 GB | Simulation Agent | Best at structured JSON, code generation, config validation |
| `qwen2.5-coder:14b` | 14B | ~9 GB | Database Agent | Faster responses, good at SQL-like structured output |
| `llama3.3:70b` | 70B | ~42 GB | Analytics + Orchestrator | Best multi-step reasoning, native structured tool calling |

All three fit simultaneously in the ~196 GB combined VRAM of the two RTX PRO 6000
Blackwell GPUs.

### Tool calling compatibility

`llama3.3:70b` outputs proper structured tool calls:
```json
{"tool_calls": [{"id": "call_abc", "function": {"name": "run_simulation", "arguments": {...}}}]}
```

`qwen2.5-coder` models output tool calls as text in `content`:
```json
{"name": "run_simulation", "arguments": {"config_json": "..."}}
```

The `_parse_content_tool_call()` function in `base_agent.py` detects this pattern,
parses the JSON, and converts it into a proper structured tool call before LangGraph
processes it. This makes both model families work identically from the agent's
perspective.

---

## 4. LangChain & LangGraph — Agent Framework

### LangChain: the building blocks

LangChain provides Python abstractions for working with LLMs:

**`ChatOllama`** wraps the Ollama API as a Python object:
```python
llm = ChatOllama(model="qwen2.5-coder:32b", base_url="http://ollama:11434")
```

**`@tool` decorator** wraps any Python function so an LLM can call it:
```python
@tool
def run_simulation(config_json: str) -> str:
    """Execute a FEM simulation. config_json is a JSON-encoded HeatConfig."""
    response = requests.post("http://fenics-runner:8080/run", json=json.loads(config_json))
    return json.dumps(response.json())
```

The docstring becomes the tool description. The type annotations become the JSON
schema. The LLM sees both and knows exactly when and how to call the function.

**`bind_tools()`** attaches the tool schemas to a model:
```python
llm_with_tools = llm.bind_tools([run_simulation, validate_config, ...])
```

### LangGraph: the state machine

LangGraph turns an agent into a **directed graph** of nodes and edges.
The state flows through the graph until it reaches an END node.

**AgentState** — the shared memory that flows between nodes:
```python
class AgentState(TypedDict):
    messages:       Annotated[list[BaseMessage], add_messages]
    iteration:      int
    status:         str
    tool_calls_log: list[dict]
    run_ids:        list[str]
    final_answer:   str
```

**The ReAct graph:**
```
                    ┌─────────────────────────────┐
                    │         AgentState          │
                    │  messages, run_ids, ...     │
                    └─────────────────────────────┘

  START ──▶ reason_node ──▶ (has tool calls?) ──▶ YES ──▶ act_node ──┐
                │                                                      │
                └──▶ NO ──▶ finish_node ──▶ END           (tool result│
                                                           appended to │
                                                           messages)   │
                            ◀────────────────────────────────────────┘
                            (loops back to reason_node)
```

**reason_node** — calls the LLM with the full message history:
```python
response = self.llm.invoke(messages)
response = self._parse_content_tool_call(response)  # normalize qwen output
```

**act_node** (LangGraph's built-in `ToolNode`) — inspects `response.tool_calls`,
finds the matching Python function, executes it, appends the result as a
`ToolMessage`:
```python
ToolMessage(content='{"status": "success", "T_max": 500.0, ...}',
            tool_call_id="call_abc123")
```

**Router** — decides whether to keep looping or finish:
```python
def _router(state):
    last_msg = state["messages"][-1]
    if last_msg.tool_calls and state["iteration"] < max_iterations:
        return "act"
    return "finish"
```

**finish_node** — extracts the final text answer, strips tokeniser artifacts:
```python
content = last_msg.content
for marker in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"):
    content = content.split(marker)[0].strip()
```

### Why LangGraph over simple loops?

| Feature | Manual loop | LangGraph |
|---------|------------|-----------|
| State persistence | Manual dict management | Typed TypedDict, automatic merge |
| Branching | if/else spaghetti | Declarative edge conditions |
| Multi-agent coordination | Very hard | Composable sub-graphs |
| Streaming | Requires async | Built-in `.stream()` |
| Debugging | print() | Full state at every step |
| Checkpointing | Manual | Built-in with SQLite/Redis |

---

## 5. The Three Agents — Specialised Workers

Each agent:
1. Inherits the `BaseAgent` class (LangGraph state machine)
2. Has a **system prompt** that defines its persona and constraints
3. Has a **specific set of tools** (the Python functions it can call)

### Agent-1: Simulation Agent

**Model:** `qwen2.5-coder:32b`
**File:** `agents/simulation_agent.py`

**Tools:**

| Tool | What it does |
|------|-------------|
| `validate_config` | Parses JSON config, checks all required fields, validates physics ranges |
| `run_simulation` | POSTs to fenics-runner:8080/run, waits for result |
| `modify_config` | Applies parameter changes to an existing config JSON |
| `debug_simulation` | Analyses error messages, suggests fixes |
| `list_recent_runs` | Fetches recent runs from the database |
| `get_run_status` | Looks up a specific run's status and metrics |
| `run_parametric_sweep` | Loops over a parameter range, runs multiple simulations |

**System prompt key rules (excerpt):**
> "Always validate before running. Always set u_init close to Dirichlet BC values
> to avoid numerical overshoot. For stability: Backward Euler (θ=1) is unconditionally
> stable — prefer it. Start with coarse meshes (32×32) then refine if needed."

**Typical reasoning trace:**
```
Iteration 1: LLM calls validate_config(config_json)
Iteration 2: validate_config returns "valid: true" → LLM calls run_simulation(config_json)
Iteration 3: run_simulation returns {T_max: 500.0, wall_time: 0.09s} → LLM writes final answer
```

---

### Agent-2: Analytics Agent

**Model:** `llama3.3:70b`
**File:** `agents/analytics_agent.py`

**Tools:**

| Tool | What it does |
|------|-------------|
| `analyze_run` | Loads result JSON, computes α, τ, identifies anomalies |
| `compare_runs` | Side-by-side comparison of multiple runs |
| `compare_study` | Analyses a full parametric study across all runs |
| `get_steady_state_time` | Estimates when the L2 norm stops changing significantly |
| `suggest_next_run` | Uses current results to propose next parameter set |
| `export_summary_report` | Writes a formatted markdown report to disk |

**What it computes:**
- Thermal diffusivity: `α = k / (ρ c_p)`
- Characteristic time: `τ = L² / α`
- Steady-state convergence: finds index where `|Δ(L2)| / L2 < 1e-4`
- Temperature uniformity: T_max − T_min as a measure of gradient

---

### Agent-3: Database Agent

**Model:** `qwen2.5-coder:14b`
**File:** `agents/database_agent.py`

**Tools:**

| Tool | What it does |
|------|-------------|
| `store_result` | Reads result.json + config.json, inserts into PostgreSQL |
| `query_runs` | Translates natural language to SQLAlchemy filter → returns matching runs |
| `catalog_study` | Groups related runs into a `ParametricStudy` record |
| `fetch_run_data` | Returns full config + results for a specific run |
| `export_to_csv` | Writes a filtered set of runs to CSV |
| `upload_to_minio` | Uploads XDMF/NPY files to MinIO object storage |
| `db_health_check` | Verifies database connectivity and table counts |

---

## 6. The Orchestrator — Supervisor

**File:** `orchestrator/graph.py`

The orchestrator is a **supervisor graph** that sits above the three specialist agents.
It uses `llama3.3:70b` to:
1. Decompose the user's high-level task into sub-tasks
2. Decide which agent to call next
3. Pass results between agents (e.g. run_ids from Simulation → Analytics)
4. Synthesise all agent outputs into a final report

### How it decides which agent to call

The supervisor prompt instructs the LLM to output a routing decision:
```
{"next": "simulation", "reason": "Need to run the FEM first before analyzing"}
{"next": "analytics",  "reason": "Have run_ids, now need to analyze results"}
{"next": "database",   "reason": "Analysis done, store everything"}
{"next": "FINISH",     "reason": "All tasks complete, ready to report"}
```

### Shared state

```python
class OrchestratorState(TypedDict):
    messages:      list[BaseMessage]
    run_ids:       list[str]          # populated by Simulation Agent
    analysis:      dict               # populated by Analytics Agent
    db_records:    list[str]          # populated by Database Agent
    final_report:  str
```

The `run_ids` list is how the orchestrator passes information from one agent to
the next without the user having to copy-paste anything.

---

## 7. FastAPI — REST Interface

**File:** `orchestrator/api.py`

FastAPI is a modern Python web framework that auto-generates OpenAPI documentation
and uses Pydantic models for request/response validation.

### Reverse-proxy awareness (`ROOT_PATH`)

When the agents service runs behind nginx under the `/agents/` subpath, FastAPI's
default Swagger UI generates an absolute OpenAPI spec URL (`/openapi.json`) that
resolves to the dashboard root rather than the agents service.

The fix is a custom `/docs` endpoint driven by the `ROOT_PATH` environment variable:

```python
_ROOT_PATH = os.getenv("ROOT_PATH", "")   # set to "/agents" in docker-compose

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui() -> HTMLResponse:
    return get_swagger_ui_html(
        openapi_url=f"{_ROOT_PATH}/openapi.json",  # → /agents/openapi.json
        title="PDE Agents API",
    )
```

nginx strips `/agents/` before forwarding, so the chain is:

```
Browser → GET /agents/docs    → nginx → agents:8000/docs         → custom HTML
Browser → GET /agents/openapi.json → nginx → agents:8000/openapi.json → JSON spec
```

Both return 200 and Swagger UI renders correctly.

### The async job pattern

LLM inference + FEM solving can take 3–5 minutes. Holding an HTTP connection open
for that long causes timeouts in browsers and load balancers.

The solution is a **submit-and-poll** pattern:

```
Client                          FastAPI
  │                               │
  │  POST /run/async {"task":"..."} │
  │  ◀────── {job_id: "abc123"} ──┤  ← returns in <1s
  │                               │  (job runs in background thread)
  │  GET /jobs/abc123             │
  │  ◀── {status: "running", ...} ┤  ← poll every 3s
  │                               │
  │  GET /jobs/abc123             │
  │  ◀── {status: "success", ...} ┤  ← result ready
```

**Implementation:**
```python
_jobs: dict[str, dict] = {}           # in-memory job store
_executor = ThreadPoolExecutor(max_workers=4)

@app.post("/run/async")
async def run_task_async(request: TaskRequest):
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"status": "running", ...}

    def _run():
        result = orchestrator.run(request.task)
        _jobs[job_id].update(status="success", result=result)

    asyncio.get_event_loop().run_in_executor(_executor, _run)
    return {"job_id": job_id, "status": "running"}
```

### All endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/run` | Synchronous multi-agent task (blocks until done) |
| `POST` | `/run/async` | Submit multi-agent task, returns job_id |
| `POST` | `/agent/{name}/async` | Submit to one agent (simulation/analytics/database) |
| `GET` | `/jobs/{job_id}` | Poll job status and result |
| `GET` | `/jobs` | List all submitted jobs |
| `GET` | `/runs` | List simulation runs from database |
| `GET` | `/runs/{run_id}` | Full details for one run |
| `GET` | `/health` | Service health + available Ollama models |
| `WS` | `/ws/stream` | WebSocket for streaming agent output |

---

## 8. PostgreSQL + SQLAlchemy — Metadata Database

### PostgreSQL

PostgreSQL is a battle-tested relational database. It stores structured simulation
metadata — small scalar values that you want to query, filter, and aggregate.

### SQLAlchemy ORM

SQLAlchemy maps Python classes to database tables. Instead of writing SQL strings,
you work with Python objects:

```python
# Create a record
run = SimulationRun(run_id="steel_01", k=50.0, nx=64, status=RunStatus.PENDING)
session.add(run)
session.commit()

# Query
runs = session.execute(
    select(SimulationRun)
    .where(SimulationRun.k > 40)
    .order_by(desc(SimulationRun.created_at))
).scalars().all()
```

### Database schema

```
simulation_runs          ← one row per simulation run
  id, run_id, status, dim, nx, ny, nz, k, rho, cp
  T_max, T_min, T_mean, n_dofs, wall_time
  config_json, output_dir, created_at

run_parameters           ← queryable key-value pairs from config
  run_id, key, value, dtype

run_results              ← scalar result metrics
  run_id, metric_name, value

convergence_records      ← per-time-step convergence data
  run_id, step, time, l2_norm, T_max, T_min

parametric_studies       ← groups of related runs
  id, name, parameter, values, description

agent_messages           ← full conversation history
  id, agent, role, content, timestamp

agent_suggestions        ← structured suggestions from Analytics Agent
  run_id, parameter, suggested_value, reason
```

### What you can query

Because all runs are in SQL, the Database Agent can answer questions like:

- *"Find all runs where T_max > 450 K and wall time < 1 s"*
- *"What is the average mesh resolution for 3D runs?"*
- *"Which value of k gave the most uniform temperature distribution?"*
- *"List the 5 fastest simulations"*

---

## 9. MinIO — Object Storage

### What it is

MinIO is a self-hosted, S3-compatible object storage server. It stores **large binary
files** that don't belong in a SQL database:

- `temperature.xdmf` / `.h5` — full time-series data (megabytes per run)
- `u_final.npy` — final temperature field arrays
- `dof_coords.npy` — mesh DOF coordinate arrays
- `snapshots/*.npy` — per-timestep animation frames

### Why not just the filesystem?

| Feature | Filesystem | MinIO |
|---------|-----------|-------|
| Web browser access | No | Yes (http://`host`:9001) |
| Multiple workers sharing data | Needs NFS | Native |
| Versioning | No | Yes |
| Presigned URLs | No | Yes |
| Bucket policies / access control | No | Yes |

### Accessing the MinIO console

MinIO's SPA uses `<base href="/">` which prevents serving it cleanly behind an
nginx subpath. The console is therefore accessed **directly on port 9001**:

| Scenario | How to access |
|----------|--------------|
| Port 9001 open in firewall | Click the MinIO link in the dashboard navbar |
| Port 9001 firewalled | `ssh -L 19001:localhost:9001 -N <server>` → open `http://localhost:19001` |

The dashboard navbar link is built dynamically as `http://<current hostname>:9001`
so it always points to the correct server without any hardcoded IP.

### How it's used

The Database Agent's `upload_to_minio` tool:

```python
client = Minio("minio:9000", access_key, secret_key)
client.fput_object(bucket="simulations", object_name=f"{run_id}/u_final.npy",
                   file_path=local_path)
```

---

## 10. Redis — Message Broker

Redis is an in-memory key-value store used as a message queue. In the current setup
it is a dependency for future scaling:

**Current use:** none (placeholder)

**Future use:** If you scale to multiple `agents` container replicas, they coordinate
through Redis rather than sharing in-process `_jobs` dict state. Celery workers
would pick simulation tasks from a Redis queue and distribute them across multiple
machines.

**Why Redis for queues?** It is extremely fast (~100k ops/s), supports pub/sub,
sorted sets (priority queues), and expiration (auto-cleanup of old jobs).

---

## 11. Plotly Dash — Visualization Dashboard

### What Dash is

Dash is a Python framework for building reactive web applications. It combines:
- **Flask** — Python web server
- **Plotly.js** — JavaScript charting library
- **React** — JavaScript UI framework

The key insight: you write everything in Python. Dash generates the React components
and JavaScript callbacks automatically.

### How reactivity works

```
Browser ──── HTTP GET ────▶ Flask serves HTML + JS
    │
    │  User changes dropdown
    │
    └── POST /_dash-update-component ──▶ Python callback runs
                                         returns updated figure JSON
                                         ◀─── React re-renders chart
```

Every `@app.callback` decorator declares:
- **Inputs** — what triggers the callback (dropdown changes, button clicks, timer)
- **States** — values read but not triggering
- **Outputs** — what gets updated (figure data, text, style)

Example:
```python
@app.callback(
    Output("field-main-plot", "figure"),
    Input("field-run-selector", "value"),
    Input("field-view-mode",    "value"),
)
def update_field_view(run_id, view_mode):
    u      = np.load(f"/workspace/results/{run_id}/u_final.npy")
    coords = np.load(f"/workspace/results/{run_id}/dof_coords.npy")
    # ... build and return plotly figure
```

### The Field Viewer — how visualisation actually works

The temperature field from FEniCSx is stored as **scattered data** — 4,225 values
at irregular DOF positions. Plotly's `go.Heatmap` requires a **regular grid**.
The bridge is `scipy.interpolate.griddata`:

```python
from scipy.interpolate import griddata

xi = np.linspace(0, 1, 250)   # regular x grid
yi = np.linspace(0, 1, 250)   # regular y grid
Xi, Yi = np.meshgrid(xi, yi)

# Interpolate scattered DOF values onto the regular grid
Zi = griddata(coords[:, :2], u_values, (Xi, Yi), method="linear")

# Now Zi is 250×250 → perfect for go.Heatmap
fig.add_trace(go.Heatmap(x=xi, y=yi, z=Zi, colorscale="Plasma"))
```

### Navigation links — dynamic hostname resolution

The dashboard navbar contains links to all other services. Because the server's
IP address varies across deployments, the links are built at runtime in the browser
using a **clientside callback** (JavaScript running client-side, not a Python
server round-trip):

```javascript
// dashboard.py — clientside_callback
function(href) {
    var host = window.location.hostname;
    return [
        '/agents/docs',              // API Docs  — via nginx subpath
        '/mlflow/',                  // MLflow    — via nginx subpath
        'http://' + host + ':9001', // MinIO     — direct port (not proxied)
        '/browser/'                  // Neo4j     — via nginx subpath
    ];
}
```

Relative paths (`/agents/docs`, `/mlflow/`, `/browser/`) resolve through nginx
on port 8050 automatically. MinIO uses an absolute URL with `window.location.hostname`
because its SPA (`<base href="/">`) cannot be served behind a subpath without
additional configuration.

All nav links use `external_link=True` so Dash opens them in new tabs rather
than routing them through its internal SPA router.

**MinIO access when port 9001 is firewalled:**
```bash
ssh -L 19001:localhost:9001 -N <server>   # tunnel to a free local port
# then open http://localhost:19001
```

### The seven view modes

| Mode | Method | Description |
|------|--------|-------------|
| 🌡 Heatmap | `go.Heatmap` + `go.Contour` | Colour map + 12 labelled isothermal lines |
| 🏔 3D Surface | `go.Surface` | Temperature raised as height, fully rotatable |
| ∇ Heat Flux | gradient of interpolated field | `\|−k∇T\|` magnitude + directional arrows |
| 〰 Profiles | `go.Scatter` slices | T(x) at 6 fixed Y values, T(y) at 6 fixed X values |
| 🍕 Z-Slice | `scipy.griddata` + `go.Heatmap` | XY plane cut of 3D run at adjustable Z |
| 📦 Volume | 3 × `go.Surface` + `go.Isosurface` | Orthogonal slice planes + isosurfaces |
| 🎬 Animation | `dcc.Interval` advancing snapshot index | Playback of saved time snapshots |

### Animation mechanism

```
dcc.Store(id="field-anim-store")  ← {"step": 3, "playing": True}
dcc.Interval(id="anim-interval", interval=600ms)
dbc.Button(id="field-play-btn")

Button click callback:
  toggle playing state in Store

Interval callback (every 600ms, when enabled):
  read current step from Store
  increment step → write back to Store

Store change triggers main callback:
  loads snapshots/u_{step:04d}.npy
  rebuilds the figure
  → browser re-renders
```

---

## 12. MLflow — Experiment Tracking

### What it is

MLflow is an open-source platform for managing machine learning and scientific
computing experiments. It logs parameters, metrics, and artifacts for every run,
making it easy to compare them later.

### How it's used here

Every simulation run is logged as an MLflow experiment:

```python
with mlflow.start_run(run_name=run_id):
    # Parameters
    mlflow.log_params({"k": 50.0, "nx": 64, "ny": 64, "dt": 0.1, ...})

    # Metrics
    mlflow.log_metrics({"T_max": 500.0, "T_min": 300.0, "wall_time": 0.16})

    # Artifacts
    mlflow.log_artifact("result.json")
    mlflow.log_artifact("config.json")
```

### What you see at http://`host`:8050/mlflow/

- A table of all runs with their parameters and metrics
- Side-by-side comparison charts: k vs T_max, mesh size vs wall time
- Artifact browser: download result JSONs, view configs
- Run reproducibility: every config is stored, so any run can be exactly repeated

---

## 13. Neo4j — Simulation Knowledge Graph

### What it is

Neo4j is a **native graph database** — data is stored as nodes, relationships, and
properties rather than rows and columns. It is the foundation of the system's
long-term memory: every simulation run is added to the graph, and agents query it
before deciding what to do next.

### Why a graph database?

| Concern | Relational (PostgreSQL) | Graph (Neo4j) |
|---------|------------------------|---------------|
| "Which runs used similar parameters?" | Complex multi-join SQL | Single MATCH with range filter |
| "What material is k=50, ρ=7800?" | Full table scan | O(1) index + relationship hop |
| "What happened after similar failures?" | 5-table join | MATCH path traversal |
| Pattern mining over relationships | Requires ORM + aggregations | First-class Cypher |

### Graph schema (Phase 1)

```
(:Run {
    run_id, dim, status, k, rho, cp,
    nx, ny, nz, t_end, dt, theta, source, u_init,
    t_max, t_min, t_mean, l2_norm, wall_time, n_dofs,
    created_at
})

(:Material {
    name, k, rho, cp, alpha,
    k_min, k_max, description, typical_uses
})

(:KnownIssue {
    code, severity, condition,
    description, recommendation, observed_in
})

Relationships:
  (:Run)-[:USES_MATERIAL {confidence}]->(:Material)
  (:Run)-[:TRIGGERED {detected_at}]->(:KnownIssue)
  (:Run)-[:SPAWNED_FROM]->(:Run)      — created from an agent suggestion
  (:Run)-[:IMPROVED_OVER {metric}]->(:Run)  — planned Phase 2
```

### What is seeded at startup

The agents container seeds static physical knowledge into Neo4j when it starts:

**10 engineering materials:**

| Material | k [W/m·K] | ρ [kg/m³] | c_p [J/kg·K] |
|----------|-----------|-----------|--------------|
| Steel (carbon) | 50 | 7800 | 500 |
| Stainless Steel 316 | 16 | 8000 | 500 |
| Aluminium 6061 | 200 | 2700 | 900 |
| Copper | 385 | 8960 | 385 |
| Titanium Ti-6Al-4V | 6.7 | 4430 | 526 |
| Silicon | 150 | 2330 | 700 |
| Concrete | 1.7 | 2300 | 880 |
| Glass (borosilicate) | 1.0 | 2230 | 830 |
| Water | 0.6 | 1000 | 4182 |
| Air | 0.026 | 1.2 | 1005 |

**6 documented failure patterns (KnownIssue nodes):**

| Code | Condition | Severity |
|------|-----------|---------|
| `GIBBS_OVERSHOOT` | `u_init` far from Dirichlet BCs | high |
| `EXPLICIT_INSTABILITY` | θ < 0.5 AND dt > h²/(2α) | high |
| `MESH_TOO_COARSE` | nx < 10 (2D) or < 8 (3D) | medium |
| `NO_STEADY_STATE_REACHED` | L2 norm still decreasing at t_end | medium |
| `PURE_NEUMANN_ILL_POSED` | No Dirichlet BC → singular system | high |
| `NEGATIVE_TEMPERATURES` | T_min < 0 K in solution | high |

### How the graph grows with usage

Every successful simulation run triggers:

```
run_simulation(config)
  │
  ├── 1. POST to fenics-runner → FEniCSx solves PDE
  ├── 2. mark_run_finished() → PostgreSQL stores metadata
  ├── 3. _upload_run_to_minio() → output files archived in MinIO
  └── 4. kg.add_run(run_id, config, results, warnings)
            │
            ├── MERGE (:Run {run_id}) with all properties
            ├── MERGE (:Run)-[:USES_MATERIAL]->(:Material)   ← material inferred
            │         by matching k to the k_range of each Material node
            └── MERGE (:Run)-[:TRIGGERED]->(:KnownIssue)    ← for each rule violation
```

After 50 runs, a typical query for context looks like:
```
"2D run with k≈50 (steel)"
  → 8 similar past runs found
  → T_max range: 478–502 K (consistent)
  → 2 runs triggered INCONSISTENT_IC (u_init was 0, BCs were 300/500 K)
  → 0 runs triggered CFL violation (all used theta=1.0)
```

### Rule-based warning engine

`knowledge_graph/rules.py` implements 8 pure-Python rules that run **instantly**
before every simulation, regardless of Neo4j availability:

| Rule code | Trigger condition |
|-----------|-----------------|
| `INCONSISTENT_IC` | \|u_init − min_BC\| > 100 K |
| `EXPLICIT_CFL_VIOLATION` | θ < 0.5 AND dt > h²/(2α) |
| `NEAR_EXPLICIT_SCHEME` | 0 ≤ θ < 0.5 |
| `COARSE_MESH_2D` | dim=2 AND (nx < 10 OR ny < 10) |
| `COARSE_MESH_3D` | dim=3 AND any direction < 8 |
| `SHORT_SIMULATION` | t_end < 5 × dt |
| `INVALID_MATERIAL_PROPS` | k, rho, or cp ≤ 0 |
| `LARGE_DT_RELATIVE_TO_DIFFUSION` | dt > 10 × h²/α |
| `NO_BOUNDARY_CONDITIONS` | bcs list is empty |

### Agent integration

**Simulation Agent** (runs before every simulation):
```
Iteration 1:
  check_config_warnings(config_json) →
    warnings: [INCONSISTENT_IC (high): u_init=0 K, BC_min=300 K]
    similar_runs: [heat_abc123 (T_max=500, wall_time=0.7s), ...]
    recommendation: "CAUTION: 1 high-severity issue. Review before running."

Iteration 2:
  LLM fixes: u_init=300 (matches BC), then calls validate_config()

Iteration 3:
  run_simulation(fixed_config) → FEniCSx → success
  [auto: MinIO upload + KG add_run]
```

**Database Agent** (material lookup):
```
User: "What are the thermal properties of titanium?"
Agent calls: query_knowledge_graph(material="titanium")
  → {name: "Titanium Ti-6Al-4V", k: 6.7, rho: 4430, cp: 526,
     description: "Very low conductivity for a metal. High thermal gradients expected.",
     typical_uses: "aerospace, biomedical implants, high-temperature applications"}
```

**Analytics Agent** (similar-run discovery):
```
User: "Find runs similar to my copper simulation"
Agent calls: query_knowledge_graph(k=385.0, dim=2)
  → similar_runs: [{run_id: "heat_xyz", t_max: 373.1, wall_time: 0.4s, material: "Copper"}, ...]
```

### REST API

```bash
GET  /kg/stats                    # node counts (runs, materials, known_issues)
POST /kg/seed                     # re-seed static knowledge (idempotent)
GET  /kg/material/{name}          # look up material by name (partial, case-insensitive)
GET  /kg/run/{run_id}/similar     # find runs similar to this run's config
GET  /kg/run/{run_id}/lineage     # SPAWNED_FROM ancestry chain
```

### Neo4j Browser

Open **http://`host`:8050/browser/** and log in with `neo4j` / `pde_graph_secret`.

> Also directly accessible at `http://host:7474` if that port is open.
> When accessed via the nginx proxy at `/browser/`, the Bolt WebSocket connection
> is tunnelled through the `/neo4j-bolt` location on port 8050 — no extra port
> forwarding needed.

Useful starter queries:
```cypher
// Overview of the graph
MATCH (n) RETURN labels(n), count(n)

// All runs and their material
MATCH (r:Run)-[:USES_MATERIAL]->(m:Material)
RETURN r.run_id, r.dim, r.k, r.t_max, m.name
ORDER BY r.created_at DESC LIMIT 20

// Which failure patterns are most common?
MATCH (r:Run)-[:TRIGGERED]->(i:KnownIssue)
RETURN i.code, i.severity, count(r) AS occurrences
ORDER BY occurrences DESC

// Runs that spawned from a suggestion chain
MATCH p=(child:Run)-[:SPAWNED_FROM*]->(root:Run)
RETURN [n IN nodes(p) | n.run_id] AS lineage
```

### Phase roadmap

| Phase | Status | Features |
|-------|--------|---------|
| **Phase 1** | ✅ Done | Neo4j container, Run nodes, Material + KnownIssue seeding, `add_run`, similarity search, pre-run context, rule engine, agent tools, REST endpoints |
| Phase 2 | Planned | Automated correlation miner (finds patterns across 50+ runs), `IMPROVED_OVER` relationships, `suggest_next_run` that reads from graph |
| Phase 3 | Planned | Graph becomes primary agent memory; agents cite specific past run IDs in their reasoning; user-facing "Why did the agent suggest this?" lineage view |


---

## 14. How Everything Works Together — End to End

Below is the **complete flow** for a single user request typed into the dashboard
chat: *"Run a 2D heat equation on a 2 cm × 4 cm steel plate, compare to aluminum,
store results"*

```
┌─────────────────────────────────────────────────────────────────────┐
│  Step 1: User submits the task                                      │
│                                                                     │
│  Dashboard (Dash) in browser                                        │
│    POST /run/async {"task": "Run a 2D heat equation on a 2cm × 4cm │
│                     steel plate, compare to aluminum, store results"}│
│    ← {job_id: "ff5646", status: "running"}  (returns in <1 second)  │
│                                                                     │
│  Dashboard starts polling GET /jobs/ff5646 every 3 seconds         │
│  Shows "⏳ Running... 45s elapsed"                                  │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 2: FastAPI dispatches to Orchestrator                         │
│                                                                     │
│  FastAPI (agents:8000)                                              │
│    Spawns thread: orchestrator.run(task)                            │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 3: Orchestrator plans the workflow                            │
│                                                                     │
│  Orchestrator (LangGraph + llama3.3:70b via Ollama)                 │
│    Calls Ollama: "What agents do I need for this task?"             │
│    Ollama responds: "simulation → simulation → analytics → database" │
└─────────────────────────────────────────────────────────────────────┘
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
┌──────────────────┐  ┌───────────────┐  ┌──────────────────┐
│   Simulation     │  │   Analytics   │  │    Database      │
│   Agent          │  │   Agent       │  │    Agent         │
│   (sequential)   │  │   (after sim) │  │   (after analy.) │
└──────────────────┘  └───────────────┘  └──────────────────┘
```

### Simulation Agent in detail

```
Simulation Agent (qwen2.5-coder:32b)

Iteration 1:
  LLM reads task: "2cm × 4cm steel plate..."
  LLM calls: validate_config({dim:2, nx:32, ny:64, k:50, rho:7800,
                               cp:500, u_init:300, bcs:[...]})
  Tool returns: "valid: true, estimated DOFs: 2145"

Iteration 2:
  LLM calls: run_simulation(steel_config)
    → HTTP POST to fenics-runner:8080/run
    → FEniCSx:
        build mesh (32×64 triangles)
        assemble A (2145×2145 sparse matrix)
        solve Ax=b for t = 0.1, 0.2, ... 10.0 s  (100 time steps)
        save u_final.npy, dof_coords.npy, snapshots/
    → returns {status: "success", T_max: 500.0, T_min: 300.0,
               wall_time: 0.65s, n_dofs: 2145}

Iteration 3:
  LLM calls: run_simulation(aluminum_config)
    → Same process for aluminum (k=237, rho=2700, cp=900)
    → returns {status: "success", T_max: 500.0, ...}

Iteration 4:
  LLM writes final answer:
    "Steel run (run_id: steel_2d) complete.
     Aluminum run (run_id: alum_2d) complete.
     Both succeeded."

  run_ids = ["steel_2d", "alum_2d"] passed to orchestrator state
```

### Analytics Agent in detail

```
Analytics Agent (llama3.3:70b)

Iteration 1:
  LLM calls: analyze_run("steel_2d")
    → loads result.json: T=[300..500], wall_time=0.65
    → computes α = 50/(7800×500) = 1.28e-5 m²/s
    → computes τ = 1²/α = 78,000 s
    → returns {alpha: 1.28e-5, tau: 78000, T_range: 200}

Iteration 2:
  LLM calls: analyze_run("alum_2d")
    → computes α = 237/(2700×900) = 9.75e-5 m²/s
    → computes τ = 1²/α = 10,253 s
    → returns {alpha: 9.75e-5, tau: 10253, T_range: 200}

Iteration 3:
  LLM calls: compare_runs(["steel_2d", "alum_2d"])
    → ratio: τ_steel/τ_alum = 78000/10253 = 7.6×
    → returns comparison table

Iteration 4:
  LLM calls: suggest_next_run(...)
    → "Try finer mesh (64×128) to verify mesh-independence"

Iteration 5:
  LLM writes final answer:
    "Aluminum reaches steady state 7.6× faster than steel.
     Both show linear T profile at steady state (pure Dirichlet BCs).
     Suggestion: refine mesh to 64×128 to confirm convergence."
```

### Database Agent in detail

```
Database Agent (qwen2.5-coder:14b)

Iteration 1:
  LLM calls: store_result("steel_2d")
    → reads result.json + config.json from /workspace/results/steel_2d/
    → creates SimulationRun record in PostgreSQL
    → creates RunParameter records for each config key
    → returns "stored: id=42"

Iteration 2:
  LLM calls: store_result("alum_2d")
    → same → returns "stored: id=43"

Iteration 3:
  LLM calls: catalog_study(["steel_2d","alum_2d"], "material_comparison")
    → creates ParametricStudy(name="material_comparison", runs=[42,43])
    → returns "study id=7 created"

Iteration 4:
  LLM writes final answer:
    "Stored 2 runs in database (IDs 42, 43).
     Cataloged as parametric study 'material_comparison'."
```

### Orchestrator synthesises

```
Orchestrator combines all agent outputs:

Final report:
  "Steel vs Aluminum Comparison Complete

   Steel plate (k=50 W/(mK), steel_2d):
     T ∈ [300, 500] K  |  α = 1.28e-5 m²/s  |  τ = 78,000 s

   Aluminum block (k=237 W/(mK), alum_2d):
     T ∈ [273, 373] K  |  α = 9.75e-5 m²/s  |  τ = 10,253 s

   Key finding: Aluminum diffuses heat 7.6× faster.

   Both runs stored as study 'material_comparison' (IDs 42, 43).

   Recommended next step: run mesh refinement study (nx=[32,64,128])
   to verify solution independence."
```

### Dashboard receives the result

```
GET /jobs/ff5646 → {status: "success", elapsed_s: 173.4, result: {...}}

Dashboard:
  → Renders the final report in the chat bubble
  → User clicks 🌡️ Field Viewer tab
  → Selects "steel_2d" run
  → Selects "Heatmap + Isolines" view
  → scipy.griddata interpolates 2145 DOF values → 250×250 grid
  → go.Heatmap renders temperature field
  → go.Contour overlays 12 white isothermal lines
  → User clicks "3D Surface" → go.Surface renders T as height
  → User clicks "Animation" → 25 frames play back the transient
```

---

## 15. The Four-Layer Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                      INTELLIGENCE LAYER                            │
│                                                                    │
│   LangGraph agents + LLMs (Ollama)                                 │
│                                                                    │
│   • Understands natural language instructions                      │
│   • Reasons about physics parameters and their implications        │
│   • Validates configs with check_config_warnings before running    │
│   • Identifies anomalies (negative temperatures, divergence)       │
│   • Proposes next experiments, cites similar past runs from KG     │
│   • Synthesises multi-step outputs into human-readable reports     │
│   • Logs every reasoning step to agent_run_logs (full trace)       │
│                                                                    │
│   Components: base_agent.py, simulation_agent.py,                  │
│               analytics_agent.py, database_agent.py,              │
│               orchestrator/graph.py, tools/knowledge_tools.py      │
└──────────────────────────┬─────────────────────────────────────────┘
                           │
                           │  Tool calls (HTTP REST / SQLAlchemy / Bolt)
                           │
┌──────────────────────────▼─────────────────────────────────────────┐
│                      KNOWLEDGE LAYER                               │
│                                                                    │
│   Neo4j Knowledge Graph                                            │
│                                                                    │
│   • Stores every run as a graph node with full config + results    │
│   • Infers material from k/rho/cp → USES_MATERIAL relationship     │
│   • Records failure patterns → TRIGGERED relationship              │
│   • Answers: "What similar runs exist? What warnings apply?"       │
│   • Grows automatically with every successful simulation           │
│   • Seeded with 10 materials + 6 documented failure patterns       │
│                                                                    │
│   Components: knowledge_graph/graph.py, rules.py, seeder.py        │
└──────────────────────────┬─────────────────────────────────────────┘
                           │
                           │  MinIO upload + PostgreSQL CRUD
                           │
┌──────────────────────────▼─────────────────────────────────────────┐
│                      EXECUTION LAYER                               │
│                                                                    │
│   FEniCSx + PostgreSQL + MinIO                                     │
│                                                                    │
│   • Solves physics (FEM weak form, PETSc linear algebra)           │
│   • Persists structured metadata + agent_run_logs (full trace)     │
│   • Archives output files automatically to MinIO every run         │
│   • Manages run lifecycle (PENDING → RUNNING → SUCCESS)            │
│                                                                    │
│   Components: heat_equation.py, fenics_runner_api.py,              │
│               database/models.py, database/operations.py           │
└──────────────────────────┬─────────────────────────────────────────┘
                           │
                           │  Files (shared Docker volume) + REST API
                           │
┌──────────────────────────▼─────────────────────────────────────────┐
│                     PRESENTATION LAYER                             │
│                                                                    │
│   Plotly Dash + MLflow + FastAPI                                   │
│                                                                    │
│   • Visualises temperature fields (heatmap/surface/volume/anim.)   │
│   • Run Explorer: browse runs, agent timeline, files, recs         │
│   • Exposes async REST API + /kg/* + /explorer/* endpoints         │
│   • Tracks experiment parameters and metrics (MLflow)              │
│   • Natural language chat (Enter-key submit, quick-prompt buttons) │
│                                                                    │
│   Components: visualization/dashboard.py, orchestrator/api.py,    │
│               MLflow server                                        │
└────────────────────────────────────────────────────────────────────┘
```

### The key design principle: replaceable layers

Each layer communicates with the others through **well-defined contracts** (REST APIs,
Bolt protocol, and file formats). This means:

- Swap **FEniCSx for OpenFOAM** (CFD) in the execution layer → agents still work
  unchanged as long as the `/run` endpoint returns the same JSON shape
- Swap **qwen2.5-coder for GPT-4** in the intelligence layer → execution and
  presentation layers are unaffected
- Swap **Dash for Streamlit** in the presentation layer → nothing else changes
- Swap **Neo4j for Memgraph** in the knowledge layer → only `knowledge_graph/graph.py`
  needs updating (same Cypher dialect)
- Add a **fourth agent** (e.g. a mesh-generation agent) → just add tools and a
  new node to the orchestrator graph
- Add a **new PDE** → seed `knowledge_graph/seeder.py` with its failure patterns;
  the KG will populate automatically once runs complete

---

## Quick Reference

### Service URLs

All web UIs are accessed through **nginx on port 8050** (single entry point):

| URL | Purpose |
|-----|---------|
| http://`host`:8050/ | Dashboard (Field Viewer, Run Explorer, Chat) |
| http://`host`:8050/agents/docs | FastAPI Swagger UI (all endpoints incl. /kg/ /explorer/) |
| http://`host`:8050/mlflow/ | MLflow experiment tracking |
| http://`host`:8050/browser/ | Neo4j Browser (log in: `neo4j` / `pde_graph_secret`) |
| http://`host`:9001 | MinIO object storage console (direct — port 9001) |
| http://`host`:8000 | Agents REST API (also proxied via `/agents/`) |
| http://`host`:11434 | Ollama LLM API |

> Replace `host` with `localhost` for local access or the server's IP/hostname when
> connecting remotely. When only port 8050 is open in the firewall, use an SSH tunnel
> to access MinIO: `ssh -L 19001:localhost:9001 -N <server>`

### Key files

| File | Purpose |
|------|---------|
| `docker-compose.yml` | All service definitions (incl. Neo4j) |
| `simulations/solvers/heat_equation.py` | FEM solver |
| `docker/fenics_runner_api.py` | FEniCSx HTTP API |
| `agents/base_agent.py` | ReAct loop + tool-call parser + agent_run_logs instrumentation |
| `orchestrator/api.py` | FastAPI + async job + /kg/ + /explorer/ endpoints |
| `visualization/dashboard.py` | Dash UI: field viewer, Run Explorer, Agent Chat |
| `database/models.py` | SQLAlchemy ORM schema (incl. AgentRunLog) |
| `database/operations.py` | CRUD + log_agent_step + search_runs |
| `knowledge_graph/graph.py` | SimulationKnowledgeGraph core class |
| `knowledge_graph/rules.py` | Rule-based pre-run warning engine (8 rules) |
| `knowledge_graph/seeder.py` | Static physical knowledge (materials + failure patterns) |
| `tools/knowledge_tools.py` | check_config_warnings + query_knowledge_graph tools |

### Data flow summary

```
User text
  → LLM (Ollama)              : intent understanding + tool selection
  → check_config_warnings()   : rule engine + KG similarity check (before every run)
  → Neo4j (Knowledge Graph)   : similar past runs, material info, failure patterns
  → Python tool function      : actual computation / DB query / API call
  → FEniCSx                   : PDE solving (if simulation tool)
  → PostgreSQL                : metadata + agent_run_logs storage
  → MinIO                     : binary storage (auto-upload every run)
  → Neo4j                     : add_run() — new Run node linked to material + issues
  → numpy files               : field data (u_final.npy, snapshots/)
  → scipy.griddata            : scattered→regular grid interpolation
  → Plotly                    : interactive web charts
  → Browser                   : user sees result + can browse Run Explorer
```

---

*Document updated: PDE Agents v1.2  
FEniCSx 0.10.0.post2 · LangGraph · Ollama · Plotly Dash · PostgreSQL · Neo4j · MinIO · Nginx*
