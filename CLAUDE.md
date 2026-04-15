# CLAUDE.md — PDE Agents Project Context

## Project Overview

**PDE Agents** is a multi-agent AI ecosystem for solving partial differential equations (PDEs) using the Finite Element Method (FEM). It combines locally-running open-source LLMs (via Ollama), a GraphRAG knowledge graph (Neo4j), a document intelligence pipeline, and a Plotly Dash visualization dashboard — all orchestrated through LangGraph.

**Hardware target:** 2× NVIDIA RTX PRO 6000 Blackwell Server Edition (~196 GB VRAM total), CUDA 13.1
**FEM Solver:** DOLFINx (FEniCSx) `0.10.0.post2`

## Architecture

### Multi-Agent System (LangGraph)

The system uses a **supervisor/orchestrator pattern** where a central orchestrator LLM routes tasks to three specialist agents:

1. **Simulation Agent** (`agents/simulation_agent.py`) — Sets up, runs, monitors, and debugs FEM simulations. Model: `qwen2.5-coder:32b`. Max 15 iterations.
2. **Analytics Agent** (`agents/analytics_agent.py`) — Analyzes results, compares runs, suggests improvements. Model: `llama3.3:70b`. Max 12 iterations.
3. **Database Agent** (`agents/database_agent.py`) — Stores results, answers history queries, manages data. Model: `qwen2.5-coder:14b`. Max 10 iterations.

**Orchestrator** (`orchestrator/graph.py`) — Supervisor LangGraph that uses `llama3.3:70b` with JSON output to decide which agent to call next. Flow: supervisor → agent → supervisor → ... → synthesize → END.

**Base Agent** (`agents/base_agent.py`) — All agents extend `BaseAgent`, a ReAct-style LangGraph `StateGraph` with nodes: `reason` → `act` (tool execution) → `reason` → ... → `finish`. Includes a robust fallback parser for qwen models that emit tool calls as JSON text in content rather than structured `tool_calls`.

### Key Design Decisions
- Agents log every reasoning step, tool call, and result to `agent_run_logs` in PostgreSQL
- `run_id` is back-filled into log rows once it becomes known from tool results
- Each agent invocation gets a unique `task_id` (UUID hex)
- Tool-calling compatibility: llama3.3 uses native Ollama tool_calls; qwen2.5-coder outputs JSON text parsed by `_parse_content_tool_call`

## Service Architecture (Docker Compose)

11 containers on a shared `pde-net` bridge network:

| Container | Service | Port | Purpose |
|-----------|---------|------|---------|
| `pde-ollama` | Ollama LLM | 11434 | Local LLM inference (GPU-accelerated) |
| `pde-postgres` | PostgreSQL 16 | 5432 | Simulation metadata & results |
| `pde-redis` | Redis 7 | 6379 | Celery broker + result backend |
| `pde-minio` | MinIO | 9000/9002 | Object storage for VTK/HDF5/mesh files |
| `pde-neo4j` | Neo4j 5 Community | 7474/7687 | Knowledge graph (materials, runs, references) |
| `pde-fenics` | FEniCSx Runner | 8080/8888 | FEM solver + JupyterLab |
| `pde-agents` | Agent Service | 8000 | FastAPI orchestrator + Celery doc worker |
| `pde-dashboard` | Dash Dashboard | (via nginx) | Plotly Dash visualization |
| `pde-nginx` | Nginx | 80/8050 | Reverse proxy (single entry point) |
| `pde-neodash` | NeoDash | 9001 | Neo4j graph visualization UI |

### Inter-container networking
- Agents connect to Ollama at `http://ollama:11434`
- Agents connect to FEniCS runner at `http://fenics-runner:8080`
- Agents connect to PostgreSQL at `postgres:5432`
- Agents connect to Neo4j at `bolt://neo4j:7687`
- Agents connect to MinIO at `minio:9000`
- Nginx proxies: `/` → dashboard, `/agents/` → agents API, `/browser/` → Neo4j Browser

## Directory Structure

```
pde-agents/
├── agents/                    # LangGraph agent implementations
│   ├── base_agent.py          # BaseAgent class (ReAct graph, tool-call parser)
│   ├── simulation_agent.py    # Agent-1: FEM simulation setup/run/debug
│   ├── analytics_agent.py     # Agent-2: Result analysis & suggestions
│   └── database_agent.py      # Agent-3: Storage, queries, cataloging
├── orchestrator/
│   ├── api.py                 # FastAPI REST interface (~1200 lines)
│   └── graph.py               # Multi-agent orchestrator (supervisor pattern)
├── tools/                     # LangChain @tool functions (agent capabilities)
│   ├── simulation_tools.py    # run_simulation, validate_config, debug, sweep
│   ├── analytics_tools.py     # analyze_run, compare_runs, suggest_next_run
│   ├── database_tools.py      # store_result, query_runs, search_history
│   └── knowledge_tools.py     # query_knowledge_graph, check_config_warnings
├── database/
│   ├── models.py              # SQLAlchemy ORM (SimulationRun, RunResult, etc.)
│   ├── operations.py          # CRUD operations (create_run, mark_run_finished, etc.)
│   └── init.sql               # PostgreSQL extensions (uuid-ossp, pg_trgm)
├── knowledge_graph/
│   ├── graph.py               # SimulationKnowledgeGraph class (~1840 lines)
│   ├── seeder.py              # Materials + KnownIssues data
│   ├── rules.py               # Rule-based config warning engine
│   ├── embeddings.py          # OllamaEmbedder (nomic-embed-text, 768-dim)
│   ├── references.py          # Curated physics reference data
│   ├── document_processor.py  # Docling PDF/DOCX structured extraction
│   ├── tasks.py               # Celery async tasks for document ingestion
│   └── web_fetcher.py         # Web resource crawler + ingester
├── simulations/
│   ├── solvers/
│   │   └── heat_equation.py   # HeatEquationSolver (DOLFINx FEM, ~920 lines)
│   ├── geometry/
│   │   └── gmsh_geometries.py # Gmsh geometry builders (9 types, ~608 lines)
│   └── configs/
│       ├── heat_2d.json       # Example 2D steel plate config
│       └── heat_3d.json       # Example 3D aluminum block config
├── visualization/
│   └── dashboard.py           # Plotly Dash multi-tab dashboard (~3676 lines)
├── docker/
│   ├── Dockerfile.agents      # Python 3.11 + requirements
│   ├── Dockerfile.dashboard   # Python 3.11 + Dash deps
│   └── Dockerfile.fenics      # dolfinx/dolfinx:stable + FastAPI
│   └── fenics_runner_api.py   # FastAPI server inside FEniCSx container
├── scripts/
│   ├── seed_knowledge_graph.py     # Seed KG with simulation runs
│   ├── seed_bc_geometry_study.py   # BC + geometry parametric studies
│   ├── seed_neodash_dashboard.py   # NeoDash dashboard config
│   ├── sweep_full_study.py         # Full 8-study parametric sweep (~848 runs)
│   └── migrate_kg_schema_v2.py     # KG schema migration v1→v2
├── nginx/nginx.conf           # Reverse proxy configuration
├── docker-compose.yml         # Full stack definition (11 services)
├── Makefile                   # CLI shortcuts (make run, make simulate-2d, etc.)
├── setup.sh                   # One-time setup script
├── requirements.txt           # Python dependencies
└── .env                       # Environment variables (gitignored)
```

## Database Schema (PostgreSQL)

**ORM:** SQLAlchemy with `psycopg2-binary`. All operations are synchronous.

### Tables
- **simulation_runs** — Core record per run: `run_id` (unique string), `status` (pending/running/success/failed/cancelled), `dim`, `nx/ny/nz`, `k/rho/cp/source`, `dt/t_end/theta`, `config_json` (full config), `wall_time`, `output_dir`, `minio_prefix`
- **run_results** — Scalar results per run: `t_max/t_min/t_mean/t_std`, `final_l2_norm`, `converged`, `max_heat_flux`
- **convergence_history** — Per-timestep L2 norms: `step`, `time`, `l2_norm`, `t_max`, `t_min`
- **run_parameters** — Key-value config store for arbitrary parameter queries
- **parametric_studies** — Named groups of related sweep runs: `study_id`, `swept_parameter`, `parameter_values`
- **study_runs** — Many-to-many link: study ↔ run with `param_value`
- **agent_run_logs** — Step-by-step agent reasoning trace: `task_id`, `agent_name`, `step_type` (reasoning/tool_call/tool_result/final_answer), `content` (JSON), `elapsed_ms`
- **agent_messages** — Inter-agent communication audit log
- **agent_suggestions** — Analytics agent improvement suggestions with priority

### Key Operations (`database/operations.py`)
- `create_run()` — Overwrites if same run_id exists (upsert semantics)
- `mark_run_started/finished/failed()` — Status transitions
- `log_agent_step()` — Never raises (logging must not break agents)
- `backfill_task_run_id()` — Retroactively tags log rows once run_id is known
- `get_agent_logs()` — Full step-by-step trace for a run
- `search_runs()` — Advanced search with joins across runs and results

## Knowledge Graph (Neo4j)

### Node Types
- **Run** — One per simulation. Carries 768-dim `embedding` vector for semantic search.
- **Material** — Engineering materials (10 seeded: Steel, Aluminium, Copper, Titanium, Silicon, Concrete, Water, Air, Glass, Stainless Steel)
- **KnownIssue** — Documented failure patterns (6 seeded: GIBBS_OVERSHOOT, EXPLICIT_INSTABILITY, MESH_TOO_COARSE, NO_STEADY_STATE_REACHED, PURE_NEUMANN_ILL_POSED, NEGATIVE_TEMPERATURES)
- **BCConfig** — Boundary-condition pattern (e.g., "dirichlet+neumann", "dirichlet+robin")
- **Domain** — Physical domain size class (micro/component/panel/structural)
- **ThermalClass** — Conductivity classification (high_conductor/medium_conductor/low_conductor/thermal_insulator)
- **Reference** — Curated or uploaded external documents with embeddings
- **ReferenceChunk** — Structured chunks from parsed documents with 768-dim embeddings

### Relationships
- `(:Run)-[:USES_MATERIAL {confidence}]->(:Material)`
- `(:Run)-[:TRIGGERED {detected_at}]->(:KnownIssue)`
- `(:Run)-[:USES_BC_CONFIG]->(:BCConfig)`
- `(:Run)-[:ON_DOMAIN]->(:Domain)`
- `(:Material)-[:HAS_THERMAL_CLASS]->(:ThermalClass)`
- `(:Run)-[:SPAWNED_FROM]->(:Run)` — suggestion lineage
- `(:Run)-[:SIMILAR_TO {score}]->(:Run)` — KNN semantic similarity
- `(:Reference)-[:HAS_CHUNK]->(:ReferenceChunk)`
- `(:ReferenceChunk)-[:CROSS_REFS {score}]->(:Run)` — chunk↔run similarity
- `(:Run)-[:CITES]->(:Reference)` — manual or auto-linked

### Vector Search
- HNSW index on `Run.embedding` (768-dim, cosine similarity)
- HNSW index on `ReferenceChunk.embedding` (768-dim, cosine similarity)
- HNSW index on `Reference.embedding` (768-dim, cosine similarity)
- Embedding model: `nomic-embed-text` via Ollama
- Fallback: Cypher parameter-distance query when embeddings unavailable

### Pre-Run Context (`get_pre_run_context`)
Before any simulation, the KG provides: rule-based warnings, similar past runs (semantic or Cypher), inferred material, past failure patterns, BC pattern statistics, domain size statistics, thermal class insights, and physics references.

## FEM Solver (`simulations/solvers/heat_equation.py`)

Solves: **ρ c_p ∂u/∂t - ∇·(k ∇u) = f**

### Key Classes
- `HeatConfig` — Dataclass with all simulation parameters
- `HeatEquationSolver` — Main solver class
- `SimulationResult` — Result dataclass with `to_dict()` and `summary()`

### Features
- 2D/3D transient heat conduction
- θ-scheme time integration (θ=1.0 Backward Euler, θ=0.5 Crank-Nicolson)
- Dirichlet, Neumann, and Robin (convective) boundary conditions
- Built-in rectangular/box meshes + Gmsh custom geometries (L-shape, circle, annulus, T-shape, etc.)
- PETSc KSP solvers (CG + Hypre by default)
- Saves: `config.json`, `result.json`, `u_final.npy`, `dof_coords.npy`, snapshots, XDMF

### Mesh Sources
1. **Built-in:** DOLFINx `create_rectangle`/`create_box` with boundary locations: left/right/top/bottom/front/back
2. **Gmsh:** 9 geometry types via `simulations/geometry/gmsh_geometries.py` — rectangle, l_shape, circle, annulus, hollow_rectangle, t_shape, stepped_notch (2D); box, cylinder (3D)

## API Endpoints (`orchestrator/api.py`)

### Core
- `POST /run` — Full multi-agent orchestrator (sync)
- `POST /run/async` — Background job, returns `job_id`
- `POST /simulate` — Direct Simulation Agent invocation
- `POST /analyze` — Direct Analytics Agent invocation
- `POST /query` — Direct Database Agent invocation
- `POST /agent/{name}` — Generic agent endpoint (sync)
- `POST /agent/{name}/async` — Generic agent endpoint (async)
- `GET /jobs/{id}` — Poll background job status
- `GET /jobs` — List all jobs

### Run Explorer
- `GET /explorer/runs` — List runs with agent log step counts
- `GET /explorer/runs/{id}/detail` — Full detail (config, results, logs, MinIO files, suggestions)
- `GET /explorer/runs/{id}/logs` — Agent trace
- `GET /explorer/runs/{id}/files` — MinIO file listing
- `POST /explorer/search` — Cross-search (text, status, dim, k range, T_max)

### Knowledge Graph
- `GET /kg/stats` — Node/relationship counts
- `POST /kg/seed` — Seed materials + known issues
- `GET /kg/material/{name}` — Material lookup
- `GET /kg/run/{id}/similar` — Vector similarity search
- `GET /kg/run/{id}/lineage` — SPAWNED_FROM ancestry
- `GET /kg/run/{id}/references` — All linked references

### Document Intelligence
- `POST /references/upload` — Upload PDF/TXT → KG (immediate + async Celery)
- `POST /references/fetch-url` — Crawl web resource → KG (async Celery)
- `GET /references/uploaded` — List uploaded references
- `GET /references/{id}/chunks` — Structured chunks with cross-ref stats
- `GET /references/{id}/status` — Processing status
- `GET /references/search-chunks` — Semantic search across all chunks
- `POST /references/{ref_id}/link/{run_id}` — Manual reference→run link

### Other
- `GET /health` — Service health + Ollama model list
- `GET /runs` — List simulation runs
- `GET /runs/{id}` — Run details
- `WS /ws/stream` — WebSocket for streaming agent output

## Tool Reference

### Simulation Tools (`tools/simulation_tools.py`)
- `run_simulation(config_json)` — Launch FEM run via FEniCS runner API. Auto-registers in DB, uploads to MinIO, populates KG.
- `validate_config(config_json)` — Check config validity, estimate DOFs/memory/time.
- `modify_config(config_json, changes_json)` — Patch config fields.
- `debug_simulation(run_id)` — Diagnose failures (divergence, OOM, convergence).
- `list_recent_runs(limit, status_filter)` — Recent runs from DB.
- `get_run_status(run_id)` — Current status and metrics.
- `run_parametric_sweep(swept_parameter, values_json, base_config_json)` — Sweep over parameter values.

### Analytics Tools (`tools/analytics_tools.py`)
- `analyze_run(run_id)` — Statistical analysis (temperature stats, convergence, performance).
- `compare_runs(run_ids_json)` — Side-by-side comparison with uniformity ranking.
- `compare_study(study_id)` — Parametric study sensitivity analysis.
- `suggest_next_run(analysis_json, strategy)` — Suggest next config (optimize_uniformity/refine_mesh/reduce_time/explore).
- `get_steady_state_time(run_id, tolerance)` — Estimate when steady state reached.
- `list_runs_for_analysis(status, dim, k_min, k_max, limit)` — Discover available runs.
- `export_summary_report(run_ids_json, output_path)` — JSON summary report.

### Database Tools (`tools/database_tools.py`)
- `store_result(result_json)` — Persist to PostgreSQL + MinIO.
- `query_runs(status, dim, limit)` — Basic run listing.
- `search_history(status, dim, text, k_min, k_max, t_max_min, limit)` — Rich search with joins.
- `get_run_summary(run_id)` — Comprehensive single-run summary (config, results, agent logs, suggestions).
- `catalog_study(...)` — Register parametric study.
- `fetch_run_data(run_id, include_convergence)` — Full run record.
- `export_to_csv(status, dim, limit, output_path)` — CSV export.
- `db_health_check()` — Connectivity + stats.
- `upload_to_minio(local_path, bucket, object_name)` — File upload.

### Knowledge Tools (`tools/knowledge_tools.py`)
- `query_knowledge_graph(question, material, k, dim, run_id, bc_pattern, domain_label)` — General-purpose KG query (material lookup, similar runs, BC stats, domain stats, lineage, thermal class).
- `check_config_warnings(config_json)` — Pre-run validation: rule warnings + similar runs + inferred material + BC/domain/thermal insights + physics references.
- `get_physics_references(config_json)` — Retrieve curated physics references for a config.

## Rule-Based Warning Engine (`knowledge_graph/rules.py`)

8 rules checked before every simulation:
- `INCONSISTENT_IC` (high) — u_init far from Dirichlet BC values
- `EXPLICIT_CFL_VIOLATION` (high) — dt exceeds CFL limit for θ<0.5
- `NEAR_EXPLICIT_SCHEME` (medium) — θ<0.5 gives conditionally stable scheme
- `COARSE_MESH_2D` (medium) — nx or ny < 10 in 2D
- `COARSE_MESH_3D` (medium) — nx/ny/nz < 8 in 3D
- `SHORT_SIMULATION` (low) — fewer than 5 time steps
- `INVALID_MATERIAL_PROPS` (high) — non-positive k/rho/cp
- `NO_BOUNDARY_CONDITIONS` (high) — no BCs specified

## Document Intelligence Pipeline

### Flow
1. User uploads PDF/TXT/DOCX via `POST /references/upload`
2. Immediate: extract text, create Reference node, auto-link to similar runs via vector search
3. Async (Celery): Docling structured parsing → classify chunks (material/bc/solver/domain/general) → embed each chunk → store as ReferenceChunk nodes → cross-reference to Runs, Materials, BCConfigs, Domains
4. Web resources: `POST /references/fetch-url` crawls and processes similarly

### Components
- `knowledge_graph/document_processor.py` — Docling PDF extraction + HybridChunker + physics domain classification
- `knowledge_graph/tasks.py` — Celery tasks (`ingest_document_task`, `ingest_web_resource_task`)
- `knowledge_graph/web_fetcher.py` — Web page crawler with same-path scope
- `knowledge_graph/embeddings.py` — OllamaEmbedder wrapping nomic-embed-text (768-dim)

## Seeded Materials

| Material | k (W/m·K) | ρ (kg/m³) | cp (J/kg·K) | k range |
|----------|-----------|-----------|-------------|---------|
| Steel (carbon) | 50 | 7800 | 500 | 40–60 |
| Stainless Steel (316) | 16 | 8000 | 500 | 14–18 |
| Aluminium (6061) | 200 | 2700 | 900 | 160–240 |
| Copper | 385 | 8960 | 385 | 370–400 |
| Titanium (Ti-6Al-4V) | 6.7 | 4430 | 526 | 5–8 |
| Silicon | 150 | 2330 | 700 | 100–160 |
| Concrete | 1.7 | 2300 | 880 | 0.8–2.5 |
| Water | 0.6 | 1000 | 4182 | 0.55–0.65 |
| Air (approx.) | 0.026 | 1.2 | 1005 | 0.024–0.030 |
| Glass (borosilicate) | 1.0 | 2230 | 830 | 0.8–1.2 |

## Environment Variables

Key variables (from `.env` / `env.example`):

```
POSTGRES_DB=pde_simulations    POSTGRES_USER=pde_user
REDIS_URL=redis://localhost:6379/0
MINIO_ENDPOINT=localhost:9000  MINIO_ROOT_USER=minio_admin
OLLAMA_BASE_URL=http://localhost:11434
SIM_MODEL=qwen2.5-coder:32b   ANALYTICS_MODEL=llama3.3:70b   DB_MODEL=qwen2.5-coder:14b
FENICS_RUNNER_URL=http://localhost:8080
NEO4J_URI=bolt://neo4j:7687   NEO4J_USER=neo4j   NEO4J_PASSWORD=pde_graph_secret
SERVER_HOST=<server-ip>        # Required for NeoDash Bolt WebSocket
```

Inside containers, `localhost` references are replaced with Docker service names (e.g., `postgres`, `ollama`, `neo4j`, `minio`, `fenics-runner`).

## Common Commands

```bash
make run              # Start all services
make stop             # Stop all services
make simulate-2d      # Run 2D steel plate simulation
make simulate-3d      # Run 3D aluminum block simulation
make sweep            # Parametric sweep over k values
make health           # Check all service health
make logs             # Tail all service logs
make db-shell         # PostgreSQL interactive shell
make shell-fenics     # Bash into FEniCSx container
make shell-agents     # Bash into agents container
make list-runs        # Show recent simulation runs
make list-jobs        # Show background API jobs
make check-run RUN_ID=<id>   # Inspect a specific run
make seed-neodash     # Seed NeoDash dashboard config
```

## Development Notes

- **No virtual environment** — all Python code runs inside Docker containers
- **Hot reload** — agents, tools, orchestrator, database, and knowledge_graph directories are bind-mounted into the agents container; dashboard and visualization are bind-mounted into the dashboard container
- **agents container** starts both a Celery worker (document ingestion queue) and the FastAPI server (uvicorn with --reload)
- **KG auto-seeds** on agent startup (`_startup_seed_kg` in `api.py`): initializes schema, seeds materials/issues if empty
- All KG operations degrade gracefully when Neo4j is unavailable — agents still work, just without knowledge graph context
- All DB logging operations never raise — logging must not break agent execution
- `run_simulation` auto-uploads to MinIO and auto-populates KG after every successful run
- The dashboard is a single 3676-line Plotly Dash app with tabs for overview, run explorer, convergence, parametric studies, knowledge graph, and document references
- Gmsh geometries support 9 types (7 2D + 2 3D) with named boundary groups for BC specification

## Coding Conventions

- Python 3.11, `from __future__ import annotations` in all modules
- Type hints throughout, `Optional` for nullable fields
- SQLAlchemy ORM with context-managed sessions (`get_db()`)
- LangChain `@tool` decorator for all agent-callable functions
- Tools return JSON strings (not dicts) for LangChain compatibility
- Graceful degradation everywhere: DB, KG, MinIO, Ollama — all optional
- `try/except ImportError` guards for environment-specific imports (DOLFINx only in fenics container, minio optional, etc.)

## Research Paper Context

### Target Contribution

The project's core research claim: **Autonomous multi-agent LLM systems with GraphRAG memory can conduct scientific simulations — setting up, running, analyzing, and iteratively improving FEM simulations — with physics-informed reasoning grounded in past experience and curated knowledge.**

This sits at the intersection of: AI for Science, multi-agent systems, knowledge-augmented generation (GraphRAG), and computational mechanics.

### Novel Contributions (Strengths)

1. **Multi-agent orchestration for FEM**: A supervisor-pattern LangGraph system where specialized LLM agents (Simulation, Analytics, Database) autonomously conduct the full simulation lifecycle — from natural language task description to converged FEM results. No prior work combines LangGraph multi-agent orchestration with production FEM solvers.

2. **GraphRAG for scientific computing memory**: A Neo4j knowledge graph that serves as institutional memory for simulations — every run is embedded (768-dim nomic-embed-text), linked to materials, BC patterns, domain classes, and known failure modes. Agents query this graph *before* running simulations to get pre-run intelligence (similar past runs, expected outcomes, physics warnings). This is a novel application of retrieval-augmented generation to computational physics.

3. **Self-improving simulation loop**: The orchestrator creates a closed loop: run → analyze → suggest improvements → run again. Run lineage (SPAWNED_FROM edges) traces the chain of improvements. The system gets better at a problem the more it iterates.

4. **Rule-based + semantic pre-run validation**: Combining deterministic physics rules (CFL checks, BC consistency, mesh quality) with semantic similarity search gives agents both hard constraints and soft experiential knowledge before committing to expensive compute.

5. **Document intelligence pipeline**: Structured extraction of scientific PDFs/DOCX via IBM Docling, physics-domain chunk classification (material/bc/solver/domain), per-chunk embeddings, and automatic cross-referencing between literature chunks and simulation runs. This is a concrete implementation of "grounding AI agents in scientific literature."

6. **Full-stack autonomous system**: Not a toy demo — 11 Docker services, real DOLFINx FEM solver, PostgreSQL for structured data, MinIO for large files, Neo4j for graph queries, Redis/Celery for async processing, Plotly Dash for visualization. The system actually runs simulations end-to-end.

### Gaps to Address for Publication

#### Critical (must-have for any venue)

1. **Quantitative evaluation framework**: Need systematic experiments measuring:
   - Agent success rate: % of natural language tasks that produce correct simulations
   - Decision quality: compare agent-chosen configs vs expert-chosen configs
   - KG impact: simulation outcomes WITH vs WITHOUT knowledge graph context
   - Iteration efficiency: how many orchestrator loops to converge on a good solution
   - Wall-clock comparison: autonomous pipeline vs manual engineering workflow

2. **Verification & Validation (V&V)**: Compare solver output against:
   - Analytical solutions (1D/2D steady-state heat with known closed forms)
   - Method of manufactured solutions (MMS) for convergence order verification
   - Published benchmark problems (e.g., NAFEMS thermal benchmarks)
   - Mesh convergence studies showing spatial order of accuracy (h-refinement)
   - Time convergence studies showing temporal order (dt-refinement)

3. **Ablation studies** isolating the value of each component:
   - Full system vs agents without KG
   - Agents with KG vs agents with only rule-based warnings (no vector search)
   - Orchestrator loop vs single-shot agent invocations
   - Pre-run context (similar runs + physics references) vs no pre-run context
   - Semantic similarity (vector search) vs parameter-distance (Cypher fallback)
   - Document intelligence pipeline: does ingesting a relevant paper improve subsequent simulation decisions?

4. **Failure mode analysis**:
   - LLM hallucination rate for simulation configs (invalid parameters, physically impossible setups)
   - Orchestrator infinite loop frequency and mitigation effectiveness
   - Agent disagreement cases (Analytics suggests change that Simulation rejects)
   - Graceful degradation testing: behavior when Neo4j is down, when Ollama is slow, when FEniCSx fails

#### Important (needed for top venues)

5. **Generalization beyond heat equation**: Implement at least one additional PDE type to demonstrate architecture generality:
   - Linear elasticity (stress/strain, different BC types)
   - Advection-diffusion (adds convective transport)
   - Stokes/Navier-Stokes (fluid flow, fundamentally different physics)
   - Even Poisson equation (steady-state, simplest case) would add breadth

6. **Scalability analysis**:
   - Performance at different problem sizes (1K, 10K, 100K, 1M DOFs)
   - KG query latency vs number of stored runs (100, 1000, 10000)
   - LLM inference cost breakdown (tokens, time, VRAM per agent invocation)
   - End-to-end latency profiling (where is time spent: LLM reasoning vs FEM solve vs DB/KG queries)

7. **Baseline comparisons**:
   - Manual expert workflow (time, iterations, outcome quality)
   - Single-agent (one LLM does everything) vs multi-agent system
   - Commercial tools (COMSOL scripting, Ansys Workbench automation) if accessible
   - Other AI-for-PDE approaches (PINN via DeepXDE, operator learning via DeepONet) — different paradigm but same problem domain

8. **Reproducibility package**:
   - Automated test suite (the `tests/` directory is currently empty)
   - Deterministic seed runs that produce known results
   - Paper figure generation scripts
   - Hardware-independent timing normalization

#### Nice-to-have (differentiators for high impact)

9. **Human-in-the-loop study**: Have 5-10 domain experts use the system and rate:
   - Quality of agent decisions vs their own choices
   - Usefulness of KG-provided pre-run context
   - Trust in the automated analysis
   - Time savings compared to their normal workflow

10. **Multi-physics coupling**: Show agents coordinating across coupled physics (e.g., thermal stress: heat equation feeds into elasticity)

11. **Transfer learning across PDE types**: Show that knowledge gained from heat equation runs (mesh quality intuitions, convergence patterns) transfers when adding a new PDE type

12. **Formal problem formulation**: Mathematical description of the multi-agent simulation optimization as a decision process: state space (simulation configs), action space (agent tool calls), reward (solution quality metrics), transition dynamics (FEM solver)

### Suggested Paper Structure

```
1. Introduction
   - The gap: scientific simulation requires deep expertise; LLMs show promise
     for automation but lack domain grounding and memory
   - Our approach: multi-agent LLMs + GraphRAG + FEM

2. Related Work
   - AI for PDE solving (PINNs, operator learning, foundation models)
   - LLM agents for science (ChemCrow, Coscientist, etc.)
   - Knowledge graphs in scientific computing
   - Multi-agent LLM systems (LangGraph, AutoGen, CrewAI)

3. System Architecture
   - Multi-agent orchestrator design (supervisor pattern)
   - Agent tool interfaces and FEM solver integration
   - Knowledge graph schema and pre-run intelligence system
   - Document intelligence pipeline

4. Experimental Setup
   - Benchmark problems (analytical solutions, NAFEMS, manufactured solutions)
   - Evaluation metrics (accuracy, efficiency, decision quality)
   - Ablation study design

5. Results
   - V&V: solver correctness against analytical benchmarks
   - Agent evaluation: success rates, decision quality, iteration counts
   - KG impact: with vs without knowledge graph (the key result)
   - Scalability: problem size, KG size, LLM cost
   - Case studies: full autonomous investigations end-to-end

6. Discussion
   - When agents succeed vs fail
   - Knowledge graph as institutional memory for scientific computing
   - Limitations and failure modes
   - Generalization prospects

7. Conclusion
```

### Target Venues (by fit)

| Venue | Fit | Key Requirement |
|-------|-----|-----------------|
| Nature Computational Science | High impact, broad audience | Need to show paradigm shift + strong V&V |
| Journal of Computational Physics | Core audience (computational scientists) | Rigorous V&V, convergence studies, benchmarks |
| NeurIPS / ICML (AI for Science track) | ML audience, novel method | Ablations, baselines, quantitative metrics |
| Computer Methods in Applied Mechanics and Engineering | Engineering audience | V&V, scalability, practical applicability |
| AAAI / IJCAI | AI audience | Multi-agent system novelty, knowledge representation |
| Scientific Reports / PLOS ONE | Broad open-access | Complete system description + basic evaluation |

### Existing Data Assets

The system already has data that can support paper experiments:
- PostgreSQL contains all past simulation runs with full configs, results, convergence histories, and agent decision logs
- Neo4j contains the knowledge graph with run embeddings, material links, BC patterns, similar-run edges
- MinIO stores all output files (temperature fields, meshes, VTK/XDMF)
- Agent run logs capture every LLM reasoning step, tool call, and result — rich data for analyzing agent behavior
- The `scripts/sweep_full_study.py` script defines 8 parametric studies (~848 runs) that can serve as the experimental dataset

### Key Metrics to Report

For the paper, instrument and report these metrics:
- **Simulation success rate**: successful_runs / total_runs (by agent-initiated vs manual)
- **Config quality score**: compare agent-generated configs to expert baseline (parameter distance, physical plausibility)
- **KG retrieval relevance**: precision@k of similar-run retrieval (are returned runs actually useful?)
- **Iteration efficiency**: number of orchestrator cycles to achieve target metric (e.g., uniformity index > 0.95)
- **LLM cost**: tokens consumed, inference time, VRAM utilization per task
- **V&V metrics**: L2 error norm vs analytical solution, convergence order (should be O(h²) for P1 elements)
- **Wall-clock speedup**: end-to-end time for autonomous pipeline vs estimated manual time

---

## Evaluation Framework

All evaluation code lives in `evaluation/`. Run everything via Makefile targets.

### 1. Verification & Validation Benchmarks (`evaluation/benchmarks/`)

Validates the FEM solver against closed-form analytical solutions. Runs inside the FEniCSx container.

**Files:**
- `analytical_solutions.py` — Closed-form solutions: steady linear, Laplace sinusoidal, Fourier mode decay, constant source (Poisson), transient step response
- `vv_runner.py` — Runs each benchmark at mesh resolutions N = 8, 16, 32, 64, 128; computes proper L2 and L∞ error norms via DOLFINx integration; fits convergence rate via log-log regression; expects O(h²) for P1 elements

**Benchmark cases:**
| Case | PDE Type | BCs | Analytical Reference |
|------|----------|-----|---------------------|
| steady_linear_2d | Laplace | Dirichlet left/right, Neumann top/bottom | T(x) = T_L + (T_R - T_L)x/L |
| steady_sinusoidal_2d | Laplace | Dirichlet all sides (sin(πx) on top) | sin(πx)sinh(πy)/sinh(π) |
| transient_fourier_2d | Heat eq. | Homogeneous Dirichlet all sides | sin(πx)sin(πy)exp(-2απ²t) |
| steady_source_2d | Poisson | Dirichlet left/right, Neumann top/bottom | f/(2k) x(L-x) |
| transient_step_2d | Heat eq. | Dirichlet left/right, Neumann top/bottom | Fourier series (200 terms) |

**Run:** `make eval-vv`
**Output:** `evaluation/results/vv_results.json`

### 2. KG Ablation Study (`evaluation/ablation/`)

Compares agent performance with and without Knowledge Graph augmentation on 10 benchmark tasks across three difficulty levels.

**Files:**
- `benchmark_tasks.py` — 10 tasks: 3 easy (explicit params), 3 medium (need material lookup), 4 hard (ambiguous/tricky numerics). Each includes ground truth for scoring.
- `run_ablation.py` — Sends each task to `/simulate` API twice: once with `disable_kg=false` (default) and once with `disable_kg=true`. Collects success rate, config quality, iteration count, wall time.

**KG toggle mechanism:**
- `SimulationAgent.__init__(disable_kg=True)` strips `check_config_warnings` and `query_knowledge_graph` from the agent's tool list
- API endpoint: `POST /simulate?disable_kg=true`
- This tests the real degradation path — same LLM, same prompts, just without KG tools

**Metrics collected per task:**
- Success (simulation completed without error)
- Config quality score (0–1): material properties within expected ranges, correct BC types
- Agent iterations (fewer = more efficient)
- First-try success (no debug/retry cycle)
- Wall time

**Run:** `make eval-ablation`
**Output:** `evaluation/results/ablation_results.json`

**Key finding (10-task run, April 2026):**
| Mode | Success | Avg Quality | Avg Iter | Avg Time |
|------|---------|-------------|----------|----------|
| KG On  | 40% | 0.26 | 4.7 | 21.4s |
| KG Off | 100% | 0.75 | 3.1 | 14.2s |

By difficulty: easy = both 100%, medium = KG Off 100% vs KG On 0%, hard = KG Off 100% vs KG On 25%.

**Interpretation:** KG-off outperforms KG-on on medium/hard tasks. The LLM's parametric
knowledge suffices for standard materials (steel, copper, aluminium). The current KG integration
adds overhead and can confuse the model when KG queries return unexpected results on complex tasks.
This is a scientifically interesting negative result — it identifies a clear design opportunity:
use KG as an optional enrichment layer rather than a mandatory first step.

**Engineering notes (from debugging this run):**
- `base_agent.py`: Added `num_predict=2048` to ChatOllama and a Pass 4 in
  `_parse_content_tool_call` that handles truncated/partial JSON tool-call output from
  qwen2.5-coder (regex-based repair + progressive bracket-closing).
- `simulation_agent.py`: KG-off mode now appends a NOTE to system prompt so the LLM
  doesn't hallucinate tool calls for unavailable KG tools.
- `run_ablation.py`: Runs direct in-process (inside agents container) to avoid
  uvicorn async/LangGraph thread-pool incompatibility. Uses pre-warmed agents and
  per-task retry (up to 3 attempts) for nondeterministic cold-starts.

### 3. Agent Decision Quality Metrics (`evaluation/metrics/`)

Mines the PostgreSQL `agent_run_logs` table for quantitative evidence of agent performance across all past runs.

**File:** `agent_quality.py`

**Metrics computed:**
- Per-agent task counts and average reasoning steps per task
- Tool call success/failure rates per tool
- Suggestion acceptance rate (analytics agent suggestions → executed runs)
- First-try simulation success rate (no debug/retry needed)
- Timing breakdown: LLM inference vs tool execution vs total wall time
- Orchestrator routing efficiency

**Run:** `make eval-metrics`
**Output:** `evaluation/results/agent_metrics.json`

### 4. LaTeX Table Generator (`evaluation/generate_tables.py`)

Reads all JSON result files and produces publication-ready LaTeX table snippets.

**Tables generated:**
- `tables/vv_convergence.tex` — Summary convergence table (case, DOFs, L2 error, rate, pass/fail)
- `tables/vv_detail.tex` — Full h-refinement data for all cases
- `tables/ablation.tex` — KG On vs KG Off comparison with difficulty breakdown
- `tables/agent_metrics.tex` — Agent ecosystem performance summary

**Run:** `make eval-tables`

### Makefile Targets

```
make eval-vv             # V&V convergence benchmarks (runs in fenics container)
make eval-ablation       # 2-way KG on/off ablation (in-process in agents container)
make eval-ablation-smart # 3-way ablation: KG On vs Off vs Smart (warm-start + lazy)
make eval-metrics        # Agent quality metrics (queries PostgreSQL)
make eval-tables         # Generate LaTeX tables from results
make eval-all            # Run eval-vv + eval-ablation + eval-metrics + eval-tables
```

### Smart KG Integration (implemented 2026-04-07)

Three KG modes in `SimulationAgent.__init__`:
- `disable_kg=False` (default "KG On"): Mandatory KG-first workflow
- `disable_kg=True` ("KG Off"): KG tools removed entirely
- `smart_kg=True` ("KG Smart"): Warm-start injection + lazy conditional retrieval

**KG Smart design** (inspired by CRAG and AriGraph):
1. **Warm-start injection**: Before agent loop, embed task description via `nomic-embed-text`,
   query Neo4j HNSW for top-3 similar past successful runs, inject configs as few-shot
   examples in system prompt (zero tool-call overhead)
2. **Lazy conditional retrieval**: KG tools remain available but prompt says "only use after
   failure or when material properties genuinely unknown" — eliminates mandatory KG-first loop

**3-way ablation results** (13 tasks, 4 difficulty levels incl. Novel):
| Metric          | KG On | KG Off | KG Smart |
|-----------------|-------|--------|----------|
| Success rate    | 69%   | 100%   | **92%**  |
| Config quality  | 0.34  | 0.68   | **0.72** |
| Avg iterations  | 5.7   | 3.0    | **4.1**  |
| Avg time (s)    | 30.0  | 13.5   | **16.0** |
| Medium success  | 33%   | 100%   | **100%** |
| Hard success    | 50%   | 100%   | **75%**  |
| Novel success   | 100%  | 100%*  | **100%** |

*KG Off succeeds on novel tasks but fabricates wrong material properties (avg quality 0.42).

Key finding: **Integration pattern, not KG content, determines utility.**
KG Smart recovers 23pp over mandatory KG by eliminating attention dilution.

### Novel Material Experiment (expanded 2026-04-15)

**Purpose**: Prove KG value for genuinely novel/proprietary data that LLMs
have never seen during training. Three fictional materials cover different
thermal regimes.

**Materials** (seeded ONLY into Neo4j):
- **Novidium**: k=73 W/(m·K), ρ=5420 kg/m³, cp=612 — moderate conductor
- **Cryonite**: k=0.42, ρ=1180, cp=1940 — extreme insulator (α≈1.8e-7)
- **Pyrathane**: k=312, ρ=3850, cp=278 — refractory super-conductor (α≈2.9e-4)

**Benchmark tasks** (7 total in `evaluation/ablation/benchmark_tasks.py`):
- G1–G3: Novidium (steady, transient, mixed BCs)
- C1–C2: Cryonite (steady, Robin BC)
- P1–P2: Pyrathane (steady, transient)

**New metrics**:
- **Material Property Fidelity (MPF)**: 1 - mean(|p_agent - p_truth| / p_truth)
- **Physics Score**: 0.5×MPF + 0.5×T_score (T_max/T_min range checks)

**Results** — KG Smart achieves 2.9× higher MPF than KG Off:
| Metric          | KG On | KG Off | KG Smart |
|-----------------|-------|--------|----------|
| Success rate    | 57%   | 100%   | **100%** |
| Config quality  | 0.55  | 0.61   | **0.96** |
| Property fidelity (MPF) | 0.57 | 0.34 | **1.00** |
| Physics score   | 0.71  | 0.63   | **1.00** |

**Critical finding**: KG Off fabricates wrong properties for ALL materials:
- Pyrathane: uses k=0.15 vs truth k=312 (**2,080× error**)
  - P2 result: T_max=2,950K from initial 2,000K (physically impossible)
- KG Smart retrieves exact properties via warm-start HNSW vector search

**Files added/modified**:
- `knowledge_graph/seeder.py`: 3 fictional materials in MATERIALS list
- `knowledge_graph/references.py`: References for Novidium, Cryonite, Pyrathane
- `evaluation/ablation/benchmark_tasks.py`: 7 novel tasks (G1–G3, C1–C2, P1–P2)
- `evaluation/ablation/run_ablation.py`: MPF/physics_score metrics, physics-aware scoring
- `tools/knowledge_tools.py`: Made `question` param optional
- `paper/main.tex`: Expanded §6.4, new tables (novel-props, novidium), MPF figure
