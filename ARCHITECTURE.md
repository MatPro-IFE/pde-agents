# PDE Agents — Architecture & Technology Stack

> A complete guide to how the system is built, how each technology works,
> and how all the layers connect into a working agentic PDE simulation platform
> with document intelligence and GraphRAG.

---

## Table of Contents

1. [Docker & Docker Compose — Infrastructure Glue](#1-docker--docker-compose)
2. [FEniCSx (DOLFINx) — Physics Engine](#2-fenicsxdolfinx--physics-engine)
3. [Gmsh — Complex Geometry Meshing](#2b-gmsh--complex-geometry-meshing)
4. [Ollama — Local LLM Server](#3-ollama--local-llm-server)
5. [LangChain & LangGraph — Agent Framework](#4-langchain--langgraph--agent-framework)
6. [The Three Agents — Specialised Workers](#5-the-three-agents--specialised-workers)
7. [The Orchestrator — Supervisor](#6-the-orchestrator--supervisor)
8. [FastAPI — REST Interface](#7-fastapi--rest-interface)
9. [PostgreSQL + SQLAlchemy — Metadata Database](#8-postgresql--sqlalchemy--metadata-database)
10. [MinIO — Object Storage](#9-minio--object-storage)
11. [Redis & Celery — Async Task Queue](#10-redis--celery--async-task-queue)
12. [Docling — Document Intelligence Pipeline](#11-docling--document-intelligence-pipeline)
13. [Plotly Dash — Visualization Dashboard](#12-plotly-dash--visualization-dashboard)
14. [Neo4j — Simulation Knowledge Graph](#13-neo4j--simulation-knowledge-graph)
15. [How Everything Works Together — End to End](#14-how-everything-works-together--end-to-end)
16. [The Four-Layer Architecture](#15-the-four-layer-architecture)

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
  ┌──────────────┐  ┌────────────────────────────────┐  ┌──────────────┐
  │fenics-runner │  │  agents (FastAPI + Celery worker)│  │  dashboard  │
  └──────────────┘  └────────────────────────────────┘  └──────────────┘
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
| `/agents/` | `agents:8000` | FastAPI REST + Swagger UI |
| `/browser/` | `neo4j:7474` | Neo4j Browser UI |
| `/neo4j-bolt` | `neo4j:7687` | Neo4j Bolt WebSocket |

MinIO console (`port 9002`) and NeoDash (`port 9001`) are accessed **directly** via their host ports.

### Service port map

| Container | Internal port | Host port | What it exposes |
|-----------|-------------|-----------|-----------------|
| `pde-nginx` | 80 | **8050** | Reverse proxy — all web UIs on one port |
| `pde-ollama` | 11434 | 11434 | LLM inference API |
| `pde-fenics` | 8080 | 8080 | FEM simulation REST API |
| `pde-agents` | 8000 | 8000 | Orchestrator + agent REST API (also via `/agents/`) |
| `pde-dashboard` | 8050 | — | Plotly Dash (internal only, accessed via nginx `/`) |
| `pde-minio` | 9000/9001 | 9000/9002 | S3 API / MinIO console (SSH tunnel to 9002) |
| `pde-postgres` | 5432 | 5432 | PostgreSQL |
| `pde-redis` | 6379 | 6379 | Redis (Celery broker) |
| `pde-neo4j` | 7687 / 7474 | 7687 / 7474 | Bolt / Browser |
| `pde-neodash` | 5005 | **9001** | NeoDash graph explorer |

### Dependency chain

```
postgres ──▶ agents (waits for DB to be healthy)
redis    ──▶ agents (waits for Redis to be healthy)
ollama   ──▶ agents (waits for Ollama to be healthy)
neo4j    ──▶ agents (waits for Neo4j to be healthy — ~40 s on first boot)
agents   ──▶ dashboard
dashboard, agents, fenics-runner, neo4j ──▶ nginx (soft dep)
```

---

## 2. FEniCSx/DOLFINx — Physics Engine

### What it is

FEniCSx is the leading open-source Finite Element Method (FEM) framework. It solves
PDEs on arbitrary domains by converting them from a strong differential form into a
**weak (variational) form** and assembling a large sparse linear system solved numerically.

Version: **DOLFINx 0.10.0.post2**, running inside the `fenics-runner` container.

### The heat equation being solved

**Strong form:**
```
ρ c_p ∂u/∂t − ∇·(k ∇u) = f    in Ω × (0, T]
```

**Boundary conditions:**

| Type | Equation | Physical meaning |
|------|----------|-----------------|
| Dirichlet | `u = g` on Γ_D | Fixed temperature (e.g. wall at 300 K) |
| Neumann | `k ∂u/∂n = h` on Γ_N | Prescribed heat flux (h=0 → insulated) |
| Robin | `k ∂u/∂n = α(u_∞ − u)` on Γ_R | Convective cooling |

### Step-by-step: how FEniCSx solves it

**Step 1** — Mesh the domain (built-in or Gmsh)

**Step 2** — Define P1 Lagrange function space: `u(x,y,t) ≈ Σᵢ uᵢ(t) · φᵢ(x,y)`

**Step 3** — Derive the weak form (θ-scheme):
```
(ρcₚ/dt)(u^{n+1} − u^n, v)_Ω
  + θ   [ k(∇u^{n+1}, ∇v)_Ω + α(u^{n+1}, v)_Γ_R ]
  + (1−θ)[ k(∇u^n, ∇v)_Ω    + α(u^n, v)_Γ_R     ]
= ∫_Ω f v dx + ∫_Γ_N h v ds + α(u_∞, v)_Γ_R
```

**Step 4** — Assemble: `A · u^{n+1} = b(u^n)`

**Step 5** — Apply boundary conditions (modify rows of A and b)

**Step 6** — Solve with PETSc CG + HYPRE AMG preconditioner

**Step 7** — Advance in time, save snapshots every N steps

**Step 8** — Save: `u_final.npy`, `dof_coords.npy`, `snapshots/`, `temperature.xdmf/.h5`, `result.json`, `config.json`

---

## 2b. Gmsh — Complex Geometry Meshing

### What it is

[Gmsh](https://gmsh.info/) is an open-source 3D finite element mesh generator.
In PDE Agents it defines non-rectangular simulation domains via Python API.

### Available geometry types

| Type | Key parameters | Physical groups (boundaries) |
|------|----------------|------------------------------|
| `rectangle` | `Lx`, `Ly`, `mesh_size` | `left`, `right`, `top`, `bottom` |
| `l_shape` | `Lx`, `Ly`, `mesh_size` | `left`, `right`, `top`, `bottom`, `inner_h`, `inner_v` |
| `circle` | `radius`, `mesh_size` | `boundary` |
| `annulus` | `r_inner`, `r_outer`, `mesh_size` | `inner_wall`, `outer_wall` |
| `hollow_rectangle` | `Lx`, `Ly`, `hole_*`, `mesh_size` | `outer_*`, `inner_*` |
| `t_shape` | `Lx`, `Ly`, `stem_*`, `mesh_size` | `left`, `right`, `top`, `bottom`, `stem_left`, `stem_right` |
| `stepped_notch` | `Lx`, `Ly`, `notch_*`, `mesh_size` | `left`, `right`, `top`, `bottom`, `notch_*` |
| `box` (3D) | `Lx`, `Ly`, `Lz`, `mesh_size` | `left`, `right`, `top`, `bottom`, `front`, `back` |
| `cylinder` (3D) | `radius`, `height`, `mesh_size` | `bottom_face`, `top_face`, `lateral` |

`heat_equation.py` dispatches on the `geometry.type` key:

```python
if config.geometry:
    result = build_gmsh_mesh(config.geometry)   # Gmsh path
else:
    mesh = create_rectangle_mesh(...)            # built-in path
```

---

## 3. Ollama — Local LLM Server

### What it is

Ollama is an open-source tool that downloads, manages, and serves LLMs locally
on GPU. It exposes a REST API identical in design to OpenAI's API.

### The three models and why each was chosen

| Model | Parameters | VRAM | Role | Why |
|-------|-----------|------|------|-----|
| `qwen2.5-coder:32b` | 32B | ~19 GB | Simulation Agent | Best at structured JSON, code generation, config validation |
| `qwen2.5-coder:14b` | 14B | ~9 GB | Database Agent | Faster responses, good at SQL-like structured output |
| `llama3.3:70b` | 70B | ~42 GB | Analytics + Orchestrator | Best multi-step reasoning, native structured tool calling |
| `nomic-embed-text` | 137M | <1 GB | Embeddings | 768-dim vectors for runs AND document chunks |

All three fit simultaneously in ~196 GB combined VRAM.

---

## 4. LangChain & LangGraph — Agent Framework

### LangGraph: the state machine

LangGraph turns an agent into a **directed graph** of nodes and edges.

**The ReAct graph:**
```
START ──▶ reason_node ──▶ (has tool calls?) ──▶ YES ──▶ act_node ──┐
              │                                                       │
              └──▶ NO ──▶ finish_node ──▶ END             (loops back)
```

**reason_node** — calls the LLM with full message history  
**act_node** — executes the tool, appends `ToolMessage` to state  
**Router** — loops back if more tool calls needed, finishes otherwise  

---

## 5. The Three Agents — Specialised Workers

### Agent-1: Simulation Agent (`qwen2.5-coder:32b`)
**Tools:** `check_config_warnings`, `query_knowledge_graph`, `validate_config`, `run_simulation`, `modify_config`, `debug_simulation`, `list_recent_runs`, `get_run_status`, `run_parametric_sweep`

The Simulation Agent supports three **KG integration modes**, set at construction time:

| Mode | Constructor flag | Behaviour |
|------|-----------------|-----------|
| **KG On** (default) | — | Mandatory `check_config_warnings` + `query_knowledge_graph` calls before every run. System prompt requires KG-first workflow. |
| **KG Off** | `disable_kg=True` | Both KG tools removed from tool list. System prompt directs agent to skip straight to `validate_config` → `run_simulation`. Used for ablation baseline. |
| **KG Smart** | `smart_kg=True` | KG tools available but lazy. Before the agent loop, task description is embedded via `nomic-embed-text`, HNSW index is queried for top-3 similar successful past runs, and their configs are injected into the system prompt as few-shot examples. Agent only calls KG tools after a failure or for unknown materials. |

KG Smart is inspired by **Corrective RAG** (adaptive retrieval) and **AriGraph** (episodic KG memory). 3-way ablation results across 10 benchmark tasks: KG On 60%, KG Off 100%, **KG Smart 90%** overall success.

### Agent-2: Analytics Agent (`llama3.3:70b`)
**Tools:** `analyze_run`, `compare_runs`, `compare_study`, `get_steady_state_time`, `suggest_next_run`, `export_summary_report`

### Agent-3: Database Agent (`qwen2.5-coder:14b`)
**Tools:** `store_result`, `query_runs`, `catalog_study`, `fetch_run_data`, `export_to_csv`, `upload_to_minio`, `db_health_check`

---

## 6. The Orchestrator — Supervisor

**File:** `orchestrator/graph.py`

Uses `llama3.3:70b` to decompose tasks, route between agents, and synthesise outputs.
Routing decisions:
```json
{"next": "simulation", "reason": "Need to run the FEM first"}
{"next": "analytics",  "reason": "Have run_ids, now need analysis"}
{"next": "database",   "reason": "Analysis done, store everything"}
{"next": "FINISH",     "reason": "All tasks complete"}
```

---

## 7. FastAPI — REST Interface

**File:** `orchestrator/api.py`

### All endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/run` | Synchronous multi-agent task |
| `POST` | `/run/async` | Submit task, returns job_id |
| `POST` | `/agent/{name}/async` | Submit to one agent |
| `GET` | `/jobs/{job_id}` | Poll job status and result |
| `GET` | `/runs` | List simulation runs |
| `GET` | `/runs/{run_id}` | Full details for one run |
| `GET` | `/health` | Service health |
| `WS` | `/ws/stream` | WebSocket for streaming agent output |
| `POST` | `/simulate` | Direct Simulation Agent invoke — query params: `disable_kg=true` (KG Off mode) or `smart_kg=true` (KG Smart mode) |
| `GET` | `/kg/stats` | Knowledge graph node counts |
| `POST` | `/kg/seed` | Re-seed static knowledge |
| `GET` | `/kg/material/{name}` | Material lookup |
| `POST` | `/kg/search` | Semantic similarity search by description |
| `POST` | `/kg/check` | Pre-run warnings + similar runs + physics refs |
| `GET` | `/kg/run/{run_id}/similar` | Semantic similarity search for a run |
| `GET` | `/kg/run/{run_id}/references` | All references linked to a run |
| `POST` | `/references/upload` | Upload PDF/TXT/Markdown to KG |
| `POST` | `/references/fetch-url` | Queue web resource crawl + index |
| `GET` | `/references/uploaded` | List all uploaded references |
| `GET` | `/references/{ref_id}/chunks` | Structured chunks for a document |
| `GET` | `/references/{ref_id}/status` | Async processing status |
| `GET` | `/references/search-chunks` | Semantic search over all chunks |
| `POST` | `/references/{ref_id}/link/{run_id}` | Manually pin doc to run |

### The async job pattern

LLM inference + FEM solving can take minutes. The solution is **submit-and-poll**:

```
Client                          FastAPI
  │  POST /run/async {"task":"..."}│
  │  ◀── {job_id: "abc123"}  ──────┤  ← returns in <1s
  │  GET /jobs/abc123              │  poll every 3s
  │  ◀── {status: "running", ...}  │
  │  GET /jobs/abc123              │
  │  ◀── {status: "success", ...}  │  result ready
```

---

## 8. PostgreSQL + SQLAlchemy — Metadata Database

PostgreSQL stores structured simulation metadata — small scalars you want to query, filter, and aggregate.

### Database schema

```
simulation_runs          ← one row per simulation run
  id, run_id, status, dim, nx, ny, nz, k, rho, cp
  T_max, T_min, T_mean, n_dofs, wall_time
  config_json, output_dir, created_at

run_parameters           ← queryable key-value pairs from config
convergence_records      ← per-time-step convergence data
parametric_studies       ← groups of related runs
agent_messages           ← full conversation history
agent_suggestions        ← structured suggestions from Analytics Agent
```

---

## 9. MinIO — Object Storage

MinIO stores **large binary files** that don't belong in SQL:
- `temperature.xdmf` / `.h5` — full time-series data
- `u_final.npy`, `dof_coords.npy`, `snapshots/*.npy` — field arrays
- `reference-uploads/{ref_id}/{filename}` — uploaded PDF/TXT/Markdown documents

**Buckets:**
- `simulation-results` — FEM output files
- `reference-uploads` — user-uploaded knowledge documents

---

## 10. Redis & Celery — Async Task Queue

### What changed from placeholder to production use

Redis was originally a placeholder for future scaling. It is now **actively used** as the
Celery message broker for the document intelligence pipeline.

### Architecture

```
FastAPI (upload endpoint)
  │
  │  task = ingest_document_task.apply_async(kwargs={...}, queue='document_ingestion')
  │
  ▼
Redis (queue: document_ingestion)
  │
  │  Celery worker (co-process inside pde-agents container)
  ▼
  1. Parse PDF/HTML with Docling
  2. Embed each chunk with nomic-embed-text
  3. Store ReferenceChunk nodes in Neo4j
  4. Build CROSS_REFS edges to similar Run nodes
  5. Update Reference.process_status = 'completed'
```

### Why Celery + Redis for document ingestion?

Document processing is expensive:
- Docling may take 30–120s for a large PDF
- Each chunk requires an Ollama embedding call (768-dim)
- A 50-page web crawl takes minutes (1s/page polite delay + Docling + embeddings)

Queuing this work prevents the upload API from timing out and keeps the UI responsive.
The user gets an immediate `{status: "queued", task_id: "..."}` response and can
poll `/references/{ref_id}/status` for progress.

### Celery tasks

| Task name | File | What it does |
|-----------|------|-------------|
| `ingest_document` | `knowledge_graph/tasks.py` | Parse uploaded PDF/TXT with Docling, embed chunks, cross-ref to runs |
| `ingest_web_resource` | `knowledge_graph/tasks.py` | BFS crawl a site, parse each page with Docling HTML pipeline, embed chunks, cross-ref to runs |

### Celery worker startup

The worker runs as a detached co-process inside the `pde-agents` container,
started alongside Uvicorn via the `docker-compose.yml` command:

```yaml
command: >
  bash -c "celery -A knowledge_graph.tasks:celery_app worker
           --loglevel=info --concurrency=2
           -Q document_ingestion,celery
           --detach --pidfile=/tmp/celery.pid --logfile=/tmp/celery.log
           && exec uvicorn orchestrator.api:app --host 0.0.0.0 --port 8000 --reload"
```

---

## 11. Docling — Document Intelligence Pipeline

### What it is

[IBM Docling](https://github.com/DS4SD/docling) is an open-source (MIT) document
understanding library that converts PDFs, DOCX, HTML, and other formats into a
structured document model with headings, text, tables, equations, and code blocks.

### Why Docling?

| Feature | PyPDF (simple) | Docling |
|---------|---------------|---------|
| Hierarchical structure (headings/sections) | ❌ | ✅ |
| Table extraction | ❌ | ✅ |
| Equation detection | ❌ | ✅ |
| Code block detection | ❌ | ✅ |
| HTML support | ❌ | ✅ |
| Context-aware chunking | ❌ | ✅ (HybridChunker) |
| License | MIT | MIT |

PyPDF is kept as a fallback for when Docling's pipeline fails (e.g. scanned PDFs without OCR).

### Processing pipeline

```
Document (PDF / HTML / TXT / DOCX)
  │
  ├── 1. Docling DocumentConverter
  │       → structured document model (headings, paragraphs, tables, equations)
  │
  ├── 2. HybridChunker (max 256 tokens, overlap 32)
  │       → list of DocumentChunk objects with heading context
  │
  ├── 3. _classify_chunk(text)
  │       → classification: material | bc | solver | domain | general
  │       (keyword matching against physics term lists)
  │
  ├── 4. OllamaEmbedder.embed_text(chunk.text)
  │       → 768-dim vector via nomic-embed-text
  │
  └── 5. kg.ingest_document_chunks(ref_id, chunks)
          ├── MERGE (:ReferenceChunk) for each chunk with embedding
          ├── vector search: find top-k similar :Run nodes (score ≥ 0.78)
          │     → CROSS_REFS {score} edges
          └── entity linking based on classification
                → RELATES_TO :Material / :BCConfig / :Domain
```

### Web crawling (`web_fetcher.py`)

```
root_url (e.g. https://jsdokken.com/dolfinx-tutorial/)
  │
  ├── 1. discover_pages(root_url, max_pages=50)
  │       BFS crawl, following links within the same domain/path prefix
  │       1s polite delay between requests
  │       Filter: same netloc + path starts with root_path
  │
  ├── 2. For each page URL:
  │       fetch_page(url) → FetchedPage(url, title, html)
  │
  ├── 3. parse_html_with_docling(html)
  │       write to temp .html file → DocumentConverter → HybridChunker
  │       → list[DocumentChunk]
  │
  └── 4. Celery task: embed + ingest each page as a child :Reference node
          (:Reference {is_web: true})  ← parent
            └─[:HAS_PAGE]→ (:Reference {url: page_url}) ← one per page
                └─[:HAS_CHUNK]→ (:ReferenceChunk)
```

**Result for FEniCSx tutorial (12 pages):**
- 298 chunks stored and embedded
- 1,073 cross-references to simulation runs
- Heat equation page → semantic search "backward Euler time stepping" → score 0.871

---

## 12. Plotly Dash — Visualization Dashboard

### What Dash is

Dash combines Flask (web server), Plotly.js (charts), and React (UI). You write
everything in Python — Dash generates the React components and JavaScript callbacks.

### Dashboard tabs

| Tab | Purpose |
|-----|---------|
| 📊 Overview | Recent runs, solver scaling scatter, peak T by conductivity, system health, run inspector with SIMILAR_TO neighbours |
| 🌡️ Field Viewer | Heatmap, 3D surface, heat flux, profiles, Z-slice, volume render, time animation |
| 📈 Convergence | L2-norm history comparison across runs |
| 🔬 Parametric | Scatter/bar comparison across swept parameters |
| 🤖 Agent Chat | Async chat with agents + quick-prompt buttons; Enter key submits |
| 🧠 Knowledge Graph | KG stats, semantic run search, physics reference browser, Add to KG panel, Semantic Chunk Search |
| 🔎 Run Explorer | Full run browser with agent timeline, config, files, and recommendations |

### Knowledge Graph tab — detailed layout

```
🧠 Knowledge Graph Tab
├── Left panel
│   ├── Graph Statistics (8 stat cards: runs, embedded runs, SIMILAR_TO edges,
│   │   references, ref_chunks, materials, bc_configs, domains, thermal_classes)
│   ├── Semantic Run Search
│   │   └── TextArea → nomic-embed-text → HNSW queryNodes() → run similarity cards
│   └── NeoDash launcher (opens port 9001)
│
└── Right panel
    ├── ➕ Add to Knowledge Graph  (tabbed card)
    │   ├── 📎 Upload File tab
    │   │   ├── DOI quick-fill (press Enter to fetch from CrossRef API)
    │   │   ├── File drag-and-drop (PDF / TXT / Markdown)
    │   │   ├── Title, Citation, URL, Subject, Type, Run IDs, Auto-link top-k
    │   │   └── Upload & Link button
    │   └── 🌐 Web Resource URL tab
    │       ├── URL input (press Enter or click Fetch)
    │       ├── Title, Subject, Max pages
    │       └── Fetch & Index button (queues Celery task)
    │
    ├── 📚 Uploaded Documents list
    │   └── Shows process_status, n_chunks, cross_refs per document
    │
    ├── Physics Reference Browser
    │   └── Type filter + cards with source links
    │
    └── 🔬 Semantic Chunk Search (full width)
        ├── Input (press Enter or click Search Chunks)
        └── Result cards showing:
            ├── Clickable title → source URL (web page / paper)
            ├── Chunk text (first 400 chars)
            ├── Classification badge (material / bc / solver / domain / general)
            ├── Chunk type badge (text / code / equation / table)
            ├── Similarity score
            └── "🌐 Open page" or "📄 View source" button
```

### Enter-key support

All search and fetch inputs support `n_submit` (Dash's Enter key trigger):
- DOI input → fetches CrossRef metadata
- Web resource URL input → queues crawl
- Semantic Chunk Search input → runs vector search

### Navigation links — dynamic hostname resolution

The dashboard navbar builds links at runtime in the browser (clientside callback):
```javascript
function(href) {
    var host = window.location.hostname;
    return [
        '/agents/docs',                     // API Docs — via nginx
        'http://' + host + ':9002',         // MinIO console — direct
        'http://' + host + ':7474/...',     // Neo4j Browser — direct
        'http://' + host + ':9001',         // NeoDash — direct (port 9001)
    ];
}
```

### Field Viewer visualisation modes

| Mode | Method | Description |
|------|--------|-------------|
| 🌡 Heatmap | `go.Heatmap` + `go.Contour` | Colour map + 12 labelled isothermal lines |
| 🏔 3D Surface | `go.Surface` | Temperature raised as height, fully rotatable |
| ∇ Heat Flux | gradient of interpolated field | `|−k∇T|` magnitude + arrows |
| 〰 Profiles | `go.Scatter` slices | T(x) at 6 fixed Y values |
| 🍕 Z-Slice | `griddata` + `go.Heatmap` | XY plane cut at adjustable Z (3D runs) |
| 📦 Volume | 3× `go.Surface` + `go.Isosurface` | Orthogonal slice planes |
| 🎬 Animation | `dcc.Interval` advancing snapshots | Playback of transient history |

---

## 13. Neo4j — Simulation Knowledge Graph (GraphRAG)

### Why a graph database?

| Concern | Relational (PostgreSQL) | Graph (Neo4j) |
|---------|------------------------|---------------|
| "Which runs used similar parameters?" | Complex multi-join SQL | Single MATCH |
| "Find semantically similar runs" | Not possible | HNSW vector index |
| "What document sections relate to this run?" | Not possible | CROSS_REFS traversal |
| "What physics fact is relevant here?" | Not possible | HAS_REFERENCE hop |

### Complete graph schema

```
(:Run {
    run_id, name, dim, status, k, rho, cp,
    nx, ny, nz, Lx, Ly, Lz, bc_types,
    t_end, dt, theta, source, u_init,
    t_max, t_min, t_mean, l2_norm, wall_time, n_dofs,
    created_at,
    embedding   ← 768-dim nomic-embed-text vector
})

(:Material { name, k, rho, cp, alpha, k_min, k_max, description, typical_uses })
(:KnownIssue { code, severity, condition, description, recommendation })
(:BCConfig { pattern, description, has_dirichlet, has_neumann, has_robin, has_source })
(:Domain { label, description, Lx_ref, Ly_ref, char_len })
  Labels: micro | component | panel | structural
(:ThermalClass { name, description, k_threshold })
  Labels: high_conductor | medium_conductor | low_conductor | thermal_insulator

(:Reference {
    ref_id, title, type, subject, source, url, tags,
    is_uploaded, is_web, parent_ref, process_status,
    n_pages, n_chunks, n_tables, parse_method,
    chunks_stored, chunks_embedded, cross_refs
})
  Static types: material_property | bc_practice | solver_guidance | domain_physics
  Uploaded types: paper | report | handbook | standard
  Web types: web_resource | web_page

(:ReferenceChunk {
    chunk_id, ref_id, chunk_index,
    heading, text,
    chunk_type,       ← text | code | equation | table | list
    classification,   ← material | bc | solver | domain | general
    confidence, page,
    embedding         ← 768-dim nomic-embed-text vector
})

Relationships:
  (:Run)-[:USES_MATERIAL {confidence}]->(:Material)
  (:Run)-[:TRIGGERED {detected_at}]->(:KnownIssue)
  (:Run)-[:USES_BC_CONFIG]->(:BCConfig)
  (:Run)-[:ON_DOMAIN]->(:Domain)
  (:Material)-[:HAS_THERMAL_CLASS]->(:ThermalClass)
  (:Run)-[:SIMILAR_TO {score, updated_at}]->(:Run)    ← KNN semantic edges
  (:Run)-[:SPAWNED_FROM]->(:Run)
  (:Run)-[:CITES]->(:Reference)                        ← pinned/auto-linked
  (:Material)-[:HAS_REFERENCE]->(:Reference)
  (:BCConfig)-[:HAS_REFERENCE]->(:Reference)
  (:Domain)-[:HAS_REFERENCE]->(:Reference)
  (:Reference)-[:HAS_CHUNK]->(:ReferenceChunk)         ← document sections
  (:Reference)-[:HAS_PAGE]->(:Reference)               ← web resource hierarchy
  (:ReferenceChunk)-[:CROSS_REFS {score}]->(:Run)      ← chunk ↔ similar run
  (:ReferenceChunk)-[:RELATES_TO]->(:Material)         ← entity linking
  (:ReferenceChunk)-[:RELATES_TO]->(:BCConfig)
  (:ReferenceChunk)-[:RELATES_TO]->(:Domain)
```

### Neo4j indexes

```cypher
-- Uniqueness constraints (one per node type)
CREATE CONSTRAINT run_id_unique ...
-- (+ Material, BCConfig, Domain, ThermalClass, KnownIssue, Reference, ReferenceChunk)

-- HNSW vector index for Run similarity
CREATE VECTOR INDEX run_embedding_index IF NOT EXISTS
FOR (r:Run) ON r.embedding
OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}

-- HNSW vector index for chunk search (document intelligence)
CREATE VECTOR INDEX chunk_embedding_index IF NOT EXISTS
FOR (c:ReferenceChunk) ON c.embedding
OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}
```

### GraphRAG features

#### Feature 1: Vector embeddings (Run nodes)

Every run generates a physics summary via `run_to_text()`:
```
"2D l_shape geometry, panel-scale, k=50.0 W/(m·K) [medium_conductor],
 rho=7800 kg/m³, cp=500 J/(kg·K), thermal_diffusivity=1.282e-05 m²/s,
 BCs: dirichlet+robin, t_end=0.2s dt=0.02s,
 T_max=800.0K T_min=753.1K, DOFs=116, wall_time=0.69s, status=success."
```

This is embedded via `nomic-embed-text` and stored as `r.embedding`.

#### Feature 2: SIMILAR_TO KNN edges

After every `add_run()`, up to 5 `SIMILAR_TO` edges are built (cosine ≥ 0.85):
```cypher
CALL db.index.vector.queryNodes('run_embedding_index', 6, $vec)
YIELD node AS neighbour, score
WHERE neighbour.run_id <> $run_id AND score >= 0.85
MERGE (src)-[rel:SIMILAR_TO]->(neighbour)
SET rel.score = round(score, 4)
```

#### Feature 3: Curated physics references

20 seeded `Reference` nodes linked to Material/BCConfig/Domain. Retrieved at run-time via:
```cypher
MATCH (ref:Reference)
WHERE EXISTS { MATCH (m:Material)-[:HAS_REFERENCE]->(ref) WHERE m.k_min <= $k <= m.k_max }
   OR EXISTS { MATCH (b:BCConfig {pattern: $bc_pattern})-[:HAS_REFERENCE]->(ref) }
   OR EXISTS { MATCH (d:Domain {label: $domain_label})-[:HAS_REFERENCE]->(ref) }
RETURN ref.*
```

#### Feature 4: Document chunk vector search

```python
vec = get_embedder().embed_text("backward Euler stability heat equation")
kg.search_chunks_by_query(vec, top_k=5)
# → [{chunk_id, heading, text, classification, ref_title, ref_url, score}, ...]
```

The result carries `ref_url` for the exact source page, enabling clickable links in the dashboard.

#### Feature 5: Cross-reference linking (CROSS_REFS)

Every `ReferenceChunk` is automatically linked to the most similar simulation runs:
```cypher
CALL db.index.vector.queryNodes('run_embedding_index', $k, $chunk_vec)
YIELD node AS run, score
WHERE score >= 0.78
MERGE (chunk)-[xr:CROSS_REFS]->(run)
SET xr.score = round(score, 4)
```

This means: when a user reads a tutorial section about "Robin boundary conditions for cooling", the chunk is linked to every simulation run that used Robin BCs with similar parameters.

### Rule-based warning engine

`knowledge_graph/rules.py` implements 9 pure-Python rules that run instantly before every simulation:

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

### KG REST endpoints

```bash
GET  /kg/stats                     # node counts incl. embedded_runs, ref_chunks
POST /kg/seed                      # re-seed static knowledge + references
GET  /kg/run/{run_id}/similar      # semantic similarity search for this run
GET  /kg/run/{run_id}/references   # all references (curated + uploaded) for a run
POST /references/upload            # upload PDF/TXT → queue Celery ingestion
POST /references/fetch-url         # queue web crawl + ingestion
GET  /references/uploaded          # list all uploaded references with status
GET  /references/{ref_id}/chunks   # structured chunks for a document
GET  /references/{ref_id}/status   # async processing status
GET  /references/search-chunks     # semantic search over all chunks
```

### Phase roadmap

| Phase | Status | Features |
|-------|--------|---------|
| **Phase 1** | ✅ Done | Neo4j container, Run/Material/KnownIssue nodes, `add_run`, similarity search, rule engine, agent tools, REST endpoints |
| **Phase 2** | ✅ Done | BCConfig, Domain, ThermalClass nodes; vector embeddings; HNSW index; SIMILAR_TO KNN edges; Reference nodes; semantic search; dashboard KG tab; NeoDash integration |
| **Phase 3** | ✅ Done | Uploaded document ingestion (Docling + Celery); ReferenceChunk nodes with chunk-level HNSW index; CROSS_REFS edges; web crawler for online tutorials; DOI auto-fill; Semantic Chunk Search in dashboard with clickable source links |
| **Phase 4** | ✅ Done | **KG Smart mode:** warm-start injection of top-3 similar runs + lazy conditional KG tool use; 3-way ablation (KG On / Off / Smart); `evaluation/` framework for V&V + ablation + agent quality metrics |
| Phase 5 | Planned | Automated correlation miner, `IMPROVED_OVER` relationships, agent cites specific ReferenceChunk IDs for full reasoning auditability |

---

## 14. How Everything Works Together — End to End

### Simulation request flow

```
User text in dashboard chat
  │
  │  POST /run/async {"task": "Run 2D heat equation on steel..."}
  ▼
FastAPI (agents:8000)
  → spawn thread: orchestrator.run(task)
  │
  ├── Orchestrator (llama3.3:70b): plan workflow
  │
  ├── Simulation Agent (qwen2.5-coder:32b):
  │   1. check_config_warnings() → rules + KG warnings
  │   2. validate_config() → parameter check
  │   3. run_simulation() → POST fenics-runner:8080/run
  │      → FEniCSx: mesh → assemble → PETSc solve → save files
  │      → Auto: PostgreSQL + MinIO + kg.add_run() + embed + SIMILAR_TO
  │
  ├── Analytics Agent (llama3.3:70b):
  │   analyze_run(), compare_runs(), suggest_next_run()
  │
  └── Database Agent (qwen2.5-coder:14b):
      store_result(), catalog_study()
```

### Document ingestion flow

```
User pastes URL in 🌐 Web Resource URL tab
  │
  │  POST /references/fetch-url {"url": "...", "max_pages": 50}
  ▼
FastAPI
  1. Create (:Reference {is_web: true, process_status: "queued"}) in Neo4j
  2. Enqueue ingest_web_resource_task to Redis queue

Celery worker (pde-agents container):
  1. discover_pages() — BFS crawl with 1s polite delays
  2. For each page:
     a. fetch_page() → FetchedPage(html)
     b. parse_html_with_docling() → list[DocumentChunk]
     c. embed_chunks() → 768-dim vectors per chunk
     d. kg.ingest_document_chunks() → (:ReferenceChunk) nodes
                                    → CROSS_REFS to similar :Run nodes
                                    → RELATES_TO :Material/:BCConfig/:Domain
  3. Update (:Reference) → process_status: "completed", n_chunks: 298, cross_refs: 1073

User sees in dashboard:
  → "FEniCSx Tutorial — completed  |  298 chunks  |  1073 cross-refs"
  → Semantic Chunk Search "heat equation backward Euler" returns
    chunk from chapter2/heat_equation.html with [🌐 Open page] link
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
│   • Validates configs with check_config_warnings before running    │
│   • Cites similar past runs and physics references from KG         │
│   • Synthesises multi-step outputs into human-readable reports     │
│                                                                    │
│   Components: base_agent.py, simulation_agent.py,                  │
│               analytics_agent.py, database_agent.py,              │
│               orchestrator/graph.py, tools/knowledge_tools.py      │
└──────────────────────────┬─────────────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────────────┐
│                      KNOWLEDGE LAYER                               │
│                                                                    │
│   Neo4j Knowledge Graph + Docling Document Intelligence            │
│                                                                    │
│   • Stores every run as a graph node with embedding                │
│   • Stores every document chunk with embedding                     │
│   • Cross-references literature sections to simulation runs        │
│   • Answers: "What similar runs exist? What warnings apply?        │
│     What does the FEniCSx tutorial say about this BC type?"        │
│   • Grows automatically with every simulation and every upload     │
│                                                                    │
│   Components: knowledge_graph/graph.py, rules.py, seeder.py,       │
│               document_processor.py, tasks.py, web_fetcher.py      │
└──────────────────────────┬─────────────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────────────┐
│                      EXECUTION LAYER                               │
│                                                                    │
│   FEniCSx + PostgreSQL + MinIO + Redis/Celery                      │
│                                                                    │
│   • Solves physics (FEM weak form, PETSc linear algebra)           │
│   • Persists structured metadata + agent logs                      │
│   • Archives output files and reference documents to MinIO         │
│   • Queues async document ingestion via Celery + Redis             │
│                                                                    │
│   Components: heat_equation.py, fenics_runner_api.py,              │
│               database/models.py, knowledge_graph/tasks.py         │
└──────────────────────────┬─────────────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────────────┐
│                     PRESENTATION LAYER                             │
│                                                                    │
│   Plotly Dash + FastAPI                                            │
│                                                                    │
│   • Visualises temperature fields (heatmap/surface/animation)      │
│   • Run Explorer: browse runs, agent timeline, config, files       │
│   • Upload documents or fetch web resources to the KG             │
│   • Semantic Chunk Search with clickable source links              │
│   • Enter-key support on all search and fetch inputs               │
│                                                                    │
│   Components: visualization/dashboard.py, orchestrator/api.py      │
└────────────────────────────────────────────────────────────────────┘
```

### The key design principle: replaceable layers

Each layer communicates through well-defined contracts:

- Swap **FEniCSx for OpenFOAM** → agents work unchanged (same `/run` JSON shape)
- Swap **qwen2.5-coder for GPT-4** → execution and presentation layers unaffected
- Swap **Docling for another parser** → only `document_processor.py` changes
- Swap **Neo4j for Memgraph** → only `knowledge_graph/graph.py` changes (same Cypher)
- Add a **fourth agent** → add tools and a new node to the orchestrator graph
- Add a **new PDE** → seed failure patterns; the KG populates automatically

---

## Quick Reference

### Service URLs

| URL | Purpose |
|-----|---------|
| http://`host`:8050/ | Dashboard (all tabs — proxied via nginx) |
| http://`host`:8050/agents/docs | FastAPI Swagger UI |
| http://`host`:7474 | Neo4j Browser (direct) |
| http://`host`:9001 | NeoDash — graph explorer (reuses ex-MinIO console port) |
| http://`host`:9002 | MinIO console (SSH tunnel recommended) |
| http://`host`:8000 | Agents REST API (direct, bypasses nginx) |
| http://`host`:11434 | Ollama LLM API |

### Key files

| File | Purpose |
|------|---------|
| `docker-compose.yml` | All service definitions incl. NeoDash, Celery co-process |
| `simulations/solvers/heat_equation.py` | FEM solver (built-in + Gmsh meshes) |
| `simulations/geometry/gmsh_geometries.py` | 9 Gmsh geometry builders |
| `docker/fenics_runner_api.py` | FEniCSx HTTP API |
| `agents/base_agent.py` | ReAct loop + tool-call parser |
| `orchestrator/api.py` | FastAPI + async jobs + /kg/ + /references/ endpoints |
| `visualization/dashboard.py` | Dash: all tabs + Enter-key + chunk search with source links |
| `knowledge_graph/graph.py` | Full KG schema, both HNSW indexes, all methods |
| `knowledge_graph/embeddings.py` | OllamaEmbedder + run_to_text() |
| `knowledge_graph/document_processor.py` | Docling pipeline: parse → chunk → classify |
| `knowledge_graph/tasks.py` | Celery tasks: ingest_document + ingest_web_resource |
| `knowledge_graph/web_fetcher.py` | BFS web crawler + Docling HTML parsing |
| `knowledge_graph/references.py` | 20 curated physics reference entries |
| `knowledge_graph/rules.py` | 9-rule pre-run warning engine |
| `knowledge_graph/seeder.py` | 10 materials + 6 failure patterns |
| `scripts/sweep_full_study.py` | 805-run comprehensive parameter sweep |

### Data flow summary

```
User text
  → LLM (Ollama)                   : intent + tool selection
  → check_config_warnings()        : rule engine + KG semantic search + physics refs
  → Neo4j                          : vector queryNodes() → similar runs + references
  → FEniCSx + Gmsh                 : PDE solving (if simulation tool)
  → PostgreSQL                     : metadata + agent_run_logs
  → MinIO                          : binary files (FEM output + reference documents)
  → Neo4j kg.add_run()             : Run node + BCConfig + Domain + material link
  → Ollama nomic-embed-text        : 768-dim vector on Run.embedding
  → Neo4j SIMILAR_TO edges         : top-5 KNN precomputed

User uploads document / pastes URL
  → FastAPI                        : create :Reference node, return task_id
  → Redis queue                    : Celery picks up task
  → Docling                        : structured parse → chunks with headings/types
  → classify_chunk()               : material | bc | solver | domain | general
  → Ollama nomic-embed-text        : 768-dim vector per chunk
  → Neo4j :ReferenceChunk nodes    : stored with embedding
  → Neo4j CROSS_REFS edges         : chunk ↔ top-k similar :Run nodes (score ≥ 0.78)
  → Neo4j RELATES_TO edges         : chunk ↔ :Material / :BCConfig / :Domain
  → Dashboard Semantic Chunk Search: user queries, gets results with clickable source links
```

---

---

## 16. Evaluation Framework (`evaluation/`)

The `evaluation/` directory contains the research paper's experiment scripts. All experiments run against the live Docker stack.

### Verification & Validation (`benchmarks/`)

```
benchmarks/analytical_solutions.py   ← 3 closed-form heat equation solutions
benchmarks/vv_runner.py               ← V&V convergence study (5 mesh resolutions)
```

Runs FEM at mesh resolutions h ∈ {1/8, 1/16, 1/32, 1/64, 1/128} and computes L2 error vs analytical solution. Uses UFL `SpatialCoordinate` expressions and high-order Gaussian quadrature for accurate error integration.

### KG Ablation (`ablation/`)

```
ablation/benchmark_tasks.py    ← 10 natural-language simulation tasks
ablation/run_ablation.py       ← 3-way runner: KG On / KG Off / KG Smart
```

CLI flags:
- `python run_ablation.py` — 2-way: KG On vs KG Off
- `python run_ablation.py --include-smart` — 3-way: adds KG Smart
- `python run_ablation.py --smart-only` — KG Smart alone

### Agent Quality Metrics (`metrics/`)

```
metrics/agent_quality.py   ← mines PostgreSQL for tool-call patterns, success rates
```

Computes: first-attempt success rate, KG tool invocation frequency, avg iterations, avg wall time per task.

### Table Generator

```
generate_tables.py   ← reads JSON from results/, outputs LaTeX fragments to results/tables/
```

### Makefile Targets

```bash
make eval-vv                # run V&V (runs inside fenics container via docker exec)
make eval-ablation          # run 2-way KG ablation
make eval-ablation-smart    # run 3-way KG ablation (includes KG Smart)
make eval-metrics           # compute agent quality metrics
make eval-tables            # generate LaTeX tables from JSON
make eval-all               # eval-vv + eval-ablation + eval-metrics + eval-tables
```

---

## 17. Paper Workflow (`paper/`)

The `paper/` directory contains the full LaTeX research paper. It is synced with Overleaf via a dedicated GitHub repository using `git subtree`.

```bash
# Compile locally (requires texlive-full)
make paper-pdf

# Push local edits to GitHub so Overleaf can pull
make paper-push

# Pull Overleaf edits back to local
make paper-pull

# Check status
make paper-status
```

Remote setup (one-time): `git remote add paper-origin git@github.com:ORG/pde-agents-paper.git`

---

*Document updated: PDE Agents v4.0
FEniCSx 0.10.0.post2 · LangGraph · Ollama · nomic-embed-text · Plotly Dash ·
PostgreSQL · Neo4j 5.x · Docling · Celery + Redis · NeoDash · MinIO · Nginx · Gmsh
KG Smart (adaptive RAG) · Evaluation framework · LaTeX paper*
