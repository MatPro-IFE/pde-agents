# PDE Agents

A multi-agent ecosystem built on open-source LLMs running locally to solve PDEs with the Finite Element Method, enhanced with a GraphRAG knowledge graph for physics-informed reasoning, and a document intelligence pipeline that cross-references scientific literature to simulation runs.

**Hardware:** 2× NVIDIA RTX PRO 6000 Blackwell Server Edition (~98 GB VRAM each · ~196 GB total) · CUDA 13.1  
**FEM Solver:** DOLFINx (FEniCSx) `0.10.0.post2`

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     ORCHESTRATOR (LangGraph)                              │
│               LLM: llama3.3:70b  (supervisor)                            │
└──────────┬──────────────┬───────────────┬────────────────────────────────┘
           │              │               │
           ▼              ▼               ▼
 ┌──────────────────┐ ┌──────────────┐ ┌───────────────┐
 │  AGENT-1         │ │  AGENT-2     │ │  AGENT-3      │
 │  Simulation      │ │  Analytics   │ │  Database     │
 │  qwen2.5-coder   │ │  llama3.3    │ │  qwen2.5-coder│
 │  :32b            │ │  :70b        │ │  :14b         │
 └───────┬──────────┘ └──────┬───────┘ └──────┬────────┘
         │                   │                 │
         │    ┌──────────────┴──────────────┐  │
         ▼    ▼                             ▼  ▼
 ┌──────────────┐  ┌──────────────┐  ┌────────────────────────────────────┐
 │  FEniCSx     │  │  NumPy       │  │  Neo4j Knowledge Graph (GraphRAG)  │
 │  DOLFINx     │  │  Plotly Dash │  │  ──────────────────────────────────│
 │  2D/3D FEM   │  │  PostgreSQL  │  │  Run nodes + 768-dim embeddings    │
 │  Gmsh meshes │  │  FastAPI     │  │  SIMILAR_TO KNN edges              │
 └──────────────┘  └──────────────┘  │  Reference + ReferenceChunk nodes  │
                                     │  Document chunk vector index (HNSW) │
                                     │  Cross-ref chunks → simulation runs │
                                     └────────────────────────────────────┘

 ┌──────────────────────────────────────────────────────────────────────────┐
 │  Document Intelligence Pipeline (Celery + Docling)                       │
 │  PDFs / TXT / Markdown / Web ebooks → structured chunks → embeddings     │
 │  → (:ReferenceChunk) nodes → CROSS_REFS to similar :Run nodes           │
 └──────────────────────────────────────────────────────────────────────────┘
```

### Agents

| Agent | Model | Role |
|-------|-------|------|
| Simulation Agent | `qwen2.5-coder:32b` | Set up, validate, run, and debug FEM simulations — three KG modes (see below) |
| Analytics Agent  | `llama3.3:70b`      | Analyze results, compare runs, query the knowledge graph |
| Database Agent   | `qwen2.5-coder:14b` | Store results, run SQL queries, catalog studies, search history |
| Orchestrator     | `llama3.3:70b`      | Coordinate all agents, synthesise final report |

#### Simulation Agent — KG integration modes

The Simulation Agent supports three knowledge-graph integration modes, selectable via constructor flag or API query parameter:

| Mode | Flag | Behaviour |
|------|------|-----------|
| **KG On** (default) | — | Mandatory `check_config_warnings` + `query_knowledge_graph` calls before every simulation |
| **KG Off** | `disable_kg=True` | KG tools removed; agent skips straight to `validate_config` → `run_simulation` |
| **KG Smart** | `smart_kg=True` | Warm-start injection of top-3 similar past runs (via HNSW) into system prompt; KG tools remain available but are only invoked after a failure or for unknown materials |

**KG Smart warm-start:** Before the agent loop begins, the task description is embedded with `nomic-embed-text` and the Neo4j HNSW index is queried for the 3 most similar successful past runs. Their configurations are injected directly into the system prompt as few-shot reference examples — no tool call needed. This pattern is inspired by CRAG (corrective RAG) and AriGraph (episodic KG memory).

Ablation results (10 benchmark tasks, 3 difficulty tiers):
- KG On: 60% success, avg 5.3 iterations
- KG Off: 100% success, avg 3.0 iterations
- **KG Smart: 90% success, avg 3.8 iterations** — recovers most KG Off performance while retaining KG access

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
| Redis | `:6379` | Celery task broker (document ingestion queue) |
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
- Start all services including Neo4j, Redis, and NeoDash
- Initialize the PostgreSQL schema
- Auto-seed the knowledge graph with 10 engineering materials, 6 known failure patterns, and 20 physics reference nodes
- Start the Celery worker for asynchronous document ingestion
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
| via SSH tunnel | MinIO console (see note above) |

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

### Use Case 5 — Large-Scale Simulation Study (805 Variations)

Run the comprehensive study script to generate 805 simulation runs across 8 physics studies,
varying geometry, mesh size, material, boundary conditions, initial conditions, time-dependent
BCs, and solver parameters:

```bash
# Execute all 805 runs (runs inside the agents container, skips already-completed runs)
docker exec pde-agents python3 /app/scripts/sweep_full_study.py
```

**Studies covered:**

| Study | Variations | What varies |
|-------|-----------|-------------|
| Geometry study | 45 | 9 geometries × 5 mesh sizes |
| Material study | 100 | 10 materials × 10 mesh densities |
| BC combination study | 120 | 12 BC combos × 10 mesh densities |
| Initial condition study | 50 | 10 IC values × 5 mesh sizes |
| Time-stepping study | 90 | 9 dt values × 10 mesh densities |
| 3D geometry study | 60 | 3 geometries × 20 runs |
| Robin convection study | 100 | 10 h-values × 10 mesh densities |
| Multi-material study | 240 | 10 materials × 8 BCs × 3 meshes |

All runs are automatically added to Neo4j, PostgreSQL, and MinIO, and embedded
for semantic similarity. The knowledge graph grows intelligently with every run.

---

### Use Case 6 — Analytics: Compare Runs

```bash
curl -s -X POST http://localhost:8000/agent/analytics \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Compare the runs steel_plate_64x64 and steel_agent_final. What is the difference in temperature distribution? Which mesh resolution is sufficient and why?"
  }' | python3 -m json.tool
```

---

### Use Case 7 — Database Queries in Natural Language

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

### Use Case 8 — Knowledge Graph Queries

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

# Semantic chunk search across all indexed documents
curl -s "http://localhost:8000/references/search-chunks?query=heat+equation+backward+euler&top_k=5" \
  | python3 -m json.tool

# Re-seed static knowledge (safe to call repeatedly)
curl -s -X POST http://localhost:8000/kg/seed | python3 -m json.tool
```

Example Cypher queries (Neo4j Browser at http://localhost:7474):
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

-- Document chunks cross-referenced to a specific run
MATCH (c:ReferenceChunk)-[xr:CROSS_REFS]->(r:Run {run_id: 'your_run_id'})
MATCH (ref:Reference)-[:HAS_CHUNK]->(c)
RETURN c.heading, c.text, c.classification, xr.score, ref.title, ref.url
ORDER BY xr.score DESC LIMIT 10

-- Full knowledge graph overview
MATCH (n) RETURN labels(n)[0] AS type, count(n) AS count ORDER BY count DESC
```

---

### Use Case 9 — Pre-Run Physics Check

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

### Use Case 10 — Upload External Documents to the Knowledge Base

Upload PDFs, TXT, or Markdown files and have them automatically chunked, embedded, and cross-referenced to relevant simulation runs.

**Via dashboard:** Open the **🧠 Knowledge Graph** tab → **➕ Add to Knowledge Graph** → **📎 Upload File** tab.

**Via API:**
```bash
# Upload a PDF paper — automatically queued for Docling-based structured extraction
curl -s -X POST http://localhost:8000/references/upload \
  -F "file=@paper.pdf" \
  -F "title=Thermal Conductivity of Alloys at Elevated Temperatures" \
  -F "source=ASTM E1461, 2022" \
  -F "subject=thermal conductivity steel copper titanium FEM" \
  -F "ref_type=paper" \
  -F "auto_link_top_k=10" \
  | python3 -m json.tool

# Auto-fill metadata from a DOI (CrossRef API)
# In the dashboard: paste DOI in the DOI field and press Enter

# Check processing status
curl -s http://localhost:8000/references/{ref_id}/status | python3 -m json.tool

# List all uploaded references with processing status
curl -s http://localhost:8000/references/uploaded | python3 -m json.tool

# Get structured chunks for a specific document
curl -s http://localhost:8000/references/{ref_id}/chunks | python3 -m json.tool

# Pin a document to a specific simulation run
curl -s -X POST http://localhost:8000/references/{ref_id}/link/{run_id}
```

**What happens after upload:**
1. File stored in MinIO (`reference-uploads/`)
2. `(:Reference)` node created in Neo4j with `process_status: queued`
3. Celery worker picks up the task and begins Docling-based structured parsing
4. Each section/paragraph extracted as a `(:ReferenceChunk)` node with heading, type, and classification
5. Each chunk embedded with `nomic-embed-text` (768-dim vector)
6. Chunks cross-referenced to semantically similar `(:Run)` nodes via `CROSS_REFS` edges
7. Chunks linked to matching `(:Material)`, `(:BCConfig)`, `(:Domain)` entities via `RELATES_TO` edges
8. `process_status` updated to `completed` with counts of chunks and cross-refs

---

### Use Case 11 — Index Web-Based Tutorials and Ebooks

Crawl and index web resources (online tutorials, ebooks, documentation sites) so agents can cite and reason from them alongside your simulation runs.

**Via dashboard:** Open the **🧠 Knowledge Graph** tab → **➕ Add to Knowledge Graph** → **🌐 Web Resource URL** tab.

**Via API:**
```bash
# Index the FEniCSx tutorial (crawls up to 50 pages)
curl -s -X POST http://localhost:8000/references/fetch-url \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://jsdokken.com/dolfinx-tutorial/",
    "title": "FEniCSx Tutorial — Dokken",
    "subject": "FEniCSx DOLFINx FEM heat equation Poisson boundary conditions mesh",
    "max_pages": 50
  }' | python3 -m json.tool
```

**What happens:**
1. `(:Reference)` parent node created with `is_web: true`
2. Celery worker crawls the site (BFS, up to `max_pages`, polite 1s delays)
3. Each page parsed through Docling's HTML pipeline
4. Structured chunks extracted (text, code, equations, tables)
5. Each page creates a child `(:Reference {type: "web_page"})` with its exact URL
6. Chunks embedded and cross-referenced to simulation runs

**Example result (FEniCSx tutorial, 50 pages):**
- ~300+ chunks extracted and embedded
- 1000+ cross-references created to simulation runs
- Semantic search for "heat equation backward Euler time stepping" returns chunk from `chapter2/heat_equation.html` with score 0.871 and a **clickable link** to that exact page

**Useful resource URLs to index:**
- FEniCSx Tutorial: `https://jsdokken.com/dolfinx-tutorial/`
- FEniCS Book: `https://fenicsproject.org/pub/tutorial/html/`
- OpenFOAM User Guide: `https://www.openfoam.com/documentation/user-guide`
- Any HTML-based ebook or documentation site

---

### Use Case 12 — Semantic Chunk Search (Dashboard)

The **🔬 Semantic Chunk Search** panel (bottom of the 🧠 Knowledge Graph tab) searches across all document chunks indexed from uploaded PDFs and web resources.

- Type a query (e.g. "thermal conductivity steel at high temperature") and press **Enter** or click **Search Chunks**
- Results show the most semantically relevant paragraphs, equations, and sections
- Each result card shows:
  - **Clickable title** → opens the source page (web resources) or paper (if URL/DOI provided)
  - Classification badge: `material`, `bc`, `solver`, `domain`, or `general`
  - Chunk type: `text`, `code`, `equation`, `table`
  - Similarity score
  - **"🌐 Open page"** or **"📄 View source"** button for direct access

---

### Use Case 13 — Run Explorer (Dashboard)

The **🔎 Run Explorer** tab provides a visual interface to browse every simulation run:

- **Left panel:** searchable, filterable run list (status, dimension, keyword)
- **Right panel:** detailed view with sub-tabs for:
  - **Overview** — KPIs (T_max, T_min, wall time, DOFs) + SIMILAR_TO neighbour table
  - **Agent Timeline** — step-by-step trace of every agent reasoning and tool-call
  - **Config** — full simulation configuration used (for reproducibility)
  - **Files** — MinIO file listing with object names and sizes
  - **Recommendations** — agent suggestions from the run

---

### Use Case 14 — Knowledge Graph Tab (Dashboard)

The **🧠 Knowledge Graph** tab exposes all GraphRAG features:

**Graph Statistics panel:** Real-time counts of all node types, embedding coverage percentage, SIMILAR_TO edge count, Reference count, and ReferenceChunk count.

**Semantic Run Search:** Type a free-text description and the KG finds the top-5 similar past runs using the HNSW vector index. Results show similarity scores with progress bars.

**Physics Reference Browser:** Browse all 20 curated reference facts filtered by type:
- 📐 Material Properties — k(T) and cp(T) dependence, phase transition limits
- 🔲 BC Practice — realistic h coefficients, heat flux magnitudes, temperature validity
- ⚙️ Solver Guidance — mesh resolution rules, Fourier number criterion, P2 vs P1
- 🌐 Domain Physics — radiation at micro-scale, buoyancy at structural scale, thermal time constants

**➕ Add to Knowledge Graph panel:**
- **📎 Upload File** tab: DOI quick-fill (press Enter to auto-fetch from CrossRef) + file drag-and-drop
- **🌐 Web Resource URL** tab: paste any URL, set page limit, press Enter or click Fetch

**🔬 Semantic Chunk Search:** Full-text semantic search across all indexed document chunks with clickable source links.

**NeoDash launcher:** Opens the open-source graph visualization tool at port 9001.

---

### Use Case 15 — Direct Python (inside containers)

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
# Stats: {'total_runs': 805, 'embedded_runs': 805, 'references': 44,
#          'ref_chunks': 325, 'materials': 10, 'bc_configs': 4, 'domains': 4, ...}

# Semantic similarity search using vector embeddings
similar = kg.get_similar_runs_semantic({
    "k": 50.0, "dim": 2, "Lx": 0.08, "Ly": 0.08,
    "bcs": [{"type": "dirichlet"}, {"type": "robin", "h": 25}],
    "geometry": {"type": "l_shape"},
}, top_k=5)
for r in similar:
    print(f"  {r['run_id'][:16]}  score={r['similarity_score']:.4f}  T_max={r['t_max']:.0f}K")

# Semantic chunk search across all documents
from knowledge_graph.embeddings import get_embedder
vec = get_embedder().embed_text("backward Euler time stepping heat equation stability")
chunks = kg.search_chunks_by_query(vec, top_k=5)
for c in chunks:
    print(f"  [{c['score']:.3f}] {c['ref_title']} — {c['heading'][:60]}")
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
print(result)  # {'processed': 805, 'total_similar_to_in_graph': 4025}
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

(:Reference {
    ref_id, title, type, subject, source, url, tags,
    is_uploaded, is_web, process_status,
    n_pages, n_chunks, n_tables, parse_method,
    chunks_stored, chunks_embedded, cross_refs
})
  Types: material_property | bc_practice | solver_guidance | domain_physics
       | paper | report | handbook | standard | web_resource | web_page

(:ReferenceChunk {
    chunk_id, ref_id, chunk_index, heading, text,
    chunk_type,        ← text | code | equation | table | list
    classification,    ← material | bc | solver | domain | general
    confidence, page,
    embedding          ← 768-dim nomic-embed-text vector
})

Relationships:
  (:Run)-[:USES_MATERIAL {confidence}]->(:Material)
  (:Run)-[:TRIGGERED {detected_at}]->(:KnownIssue)
  (:Run)-[:USES_BC_CONFIG]->(:BCConfig)
  (:Run)-[:ON_DOMAIN]->(:Domain)
  (:Material)-[:HAS_THERMAL_CLASS]->(:ThermalClass)
  (:Run)-[:SIMILAR_TO {score, updated_at}]->(:Run)   — KNN semantic edges
  (:Run)-[:SPAWNED_FROM]->(:Run)                     — created from agent suggestion
  (:Material)-[:HAS_REFERENCE]->(:Reference)
  (:BCConfig)-[:HAS_REFERENCE]->(:Reference)
  (:Domain)-[:HAS_REFERENCE]->(:Reference)
  (:Run)-[:CITES]->(:Reference)                      — manually pinned or auto-linked
  (:Reference)-[:HAS_CHUNK]->(:ReferenceChunk)       — document sections
  (:Reference)-[:HAS_PAGE]->(:Reference)             — web resource → page hierarchy
  (:ReferenceChunk)-[:CROSS_REFS {score}]->(:Run)    — chunk ↔ similar run
  (:ReferenceChunk)-[:RELATES_TO]->(:Material)       — entity linking
  (:ReferenceChunk)-[:RELATES_TO]->(:BCConfig)
  (:ReferenceChunk)-[:RELATES_TO]->(:Domain)

HNSW Vector Indexes:
  run_embedding_index   — on (:Run).embedding    (768-dim, cosine)
  chunk_embedding_index — on (:ReferenceChunk).embedding  (768-dim, cosine)
```

### Feature 1 — Vector Embeddings + Semantic Search

Every Run node stores a 768-dimensional `embedding` computed by `nomic-embed-text`. A **Neo4j HNSW vector index** enables sub-millisecond semantic similarity search.

Every `ReferenceChunk` node also stores a 768-dim embedding, enabling semantic search across all indexed literature and web resources through a second HNSW index.

### Feature 2 — SIMILAR_TO KNN Edges

After every run, up to 5 `SIMILAR_TO` edges are created to the most similar embedded neighbours (cosine ≥ 0.85). These precompute the neighbourhood graph for fast agent traversal.

### Feature 3 — Physics Reference Nodes (Curated)

20 curated reference facts are seeded at startup and linked to Material, BCConfig, and Domain nodes:

| Type | Count | Examples |
|------|-------|---------|
| `material_property` | 8 | Steel k(T) dependence, Curie point anomaly, copper melting, silicon k drop |
| `bc_practice` | 5 | h = 5–25 W/(m²·K) natural air, h = 500–10,000 water cooling, heat flux ranges |
| `solver_guidance` | 4 | Min 10 elements, Fourier criterion, P2 vs P1, CG vs GMRES |
| `domain_physics` | 3 | Radiation at micro-scale, convection at structural scale, τ = ρc_pL²/(π²k) |

### Feature 4 — Document Intelligence Pipeline (Docling + Celery)

Uploaded documents and web resources are processed asynchronously:

1. **Docling** parses PDFs/HTML/DOCX into structured sections with headings and types
2. **HybridChunker** produces overlapping, context-aware chunks (max 256 tokens)
3. **Classification** labels each chunk: `material`, `bc`, `solver`, `domain`, or `general`
4. **nomic-embed-text** embeds each chunk (768-dim vector)
5. **CROSS_REFS** edges link chunks to semantically similar Run nodes (score ≥ 0.78)
6. **RELATES_TO** edges link chunks to matching Material/BCConfig/Domain entities
7. Web crawls discover all linked pages within the same domain path (BFS, polite 1s delays)

---

## Project Structure

```
pde-agents/
├── agents/
│   ├── base_agent.py          # LangGraph ReAct base: reason → act loop + tool-call parser
│   ├── simulation_agent.py    # Agent-1: KG-aware setup, run, debug FEM simulations
│   ├── analytics_agent.py     # Agent-2: analysis, comparison, KG pattern queries
│   └── database_agent.py      # Agent-3: DB storage, SQL queries, catalog studies, search history
│
├── tools/                     # LangChain @tool functions (agent "hands")
│   ├── simulation_tools.py    # run_simulation, validate_config, run_parametric_sweep
│   ├── analytics_tools.py     # analyze_run, compare_runs, list_runs_for_analysis
│   ├── database_tools.py      # search_history, get_run_summary, query_runs
│   └── knowledge_tools.py     # check_config_warnings, query_knowledge_graph,
│                              #   get_physics_references
│
├── knowledge_graph/
│   ├── graph.py               # SimulationKnowledgeGraph: full schema, both HNSW
│   │                          #   vector indexes, semantic search, SIMILAR_TO edges,
│   │                          #   Reference + ReferenceChunk management, chunk search,
│   │                          #   cross-ref linking, web resource ingestion
│   ├── embeddings.py          # OllamaEmbedder: nomic-embed-text 768-dim, run_to_text()
│   ├── references.py          # 20 curated physics reference entries
│   ├── rules.py               # Pure-Python rule engine (9 rules: IC, CFL, mesh, BCs…)
│   ├── seeder.py              # Static knowledge: 10 materials + 6 failure patterns
│   ├── document_processor.py  # Docling-based structured PDF/HTML extraction pipeline:
│   │                          #   DocumentChunk, ParsedDocument, parse_document(),
│   │                          #   _classify_chunk(), embed_chunks(); pypdf fallback
│   ├── tasks.py               # Celery async tasks: ingest_document_task (PDF/file),
│   │                          #   ingest_web_resource_task (site crawl + index)
│   └── web_fetcher.py         # BFS web crawler: discover_pages(), fetch_page(),
│                              #   parse_html_with_docling(), fetch_and_parse_site()
│
├── orchestrator/
│   ├── graph.py               # Multi-agent supervisor graph (LangGraph)
│   └── api.py                 # FastAPI REST interface + /kg/* + /references/* endpoints
│
├── simulations/
│   ├── solvers/
│   │   └── heat_equation.py   # DOLFINx 0.10 FEM solver (2D/3D, all BC types,
│   │                          #   built-in + Gmsh mesh dispatch)
│   ├── geometry/
│   │   ├── __init__.py
│   │   └── gmsh_geometries.py # 9 Gmsh geometry builders (rectangle, l_shape, circle,
│   │                          #   annulus, hollow_rectangle, t_shape, stepped_notch,
│   │                          #   box, cylinder) + GmshMeshResult dataclass
│   └── configs/
│       ├── heat_2d.json       # Steel plate example config
│       └── heat_3d.json       # Aluminum block with convection config
│
├── database/
│   ├── models.py              # SQLAlchemy ORM (SimulationRun, AgentRunLog, …)
│   ├── operations.py          # CRUD, log_agent_step, search_runs, get_agent_logs
│   └── init.sql               # PostgreSQL init (extensions, permissions)
│
├── visualization/
│   └── dashboard.py           # Plotly Dash: Overview, Field Viewer, Convergence,
│                              #   Parametric, Agent Chat, 🧠 Knowledge Graph,
│                              #   🔎 Run Explorer; Enter-key on all search fields
│
├── scripts/
│   ├── sweep_full_study.py    # 805-run comprehensive parameter sweep (8 studies)
│   ├── migrate_kg_schema_v2.py  # One-off: add BCConfig/Domain/ThermalClass to existing runs
│   ├── seed_knowledge_graph.py  # 23 representative simulations for KG bootstrapping
│   ├── seed_bc_geometry_study.py  # BC + geometry parametric study
│   └── ollama-init.sh         # Pull all required Ollama models
│
├── docker/
│   ├── Dockerfile.fenics      # dolfinx/dolfinx:stable + Gmsh + uvicorn API
│   ├── Dockerfile.agents      # Python 3.11 + LangGraph + FastAPI + neo4j + Docling
│   ├── Dockerfile.dashboard   # Python 3.11 + Dash + pandas + neo4j driver
│   └── fenics_runner_api.py   # FastAPI server inside the FEniCSx container
│
├── nginx/
│   └── nginx.conf             # Nginx reverse proxy (subpath routing, WebSocket)
│
├── evaluation/                # Research paper evaluation framework
│   ├── benchmarks/
│   │   ├── analytical_solutions.py  # 3 closed-form V&V benchmark cases
│   │   └── vv_runner.py             # V&V convergence study (5 mesh resolutions)
│   ├── ablation/
│   │   ├── benchmark_tasks.py       # 10 NL benchmark tasks (easy/medium/hard)
│   │   └── run_ablation.py          # 3-way KG ablation runner (--include-smart)
│   ├── metrics/
│   │   └── agent_quality.py         # Production metrics from PostgreSQL
│   ├── generate_tables.py           # LaTeX table generator from JSON results
│   └── results/                     # JSON outputs + generated LaTeX tables
│
├── paper/                     # Research paper (LaTeX)
│   ├── main.tex               # Main document (compile with pdflatex + bibtex)
│   ├── references.bib         # BibTeX (29 entries incl. CRAG, AriGraph)
│   ├── figs/                  # TikZ/PGFPlots figures
│   └── tables/                # LaTeX table fragments (\input{} in main.tex)
│
├── docker-compose.yml         # Full service stack incl. NeoDash (port 9001), Celery worker
├── CLAUDE.md                  # Comprehensive project context file
├── env.example                # Environment variable template (copy to .env)
├── requirements.txt           # Python deps: LangGraph, FastAPI, neo4j, Docling, Celery, pypdf
├── Makefile                   # Common commands (incl. eval-* and paper-* targets)
└── setup.sh                   # One-time setup script
```

---

## LLM Models

| Model | Params | VRAM | Role |
|-------|--------|------|------|
| `qwen2.5-coder:14b` | 14B | ~9 GB  | Database Agent — fast structured queries |
| `qwen2.5-coder:32b` | 32B | ~19 GB | Simulation Agent — code generation & debugging |
| `llama3.3:70b`      | 70B | ~42 GB | Analytics Agent + Orchestrator — reasoning |
| `nomic-embed-text`  | 137M | <1 GB | Embedding — 768-dim vectors for runs and document chunks |

All LLMs fit simultaneously (~70 GB out of 196 GB available VRAM).

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

# ── Evaluation (paper experiments) ──────────────────────────────────────────
make eval-vv                # V&V convergence benchmarks (runs inside fenics container)
make eval-ablation          # 2-way KG ablation: KG On vs KG Off
make eval-ablation-smart    # 3-way KG ablation: KG On vs KG Off vs KG Smart
make eval-metrics           # production agent quality metrics from PostgreSQL
make eval-tables            # generate LaTeX tables from JSON results
make eval-all               # run V&V + 2-way ablation + metrics + tables

# ── Paper (Overleaf sync via GitHub subtree) ─────────────────────────────────
# Setup (one-time): git remote add paper-origin git@github.com:ORG/pde-agents-paper.git
make paper-push             # push local paper/ edits → GitHub → Overleaf pulls
make paper-pull             # pull Overleaf edits ← GitHub → local paper/
make paper-status           # show uncommitted paper changes + commits ahead of remote
make paper-pdf              # compile paper/main.tex locally (requires texlive)
```

---

## Troubleshooting

### Knowledge Graph tab shows "Knowledge graph unavailable"
```bash
# Recreate the container (picks up updated docker-compose.yml)
docker compose up dashboard -d --force-recreate

# Verify
docker exec pde-dashboard python3 -c "
import sys; sys.path.insert(0, '/app')
from knowledge_graph.graph import get_kg
kg = get_kg()
print('Available:', kg.available)
print('Stats:', kg.stats())
"
```

### Document upload stuck in "queued" status — Celery worker not running
```bash
# Check worker is alive
docker exec pde-agents cat /tmp/celery.log | tail -5

# Restart Celery worker manually
docker exec pde-agents bash -c "
kill \$(cat /tmp/celery.pid 2>/dev/null) 2>/dev/null; sleep 2
celery -A knowledge_graph.tasks:celery_app worker --loglevel=info \
  --concurrency=2 -Q document_ingestion,celery \
  --detach --pidfile=/tmp/celery.pid --logfile=/tmp/celery.log
"
```

### NeoDash not reachable at port 9001
```bash
docker compose up neodash -d
docker logs pde-neodash --tail=20
```

### MinIO console access
The MinIO console is on host port 9002 (port 9001 is now NeoDash). Access via SSH tunnel:
```bash
ssh -L 19002:localhost:9002 -N <server>   # then open http://localhost:19002
```

### Embedding model not available (semantic search returns empty)
```bash
docker exec pde-ollama ollama pull nomic-embed-text
docker exec pde-agents python3 -c "
import sys; sys.path.insert(0, '/app')
from knowledge_graph.graph import get_kg
kg = get_kg()
print(kg.backfill_embeddings(batch_size=50))
print(kg.build_all_similar_to_edges(k=5, min_score=0.85))
"
```

### Knowledge graph is empty after restart
```bash
# Seed static knowledge (materials + failure patterns + references)
curl -s -X POST http://localhost:8000/kg/seed | python3 -m json.tool

# Re-run representative simulations to rebuild Run nodes
docker exec pde-agents python3 /app/scripts/seed_knowledge_graph.py

# Backfill embeddings and KNN edges
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
| `neo4j_data` | All Run nodes, ReferenceChunk nodes, embeddings, SIMILAR_TO edges, Reference links |
| `postgres_data` | All simulation records, agent logs |
| `minio_data` | All uploaded XDMF/NPY result files and reference documents |
| `redis_data` | Any queued Celery document ingestion jobs |

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
