# PDE Agents - Makefile
# Usage: make <target>

.PHONY: all setup run stop restart logs clean \
        simulate-2d simulate-3d sweep \
        pull-models list-models \
        db-shell db-init \
        jupyter shell-fenics shell-agents \
        test lint

COMPOSE = docker compose
AGENTS_API = http://localhost:8000

all: setup

# ─── Lifecycle ────────────────────────────────────────────────────────────────

setup:
	@chmod +x setup.sh && ./setup.sh

run:
	$(COMPOSE) up -d
	@echo "All services started."
	@echo "  Dashboard  : http://localhost:8050"
	@echo "  Agents API : http://localhost:8000"
	@echo "  JupyterLab : http://localhost:8888"
	@echo "  MLflow     : http://localhost:5000"
	@echo "  MinIO      : http://localhost:9001"

infra:
	$(COMPOSE) up -d postgres redis minio ollama

stop:
	$(COMPOSE) down

restart:
	$(COMPOSE) restart

logs:
	$(COMPOSE) logs -f --tail=50

logs-agents:
	$(COMPOSE) logs -f --tail=100 agents

logs-fenics:
	$(COMPOSE) logs -f --tail=100 fenics-runner

logs-ollama:
	$(COMPOSE) logs -f --tail=50 ollama

clean:
	$(COMPOSE) down -v --remove-orphans
	rm -rf results/* meshes/* mlflow_data/*
	@echo "Cleaned all volumes and result directories."

# ─── Simulations ──────────────────────────────────────────────────────────────

simulate-2d:
	@echo "Running 2D heat equation (steel plate, T=300-500K)..."
	curl -s -X POST $(AGENTS_API)/simulate \
	  -H "Content-Type: application/json" \
	  -d '{"description": "Run a 2D heat equation on a steel plate. Left wall at 300K, right wall at 500K, top and bottom are insulated. Use 64x64 mesh, k=50 W/(mK), rho=7800 kg/m3, cp=500 J/(kgK), run for 100 seconds with dt=0.5s."}' \
	  | python3 -m json.tool

simulate-3d:
	@echo "Running 3D heat equation (aluminum block with convective cooling)..."
	curl -s -X POST $(AGENTS_API)/simulate \
	  -H "Content-Type: application/json" \
	  -d '{"description": "Run a 3D heat equation on an aluminum block. Left wall 273K, right wall 373K, front and back faces have Robin convective BC with h=10 W/(m2K) and T_inf=293K. Use 24x24x24 mesh, k=200 W/(mK), rho=2700 kg/m3, cp=900 J/(kgK)."}' \
	  | python3 -m json.tool

sweep:
	@echo "Running parametric sweep over thermal conductivity k..."
	curl -s -X POST $(AGENTS_API)/run \
	  -H "Content-Type: application/json" \
	  -d '{"task": "Run a parametric sweep varying the thermal conductivity k over values [0.5, 1.0, 2.0, 5.0, 10.0] for a 2D heat equation on a unit square. Left wall T=0, right wall T=1, top and bottom insulated. Use 32x32 mesh, t_end=0.5, dt=0.02. Then analyze the results and tell me which k gives the most uniform temperature distribution.", "max_iterations": 30}' \
	  | python3 -m json.tool

run-from-config:
	@echo "Running simulation from config file..."
	$(COMPOSE) exec fenics-runner python /workspace/simulations/solvers/heat_equation.py \
	  --config /workspace/simulations/configs/heat_2d.json

analyze:
	@[ "$(RUN_ID)" ] || (echo "Usage: make analyze RUN_ID=your_run_id" && exit 1)
	curl -s -X POST $(AGENTS_API)/analyze \
	  -H "Content-Type: application/json" \
	  -d '{"run_ids": ["$(RUN_ID)"], "goal": "Full analysis of this run"}' \
	  | python3 -m json.tool

query:
	@[ "$(Q)" ] || (echo "Usage: make query Q='list all successful 2D runs'" && exit 1)
	curl -s -X POST $(AGENTS_API)/query \
	  -H "Content-Type: application/json" \
	  -d '{"query": "$(Q)"}' \
	  | python3 -m json.tool

# ─── LLM Models ───────────────────────────────────────────────────────────────

pull-models:
	$(COMPOSE) exec ollama bash /ollama-init.sh

list-models:
	curl -s http://localhost:11434/api/tags | python3 -m json.tool

pull-model:
	@[ "$(MODEL)" ] || (echo "Usage: make pull-model MODEL=qwen2.5-coder:32b" && exit 1)
	$(COMPOSE) exec ollama ollama pull $(MODEL)

# ─── Database ─────────────────────────────────────────────────────────────────

db-init:
	$(COMPOSE) exec agents python -c "from database.operations import init_db; init_db(); print('DB initialized.')"

db-shell:
	$(COMPOSE) exec postgres psql -U pde_user -d pde_simulations

db-stats:
	$(COMPOSE) exec postgres psql -U pde_user -d pde_simulations -c \
	  "SELECT status, COUNT(*), AVG(wall_time)::numeric(10,2) as avg_time_s FROM simulation_runs GROUP BY status;"

# List all runs in the database (most recent first)
list-runs:
	@$(COMPOSE) exec agents python3 -c "\
import sys; sys.path.insert(0,'/app'); \
from database.operations import init_db, list_runs; init_db(); \
runs = list_runs(limit=20); \
print('{:<36} {:<10} {:<5} {:<8} {:<12} {}'.format('run_id','status','dim','nx×ny','wall_time(s)','created_at')); \
print('-'*95); \
[print('{:<36} {:<10} {:<5} {:<8} {:<12} {}'.format(r.run_id, r.status.value, str(r.dim or '-'), '{}x{}'.format(r.nx or '-', r.ny or '-'), str(round(r.wall_time,2)) if r.wall_time else '-', str(r.created_at)[:19])) for r in runs]"

# Inspect a specific run in detail: make check-run RUN_ID=<id>
check-run:
	@$(COMPOSE) exec agents python3 -c "\
import sys, json; sys.path.insert(0,'/app'); \
from database.operations import init_db, get_run; init_db(); \
r = get_run('$(RUN_ID)'); \
print('run_id     :', r.run_id) if r else print('Run not found: $(RUN_ID)'); \
r and [print(k.ljust(11), ':', v) for k,v in [('status',r.status.value),('dim',r.dim),('mesh','{}x{}{}'.format(r.nx,r.ny,' x'+str(r.nz) if r.nz else '')),('DOFs',r.n_dofs),('k',r.k),('rho',r.rho),('cp',r.cp),('t_end',r.t_end),('dt',r.dt),('theta',r.theta),('wall_time',str(r.wall_time)+'s' if r.wall_time else '-'),('output_dir',r.output_dir),('created_at',str(r.created_at)[:19])]] and \
print() and print('BCs:') and [print(' ',bc) for bc in (r.config_json if isinstance(r.config_json,dict) else json.loads(r.config_json or '{}')).get('bcs',[])]"

# Check the status of a background API job: make check-job JOB_ID=<id>
check-job:
	@curl -sf http://localhost:8000/jobs/$(JOB_ID) | python3 -m json.tool || echo "Job not found or API down"

# List all submitted background jobs
list-jobs:
	@curl -sf http://localhost:8000/jobs | python3 -m json.tool || echo "API down"

# ─── Shells ───────────────────────────────────────────────────────────────────

jupyter:
	@echo "JupyterLab available at: http://localhost:8888"

shell-fenics:
	$(COMPOSE) exec fenics-runner bash

shell-agents:
	$(COMPOSE) exec agents bash

shell-ollama:
	$(COMPOSE) exec ollama bash

# ─── Testing ──────────────────────────────────────────────────────────────────

test:
	$(COMPOSE) exec agents python -m pytest tests/ -v

test-solver:
	$(COMPOSE) exec fenics-runner python -c "\
	from simulations.solvers.heat_equation import run_2d_heat;\
	r = run_2d_heat(nx=8, ny=8, t_end=0.1, dt=0.05, run_id='test_2d');\
	print('PASS:', r.summary())"

health:
	@echo "=== Service Health ==="
	@curl -sf http://localhost:8000/health | python3 -m json.tool || echo "Agents API: DOWN"
	@curl -sf http://localhost:11434/api/tags | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Ollama: OK  ({len(d[\"models\"])} models)')" || echo "Ollama: DOWN"
	@curl -sf http://localhost:8050 > /dev/null && echo "Dashboard: OK" || echo "Dashboard: DOWN"
	@curl -sf http://localhost:5000 > /dev/null && echo "MLflow: OK" || echo "MLflow: DOWN"
	@curl -sf http://localhost:9000/minio/health/live > /dev/null && echo "MinIO: OK" || echo "MinIO: DOWN"

# ─── NeoDash ──────────────────────────────────────────────────────────────────

seed-neodash:
	$(COMPOSE) exec agents python3 /app/scripts/seed_neodash_dashboard.py

