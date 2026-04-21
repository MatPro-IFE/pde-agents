#!/usr/bin/env bash
#
# Master experiment runner v2 — methodologically sound ablation + KG growth.
#
# Protocol:
#   1. Snapshot the current KG
#   2. Apply DB migration (add experiment_phase column)
#   3. Run frozen-KG ablation (50 tasks × 4 modes, KG_READ_ONLY=true)
#   4. Restore KG snapshot
#   5. Run clean KG growth experiment (empty → full, KG writes enabled)
#   6. Restore KG snapshot again
#   7. Run statistical analysis
#
# Usage:
#   docker compose exec agents bash /app/evaluation/run_experiments_v2.sh
#
# Or run phases individually:
#   docker compose exec agents bash /app/evaluation/run_experiments_v2.sh --phase ablation
#   docker compose exec agents bash /app/evaluation/run_experiments_v2.sh --phase growth
#   docker compose exec agents bash /app/evaluation/run_experiments_v2.sh --phase analyze

set -euo pipefail
cd /app

PHASE="${1:---all}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/app/evaluation/results/logs"
mkdir -p "$LOG_DIR"

echo "═══════════════════════════════════════════════════════════════"
echo "  EXPERIMENT RUNNER v2 — ${TIMESTAMP}"
echo "  Phase: ${PHASE}"
echo "═══════════════════════════════════════════════════════════════"

# ─── Step 1: DB Migration ────────────────────────────────────────────────────
run_migration() {
    echo ""
    echo "──── Step 1: DB Migration (add experiment_phase column) ────"
    python3 -c "
from database.operations import get_engine
from sqlalchemy import text
engine = get_engine()
with engine.connect() as conn:
    # Add column if it doesn't exist
    try:
        conn.execute(text(
            'ALTER TABLE simulation_runs ADD COLUMN IF NOT EXISTS '
            'experiment_phase VARCHAR(64)'
        ))
        conn.execute(text(
            'CREATE INDEX IF NOT EXISTS ix_simulation_runs_experiment_phase '
            'ON simulation_runs (experiment_phase)'
        ))
        conn.commit()
        print('  Migration complete.')
    except Exception as e:
        print(f'  Migration note: {e}')
        conn.rollback()
"
}

# ─── Step 2: KG Snapshot ─────────────────────────────────────────────────────
snapshot_kg() {
    echo ""
    echo "──── Step 2: KG Snapshot ────"
    python3 /app/evaluation/kg_snapshot.py save --name "pre_experiment_${TIMESTAMP}"
    python3 /app/evaluation/kg_snapshot.py stats
}

# ─── Step 3: Frozen-KG Ablation (50 tasks × 4 modes) ────────────────────────
run_ablation() {
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  Step 3: FROZEN-KG ABLATION (50 tasks × 4 modes)"
    echo "══════════════════════════════════════════════════════════════"
    
    export KG_READ_ONLY=true
    export EXPERIMENT_PHASE=ablation_v2
    
    python3 /app/evaluation/ablation/run_ablation_v2.py \
        --direct \
        --seed 42 \
        2>&1 | tee "$LOG_DIR/ablation_v2_${TIMESTAMP}.log"
    
    unset KG_READ_ONLY
    unset EXPERIMENT_PHASE
}

# ─── Step 4: Restore KG ─────────────────────────────────────────────────────
restore_kg() {
    echo ""
    echo "──── Step 4: Restoring KG snapshot ────"
    python3 /app/evaluation/kg_snapshot.py restore --name "pre_experiment_${TIMESTAMP}"
}

# ─── Step 5: KG Growth Experiment ────────────────────────────────────────────
run_growth() {
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  Step 5: KG GROWTH EXPERIMENT (empty → full)"
    echo "══════════════════════════════════════════════════════════════"
    
    export EXPERIMENT_PHASE=kg_growth_clean
    
    python3 /app/evaluation/kg_growth_experiment.py \
        --batch-size 5 \
        --seed 42 \
        2>&1 | tee "$LOG_DIR/kg_growth_${TIMESTAMP}.log"
    
    unset EXPERIMENT_PHASE
    
    # Restore original KG after experiment
    echo "  Restoring KG after growth experiment..."
    python3 /app/evaluation/kg_snapshot.py restore --name "pre_experiment_${TIMESTAMP}"
}

# ─── Step 6: Statistical Analysis ────────────────────────────────────────────
run_analysis() {
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  Step 6: STATISTICAL ANALYSIS"
    echo "══════════════════════════════════════════════════════════════"
    
    if [ -f /app/evaluation/results/ablation_v2_results.json ]; then
        python3 /app/evaluation/statistical_analysis.py \
            /app/evaluation/results/ablation_v2_results.json \
            2>&1 | tee "$LOG_DIR/statistics_${TIMESTAMP}.log"
    else
        echo "  WARNING: No ablation v2 results found. Run ablation first."
    fi
}

# ─── Dispatch ────────────────────────────────────────────────────────────────
case "$PHASE" in
    --all)
        run_migration
        snapshot_kg
        run_ablation
        restore_kg
        run_growth
        run_analysis
        ;;
    --phase)
        case "${2:-}" in
            ablation)
                run_migration
                snapshot_kg
                run_ablation
                restore_kg
                ;;
            growth)
                snapshot_kg
                run_growth
                ;;
            analyze)
                run_analysis
                ;;
            *)
                echo "Unknown phase: ${2:-}"
                echo "Available: ablation, growth, analyze"
                exit 1
                ;;
        esac
        ;;
    *)
        echo "Usage: $0 [--all | --phase ablation|growth|analyze]"
        exit 1
        ;;
esac

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  EXPERIMENT RUNNER v2 COMPLETE — $(date)"
echo "═══════════════════════════════════════════════════════════════"
