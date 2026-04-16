#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Model Upgrade Ablation: Qwen3-Coder-Next + Llama4 Scout + Qwen3-Coder-30B
#
# This script:
#   1. Pulls new models into the Ollama container
#   2. Verifies tool-calling works with a smoke test
#   3. Runs the full 17-task ablation (standard + novel) with new models
#   4. Runs it again with old models for direct comparison
#   5. Saves results to evaluation/results/
#
# Run: nohup bash evaluation/run_model_upgrade_ablation.sh &
# Log: tail -f evaluation/results/model_upgrade_ablation.log
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

LOGFILE="$(pwd)/evaluation/results/model_upgrade_ablation.log"
RESULTS_DIR="$(pwd)/evaluation/results"
mkdir -p "$RESULTS_DIR"

exec > >(tee -a "$LOGFILE") 2>&1

echo "═══════════════════════════════════════════════════════════════"
echo "  MODEL UPGRADE ABLATION — $(date)"
echo "═══════════════════════════════════════════════════════════════"

# ── Step 1: Pull new models ──────────────────────────────────────────────────
echo ""
echo "─── Step 1/5: Pulling new models into Ollama ───"

for model in "qwen3-coder-next" "llama4:scout" "qwen3-coder:30b-a3b"; do
    echo "  Pulling $model ..."
    docker compose exec -T ollama ollama pull "$model" 2>&1 || {
        echo "  ⚠ Failed to pull $model — trying alternate tag..."
        # Fallback: some model names may differ slightly on ollama.com
        case "$model" in
            "qwen3-coder-next")
                docker compose exec -T ollama ollama pull "qwen3-coder:next" 2>&1 || \
                docker compose exec -T ollama ollama pull "qwen/qwen3-coder-next" 2>&1 || \
                echo "  ✗ Could not pull qwen3-coder-next — skipping"
                ;;
            "llama4:scout")
                docker compose exec -T ollama ollama pull "llama4" 2>&1 || \
                echo "  ✗ Could not pull llama4:scout — skipping"
                ;;
            "qwen3-coder:30b-a3b")
                docker compose exec -T ollama ollama pull "qwen3-coder:30b" 2>&1 || \
                echo "  ✗ Could not pull qwen3-coder:30b-a3b — skipping"
                ;;
        esac
    }
    echo "  ✓ Done with $model"
done

echo ""
echo "  Currently loaded models:"
docker compose exec -T ollama ollama list 2>&1

# ── Step 2: Determine which models are available ─────────────────────────────
echo ""
echo "─── Step 2/5: Determining available models ───"

AVAILABLE_MODELS=$(docker compose exec -T ollama ollama list 2>&1 | tail -n +2 | awk '{print $1}')
echo "  Available: $AVAILABLE_MODELS"

# Determine new SIM model
NEW_SIM_MODEL=""
for candidate in "qwen3-coder-next" "qwen3-coder:next" "qwen3-coder-next:latest"; do
    if echo "$AVAILABLE_MODELS" | grep -q "$candidate"; then
        NEW_SIM_MODEL="$candidate"
        break
    fi
done

NEW_ANALYTICS_MODEL=""
for candidate in "llama4:scout" "llama4" "llama4:latest"; do
    if echo "$AVAILABLE_MODELS" | grep -q "$candidate"; then
        NEW_ANALYTICS_MODEL="$candidate"
        break
    fi
done

NEW_DB_MODEL=""
for candidate in "qwen3-coder:30b-a3b" "qwen3-coder:30b" "qwen3-coder:30b-a3b-instruct"; do
    if echo "$AVAILABLE_MODELS" | grep -q "$candidate"; then
        NEW_DB_MODEL="$candidate"
        break
    fi
done

echo "  New SIM model:       ${NEW_SIM_MODEL:-NONE (will use fallback)}"
echo "  New Analytics model:  ${NEW_ANALYTICS_MODEL:-NONE (will use fallback)}"
echo "  New DB model:         ${NEW_DB_MODEL:-NONE (will use fallback)}"

# Use best available or fall back to current
SIM_MODEL="${NEW_SIM_MODEL:-qwen2.5-coder:32b}"
ANALYTICS_MODEL="${NEW_ANALYTICS_MODEL:-llama3.3:70b}"
DB_MODEL="${NEW_DB_MODEL:-qwen2.5-coder:14b}"

echo ""
echo "  Final model selection:"
echo "    SIM_MODEL=$SIM_MODEL"
echo "    ANALYTICS_MODEL=$ANALYTICS_MODEL"
echo "    DB_MODEL=$DB_MODEL"

# ── Step 3: Smoke test with new SIM model ────────────────────────────────────
echo ""
echo "─── Step 3/5: Smoke test (single easy task with new SIM model) ───"

docker compose exec -T \
    -e SIMULATION_AGENT_MODEL="$SIM_MODEL" \
    agents python /app/evaluation/ablation/run_ablation.py \
        --tasks E1 --include-smart 2>&1 | tail -20

echo "  ✓ Smoke test complete"

# ── Step 4: Full ablation with NEW models ────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Step 4/5: FULL ABLATION — NEW MODELS ($SIM_MODEL)"
echo "  Started: $(date)"
echo "═══════════════════════════════════════════════════════════════"

# Back up existing results
if [ -f "$RESULTS_DIR/ablation_results.json" ]; then
    cp "$RESULTS_DIR/ablation_results.json" \
       "$RESULTS_DIR/ablation_results_old_models_backup.json"
fi

docker compose exec -T \
    -e SIMULATION_AGENT_MODEL="$SIM_MODEL" \
    agents python /app/evaluation/ablation/run_ablation.py \
        --include-smart 2>&1

# Save results with model label
if [ -f "$RESULTS_DIR/ablation_results.json" ]; then
    cp "$RESULTS_DIR/ablation_results.json" \
       "$RESULTS_DIR/ablation_results_new_models.json"
    echo "  ✓ New model results saved to ablation_results_new_models.json"
fi

echo "  Finished new-model ablation: $(date)"

# ── Step 5: Full ablation with OLD models (for direct comparison) ────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Step 5/5: FULL ABLATION — OLD MODELS (qwen2.5-coder:32b)"
echo "  Started: $(date)"
echo "═══════════════════════════════════════════════════════════════"

docker compose exec -T \
    -e SIMULATION_AGENT_MODEL="qwen2.5-coder:32b" \
    agents python /app/evaluation/ablation/run_ablation.py \
        --include-smart 2>&1

if [ -f "$RESULTS_DIR/ablation_results.json" ]; then
    cp "$RESULTS_DIR/ablation_results.json" \
       "$RESULTS_DIR/ablation_results_old_models.json"
    echo "  ✓ Old model results saved to ablation_results_old_models.json"
fi

echo "  Finished old-model ablation: $(date)"

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ABLATION COMPLETE — $(date)"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  Results:"
echo "    New models: $RESULTS_DIR/ablation_results_new_models.json"
echo "    Old models: $RESULTS_DIR/ablation_results_old_models.json"
echo "    Full log:   $LOGFILE"
echo ""
echo "  Next: compare with"
echo "    python3 -c \"import json; n=json.load(open('$RESULTS_DIR/ablation_results_new_models.json')); o=json.load(open('$RESULTS_DIR/ablation_results_old_models.json')); [print(f'{m}: new={n[m][\"aggregate\"][\"avg_physics_score\"]:.2f} old={o[m][\"aggregate\"][\"avg_physics_score\"]:.2f}') for m in ['kg_on','kg_off','kg_smart']]\""
echo ""
echo "═══════════════════════════════════════════════════════════════"
