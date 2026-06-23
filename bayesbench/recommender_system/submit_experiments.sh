#!/bin/bash
# Example SLURM launcher for the recommender-system cold-start task.
#
# This is a reference example, not a turnkey script — adapt the cluster
# settings below to your environment. It sweeps one model across all
# condition x pop-info cells; pick the model with MODEL=... (default: qwen14b).
#
# Conditions: single_turn, multi_turn_actual
# Pop info:   explicit_types, zero_shot, anonymized
# (multi_turn_actual x zero_shot is skipped: no profile to anchor on)
#
# Usage:
#   ./submit_experiments.sh           # full sweep for $MODEL
#   ./submit_experiments.sh test      # quick validation run
#   ./submit_experiments.sh agg       # aggregate saved experiments for $MODEL
#   MODEL=qwen7b GPUS=1 ./submit_experiments.sh
#   LOCAL=1 ./submit_experiments.sh test   # run directly on this GPU node (no SLURM)

set -e

#==============================================================================
# CONFIGURATION — override via environment variables for your cluster
#==============================================================================
BAYES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${MODEL:-qwen14b}"           # model key
GPUS="${GPUS:-2}"                   # GPUs per job
TIME="${TIME:-16:00:00}"            # SLURM walltime
PARTITION="${PARTITION:-gpu}"       # SLURM partition
CONDA_PATH="${CONDA_PATH:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-bayesbench}"
LOCAL="${LOCAL:-0}"                 # LOCAL=1 runs jobs directly (no SLURM)
#==============================================================================

mkdir -p "${BAYES_DIR}/bayesbench/recommender_system/logs"
cd "${BAYES_DIR}"

ACTION=${1:-run}

CONDITIONS=(single_turn multi_turn_actual)
POP_INFOS=(explicit_types zero_shot anonymized)
TRUE_TYPES="0 1 2 3"
TRIALS="0,1,2,3,4"
N_RATINGS=50
K=1
N_TYPES=4
MAX_MODEL_LEN=16384

submit_job() {
    local job_name=$1
    local cmd=$2
    if [[ "$LOCAL" == "1" ]]; then
        echo "=== [local] ${job_name} ==="
        ( cd "${BAYES_DIR}" && eval "$cmd" ) 2>&1 | tee "bayesbench/recommender_system/logs/${job_name}.out"
        return
    fi
    sbatch --partition="$PARTITION" \
           --nodes=1 \
           --gpus-per-node="$GPUS" \
           --cpus-per-gpu=10 \
           --time="$TIME" \
           --job-name="$job_name" \
           --output="bayesbench/recommender_system/logs/${job_name}.out" \
           --error="bayesbench/recommender_system/logs/${job_name}.err" \
           --wrap="bash -c 'source ${CONDA_PATH}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && cd ${BAYES_DIR} && $cmd'"
    echo "Submitted: $job_name"
}

run_sweep() {
    echo "=== recommender system sweep: ${MODEL} (${GPUS} GPU) — 8 cells ==="
    for pop_info in "${POP_INFOS[@]}"; do
        for cond in "${CONDITIONS[@]}"; do
            if [[ "$cond" == "multi_turn_actual" && "$pop_info" == "zero_shot" ]]; then
                continue
            fi
            job_name="rec_${MODEL}_${cond}_${pop_info}"
            cmd="python -m bayesbench.recommender_system.runner --model ${MODEL} --condition ${cond} --pop-info ${pop_info} --k ${K} --true-type ${TRUE_TYPES} --trials ${TRIALS} --n-ratings ${N_RATINGS} --n-types ${N_TYPES} --max-model-len ${MAX_MODEL_LEN} --batched"
            submit_job "$job_name" "$cmd"
        done
    done
}

run_test() {
    echo "=== Quick test (10 ratings, 1 trial) ==="
    cmd="python -m bayesbench.recommender_system.runner --model ${MODEL} --condition single_turn --pop-info zero_shot --k 1 --true-type 0 --trials 0 --n-ratings 10 --n-types ${N_TYPES} --max-model-len ${MAX_MODEL_LEN}"
    submit_job "rec_test_${MODEL}" "$cmd"
}

run_agg() {
    echo "=== Aggregate ${MODEL} ==="
    cmd="python -m bayesbench.recommender_system.aggregate --experiments-dir bayesbench/recommender_system/experiments --model ${MODEL} --output bayesbench/recommender_system/experiments/${MODEL}_analysis.json"
    submit_job "rec_agg_${MODEL}" "$cmd"
}

case $ACTION in
    run)  run_sweep ;;
    test) run_test ;;
    agg|aggregate) run_agg ;;
    *)
        echo "Usage: [MODEL=<key>] ./submit_experiments.sh [run|test|agg]"
        exit 1
        ;;
esac
