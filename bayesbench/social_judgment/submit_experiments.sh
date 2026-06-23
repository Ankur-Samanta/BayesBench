#!/bin/bash
# Example SLURM launcher for the AITA moral-judgment task.
#
# This is a reference example, not a turnkey script — adapt the cluster
# settings below to your environment. It submits the scripted + active
# conditions for one model; pick the model with MODEL=... (default: qwen14b).
#
# Conditions: single_turn, multi_turn_passive (scripted, 1 run each)
#             multi_turn_active x {neutral, conceding, defending} (5 runs each)
#
# Usage:
#   ./submit_experiments.sh           # scripted + active for $MODEL (posts 0-100)
#   ./submit_experiments.sh scripted  # scripted conditions only
#   ./submit_experiments.sh active    # active conditions only
#   ./submit_experiments.sh test      # quick 5-post validation
#   ./submit_experiments.sh agg       # aggregate + DDM for $MODEL
#   MODEL=qwen7b GPUS=2 ./submit_experiments.sh
#   LOCAL=1 ./submit_experiments.sh test   # run directly on this GPU node (no SLURM)

set -e

#==============================================================================
# CONFIGURATION — override via environment variables for your cluster
#==============================================================================
BAYES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${MODEL:-qwen14b}"           # model key
GPUS="${GPUS:-2}"                   # GPUs per job
TIME="${TIME:-48:00:00}"            # SLURM walltime
PARTITION="${PARTITION:-gpu}"       # SLURM partition
CONDA_PATH="${CONDA_PATH:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-bayesbench}"
LOCAL="${LOCAL:-0}"                 # LOCAL=1 runs jobs directly (no SLURM)
#==============================================================================

mkdir -p "${BAYES_DIR}/bayesbench/social_judgment/logs"
cd "${BAYES_DIR}"

ACTION=${1:-run}

STYLES=(neutral conceding defending)

submit_job() {
    local job_name=$1
    local cmd=$2
    if [[ "$LOCAL" == "1" ]]; then
        echo "=== [local] ${job_name} ==="
        ( cd "${BAYES_DIR}" && eval "$cmd" ) 2>&1 | tee "bayesbench/social_judgment/logs/${job_name}.out"
        return
    fi
    sbatch --partition="$PARTITION" \
           --nodes=1 \
           --gpus-per-node="$GPUS" \
           --cpus-per-gpu=10 \
           --time="$TIME" \
           --job-name="$job_name" \
           --output="bayesbench/social_judgment/logs/${job_name}.out" \
           --error="bayesbench/social_judgment/logs/${job_name}.err" \
           --wrap="bash -c 'source ${CONDA_PATH}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && cd ${BAYES_DIR} && $cmd'"
    echo "Submitted: $job_name"
}

submit_scripted() {
    for cond in single_turn multi_turn_passive; do
        job_name="eval_${MODEL}_${cond}"
        cmd="python -m bayesbench.social_judgment.runner --model ${MODEL} --condition ${cond} --posts 0-100"
        submit_job "$job_name" "$cmd"
    done
}

submit_active() {
    for style in "${STYLES[@]}"; do
        job_name="eval_${MODEL}_active_${style}"
        cmd="python -m bayesbench.social_judgment.runner --model ${MODEL} --condition multi_turn_active --style ${style} --posts 0-100 --runs 5 --batch-size 32"
        submit_job "$job_name" "$cmd"
    done
}

run_test() {
    echo "=== Quick test (5 posts) ==="
    cmd="python -m bayesbench.social_judgment.runner --model ${MODEL} --condition single_turn --posts 0-5"
    submit_job "eval_test_${MODEL}" "$cmd"
}

run_agg() {
    echo "=== Aggregate + DDM ${MODEL} ==="
    cmd="python -m bayesbench.social_judgment.aggregate --experiments-dir bayesbench/social_judgment/experiments --model ${MODEL} --output bayesbench/social_judgment/experiments/${MODEL}_analysis.json && python -m bayesbench.social_judgment.ddm --experiments-dir bayesbench/social_judgment/experiments --model ${MODEL} --output bayesbench/social_judgment/experiments/${MODEL}_ddm.json"
    submit_job "eval_agg_${MODEL}" "$cmd"
}

case $ACTION in
    run)
        echo "=== AITA sweep: ${MODEL} (${GPUS} GPU) — scripted + active ==="
        submit_scripted
        submit_active
        ;;
    scripted) submit_scripted ;;
    active)   submit_active ;;
    test)     run_test ;;
    agg|aggregate) run_agg ;;
    *)
        echo "Usage: [MODEL=<key>] ./submit_experiments.sh [run|scripted|active|test|agg]"
        exit 1
        ;;
esac
