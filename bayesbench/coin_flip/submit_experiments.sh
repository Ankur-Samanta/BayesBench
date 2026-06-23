#!/bin/bash
# Example SLURM launcher for the Coin Flip task.
#
# This is a reference example, not a turnkey script — adapt the cluster
# settings below to your environment. It sweeps one model across all
# conditions x coin specs; pick the model with MODEL=... (default: qwen14b).
#
# Conditions: single_turn, multi_turn_minimal, multi_turn_actual
# Coin specs: unspecified, unknown_bias, fair
#
# Usage:
#   ./submit_experiments.sh           # full sweep for $MODEL (via SLURM)
#   ./submit_experiments.sh test      # quick 20-flip validation run
#   ./submit_experiments.sh agg       # aggregate saved experiments for $MODEL
#   MODEL=qwen7b GPUS=2 ./submit_experiments.sh
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

mkdir -p "${BAYES_DIR}/bayesbench/coin_flip/logs"
cd "${BAYES_DIR}"

ACTION=${1:-run}

CONDITIONS=(single_turn multi_turn_minimal multi_turn_actual)
COIN_SPECS=(unspecified unknown_bias fair)
P_VALUES="0.25 0.5 0.75"
TRIALS="0,1,2,3,4"
N_FLIPS=100
K=1

submit_job() {
    local job_name=$1
    local cmd=$2
    if [[ "$LOCAL" == "1" ]]; then
        echo "=== [local] ${job_name} ==="
        ( cd "${BAYES_DIR}" && eval "$cmd" ) 2>&1 | tee "bayesbench/coin_flip/logs/${job_name}.out"
        return
    fi
    sbatch --partition="$PARTITION" \
           --nodes=1 \
           --gpus-per-node="$GPUS" \
           --cpus-per-gpu=10 \
           --time="$TIME" \
           --job-name="$job_name" \
           --output="bayesbench/coin_flip/logs/${job_name}.out" \
           --error="bayesbench/coin_flip/logs/${job_name}.err" \
           --wrap="bash -c 'source ${CONDA_PATH}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && cd ${BAYES_DIR} && $cmd'"
    echo "Submitted: $job_name"
}

run_sweep() {
    echo "=== Coin Flip sweep: ${MODEL} (${GPUS} GPU) — 9 jobs ==="
    for coin_spec in "${COIN_SPECS[@]}"; do
        for cond in "${CONDITIONS[@]}"; do
            job_name="coinflip_${MODEL}_${cond}_${coin_spec}"
            cmd="python -m bayesbench.coin_flip.runner --model ${MODEL} --condition ${cond} --k ${K} --coin-spec ${coin_spec} --p ${P_VALUES} --trials ${TRIALS} --n-flips ${N_FLIPS}"
            submit_job "$job_name" "$cmd"
        done
    done
}

run_test() {
    echo "=== Quick test (20 flips, 1 trial) ==="
    cmd="python -m bayesbench.coin_flip.runner --model ${MODEL} --condition single_turn --k 1 --coin-spec unknown_bias --p 0.5 --trials 0 --n-flips 20"
    submit_job "coinflip_test_${MODEL}" "$cmd"
}

run_agg() {
    echo "=== Aggregate ${MODEL} ==="
    cmd="python -m bayesbench.coin_flip.aggregate --experiments-dir bayesbench/coin_flip/experiments --model ${MODEL} --output bayesbench/coin_flip/experiments/${MODEL}_analysis.json"
    submit_job "coinflip_agg_${MODEL}" "$cmd"
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
