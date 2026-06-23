#!/bin/bash
# Example SLURM launcher for the Medical triage task.
#
# This is a reference example, not a turnkey script — adapt the cluster
# settings below to your environment. It sweeps one model across all
# patient profiles; pick the model with MODEL=... (default: qwen14b).
#
# Pop info: zero_shot
# Profiles: accurate, hypochondriac, minimizer, cyberchondriac
#
# Usage:
#   ./submit_experiments.sh                 # full sweep for $MODEL (cases 0-100)
#   ./submit_experiments.sh run 0-25        # partial case range
#   ./submit_experiments.sh test            # quick 5-case validation
#   ./submit_experiments.sh baseline        # t=0 per-condition priors
#   ./submit_experiments.sh agg             # aggregate saved experiments for $MODEL
#   MODEL=qwen7b GPUS=2 ./submit_experiments.sh
#   LOCAL=1 ./submit_experiments.sh test    # run directly on this GPU node (no SLURM)

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

mkdir -p "${BAYES_DIR}/bayesbench/medical_triage/logs"
cd "${BAYES_DIR}"

ACTION=${1:-run}
CASES_RANGE=${2:-0-100}             # e.g. "0-25" for a partial run

POP_INFO=zero_shot
PROFILES=(accurate hypochondriac minimizer cyberchondriac)

submit_job() {
    local job_name=$1
    local cmd=$2
    if [[ "$LOCAL" == "1" ]]; then
        echo "=== [local] ${job_name} ==="
        ( cd "${BAYES_DIR}" && eval "$cmd" ) 2>&1 | tee "bayesbench/medical_triage/logs/${job_name}.out"
        return
    fi
    sbatch --partition="$PARTITION" \
           --nodes=1 \
           --gpus-per-node="$GPUS" \
           --cpus-per-gpu=10 \
           --time="$TIME" \
           --job-name="$job_name" \
           --output="bayesbench/medical_triage/logs/${job_name}.out" \
           --error="bayesbench/medical_triage/logs/${job_name}.err" \
           --wrap="bash -c 'source ${CONDA_PATH}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && cd ${BAYES_DIR} && $cmd'"
    echo "Submitted: $job_name"
}

run_sweep() {
    echo "=== Medical triage sweep: ${MODEL} (${GPUS} GPU) — 4 profiles, cases ${CASES_RANGE} ==="
    for profile in "${PROFILES[@]}"; do
        job_name="triage_${MODEL}_${POP_INFO}_${profile}"
        cmd="python -m bayesbench.medical_triage.runner --model ${MODEL} --pop-info ${POP_INFO} --profile ${profile} --cases ${CASES_RANGE} --runs 5"
        submit_job "$job_name" "$cmd"
    done
}

run_test() {
    echo "=== Quick test (5 cases, 1 run) ==="
    cmd="python -m bayesbench.medical_triage.runner --model ${MODEL} --pop-info zero_shot --profile hypochondriac --cases 0-5 --runs 1"
    submit_job "triage_test_${MODEL}" "$cmd"
}

run_baseline() {
    echo "=== Baseline (t=0 priors, k=5 per condition) ==="
    cmd="python -m bayesbench.medical_triage.baselines --model ${MODEL} --pop-info all --k 5"
    submit_job "triage_baseline_${MODEL}" "$cmd"
}

run_agg() {
    echo "=== Aggregate ${MODEL} ==="
    cmd="python -m bayesbench.medical_triage.aggregate --experiments-dir bayesbench/medical_triage/experiments --model ${MODEL} --output bayesbench/medical_triage/experiments/${MODEL}_analysis.json"
    submit_job "triage_agg_${MODEL}" "$cmd"
}

case $ACTION in
    run)      run_sweep ;;
    test)     run_test ;;
    baseline) run_baseline ;;
    agg|aggregate) run_agg ;;
    *)
        echo "Usage: [MODEL=<key>] ./submit_experiments.sh [run|test|baseline|agg] [cases_range]"
        exit 1
        ;;
esac
