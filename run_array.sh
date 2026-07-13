#!/usr/bin/env bash
set -e

# --------------------------------------------------
# 0. Sanity checks (must happen before using the vars)
# --------------------------------------------------
if [[ -z "${SLURM_ARRAY_TASK_ID}" || -z "${SLURM_PROCID}" ]]; then
    echo "Usage: this script expects SLURM_ARRAY_TASK_ID and SLURM_PROCID to be set"
    exit 1
fi

# --------------------------------------------------
# 1. Working directory
# --------------------------------------------------
cd /workspaces/OfflineRL-Kit

# --------------------------------------------------
# 2. Define environments, algorithms and seeds
# --------------------------------------------------
envs=(
    "walker2d_custom-v2#10"
    "walker2d_custom-v2#50"
    "walker2d_custom-v2#100"
    "walker2d_custom-v2#500"
    "walker2d_custom-v2#1000"
    "walker2d_custom-v2#2000"
    "walker2d_custom-v2#5000"
    "hopper_custom-v2#10"
    "hopper_custom-v2#50"
    "hopper_custom-v2#100"
    "hopper_custom-v2#500"
    "hopper_custom-v2#1000"
    "hopper_custom-v2#2000"
    "hopper_custom-v2#5000"
    "halfcheetah_custom-v2#10"
    "halfcheetah_custom-v2#50"
    "halfcheetah_custom-v2#100"
    "halfcheetah_custom-v2#500"
    "halfcheetah_custom-v2#1000"
    "halfcheetah_custom-v2#2000"
    "halfcheetah_custom-v2#5000"
    "invertedpendulum_custom-v2#10"
    "invertedpendulum_custom-v2#50"
    "invertedpendulum_custom-v2#100"
    "invertedpendulum_custom-v2#500"
    "invertedpendulum_custom-v2#1000"
    "invertedpendulum_custom-v2#2000"
    "invertedpendulum_custom-v2#5000"
)

algorithms=(
    "bc"
    "cql"
)

seeds=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)

num_envs=${#envs[@]}          # 28
num_seeds=${#seeds[@]}        # 20
num_algos=${#algorithms[@]}   # 2
total_jobs=$(( num_envs * num_seeds * num_algos ))  # 1120

# --------------------------------------------------
# 3. Each task = 1 GPU. Each GPU runs 2 experiments
#    (sequentially). 4 tasks/node * 2 experiments = 8
#    experiments per array job.
#
#    The cluster's MaxArraySize is 1000, so job.sh uses a plain
#    --array=0-139 (one "chunk" per array job) instead of stepping by 8.
#    Here we expand that chunk number back into the real task_id range.
# --------------------------------------------------
gpus_per_node=4
experiments_per_gpu=2
experiments_per_array_job=$(( gpus_per_node * experiments_per_gpu ))  # 8

chunk_id=${SLURM_ARRAY_TASK_ID}
task_id_base=$(( chunk_id * experiments_per_array_job ))

for i in 0 1; do
    task_id=$(( task_id_base + SLURM_PROCID * experiments_per_gpu + i ))

    # Guard in case of any future mismatch between array bounds and total_jobs
    if (( task_id >= total_jobs )); then
        echo "task_id=${task_id} >= total_jobs=${total_jobs}, skipping."
        continue
    fi

    seeds_algos=$(( num_seeds * num_algos ))
    env_idx=$(( task_id / seeds_algos ))
    rem=$(( task_id % seeds_algos ))
    seed_idx=$(( rem / num_algos ))
    algo_idx=$(( rem % num_algos ))

    export ENV_NAME=${envs[$env_idx]}_${seeds[$seed_idx]}
    export SEED=0
    export WANDB_MODE=offline

    if [[ "${algorithms[$algo_idx]}" == "bc" ]]; then
        export ALGORITHM_SCRIPT="run_bc.py"
    elif [[ "${algorithms[$algo_idx]}" == "cql" ]]; then
        export ALGORITHM_SCRIPT="run_cql.py"
    else
        echo "Invalid algorithm selected."
        exit 1
    fi

    echo "[task_id=${task_id}] chunk=${chunk_id} PROCID=${SLURM_PROCID} slot=${i} -> ENV_NAME=${ENV_NAME} ALGO=${ALGORITHM_SCRIPT}"

    # --------------------------------------------------
    # 4. Run pipeline
    # --------------------------------------------------
    python run_example/${ALGORITHM_SCRIPT} --task ${ENV_NAME} --seed ${SEED}
done