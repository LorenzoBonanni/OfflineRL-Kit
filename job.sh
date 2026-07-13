#!/usr/local/bin/bash
# Total experiments = num_envs(7) * num_seeds (20) * num_algos (2) = 7 * 20 * 2 = 280
# Per array job: 20 srun tasks × 1 experiment each = 20 experiments
# Array step = 20  →  indices 0, 20, 40, ..., 260  (14 array jobs total)
 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-task=0
#SBATCH --mem-per-cpu=3G
#SBATCH --partition=normal
#SBATCH --job-name=offlinerlkit
#SBATCH --time=24:00:00
# 1120
#SBATCH --array=0-1000:20
 
srun --environment=offlinerl_kit /workspace/OfflineRLKit/run_bc_array.job