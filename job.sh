#!/usr/local/bin/bash
# Total experiments = num_envs(28) * num_seeds(20) * num_algos(2) = 1120
# Parallelism unit: 1 node = 4 GPUs (ntasks-per-node=4, 1 gpu per task)
# Each GPU runs 2 experiments sequentially -> 4 * 2 = 8 experiments per array job
# 1120 / 8 = 140 array jobs needed.
#
# Cluster MaxArraySize is 1000, so the array index itself must stay small.
# Instead of stepping the Slurm index by 8 (which would reach 1112, over the
# limit), the array index below is just a "chunk number" 0..139, and
# run_array.sh multiplies it by 8 internally to get the real task_id range.
 

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --partition=normal
#SBATCH --job-name=offlinerlkit
#SBATCH --time=12:00:00
#SBATCH --array=0-139


srun --environment=offlinerlkit /workspace/OfflineRLKit/run_array.sh