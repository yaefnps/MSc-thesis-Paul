#!/bin/bash
#SBATCH --job-name=mse_final
#SBATCH --cpus-per-task=6          # = --workers (one task per hour of day, ~4 GB each)
#SBATCH --mem=400GB
#SBATCH --time=03:00:00
#SBATCH --output=logs/finalize_%j.out
#SBATCH --error=logs/finalize_%j.err
##SBATCH --partition=YOUR_PARTITION
##SBATCH --account=YOUR_ACCOUNT

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs


python -u create_mse_clim.py finalize --workers ${SLURM_CPUS_PER_TASK}
