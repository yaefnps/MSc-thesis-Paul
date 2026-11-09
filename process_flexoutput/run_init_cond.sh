#!/bin/bash
#SBATCH --job-name=init_conds
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20GB
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=a11925151@unet.unvie.ac.at

source ~/activate_flexpart_post.sh
RUN='oper/2024/oper_01'

/usr/bin/time -v python3 -u initial_conds_oper_massw.py $RUN