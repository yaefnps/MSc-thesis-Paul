#!/bin/bash
#SBATCH --job-name=init_conds
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=a11925151@unet.unvie.ac.at

source ~/activate_flexpart_post.sh
RUN='oper/1991/'

/usr/bin/time -v python3 -u initial_conds_oper.py $RUN