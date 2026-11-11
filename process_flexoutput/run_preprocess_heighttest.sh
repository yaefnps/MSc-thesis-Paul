#!/bin/bash
#SBATCH --job-name=preprocess_flex_heighttest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=10GB
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=a11925151@unet.unvie.ac.at

source ~/activate_flexpart_post.sh
RUN='2024/oper_01'
HEIGHT=2000

/usr/bin/time -v python3 -u preprocess_height_test_batched.py $RUN $HEIGHT
