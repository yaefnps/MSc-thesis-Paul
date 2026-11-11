#!/bin/bash
#SBATCH --job-name=preprocess_flex
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=a11925151@unet.unvie.ac.at

source ~/activate_flexpart_post.sh
YEAR='1991'

/usr/bin/time -v python3 -u preprocess_full_norad_batched.py $YEAR
