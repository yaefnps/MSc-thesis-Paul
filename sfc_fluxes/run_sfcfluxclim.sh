#!/bin/bash
#SBATCH --job-name=sfcfluxclim
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=a11925151@unet.unvie.ac.at

source ~/activate_flexpart_post.sh

/usr/bin/time -v python3 -u create_sfcflux_clim.py 
