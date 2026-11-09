#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=radfluxes
#SBATCH --output=logs/radiative_fluxes_%A.out
#SBATCH --ntasks=1
#SBATCH --mem=5GB

source ~/activate_flexpart_post.sh
year_str=2023
month_str=7

python3 download_mars_rad_fluxes_12h.py $year_str $month_str
