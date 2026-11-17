#!/bin/bash
#SBATCH --job-name=regrid_rad
#SBATCH --output=logs/regrid_rad_%A.out
#SBATCH --time=24:00:00
#SBATCH --ntasks=4
#SBATCH --mem=400GB

YEAR=2024
MONTH=8

/usr/bin/time -v python3 -u regrid_rad_g.py $YEAR $MONTH