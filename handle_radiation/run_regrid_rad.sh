#!/bin/bash
#SBATCH --job-name=regrid_rad
#SBATCH --output=logs/regrid_rad_%A.out
#SBATCH --time=12:00:00
#SBATCH --ntasks=4
#SBATCH --mem=300GB

YEAR=2024
MONTH=5

/usr/bin/time -v python3 -u regrid_rad_g.py $YEAR $MONTH