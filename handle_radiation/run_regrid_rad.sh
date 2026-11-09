#!/bin/bash
#SBATCH --job-name=regrid_rad
#SBATCH --output=logs/regrid_rad_%A.out
#SBATCH --time=12:00:00
#SBATCH --ntasks=8
#SBATCH --mem=200GB

YEAR=2023
MONTH=6

/usr/bin/time -v python3 -u regrid_rad_g.py $YEAR $MONTH