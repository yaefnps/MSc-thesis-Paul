#!/bin/bash
#SBATCH --job-name=regrid_rad
#SBATCH --output=logs/regrid_rad_%A.out
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --mem=400GB

/usr/bin/time -v python3 -u regrid_radiation_newnaming.py