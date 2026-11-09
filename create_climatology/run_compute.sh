#!/bin/bash
# Don't call sbatch on this directly; use submit_range.sh, which sets the array size
# and the date range (RANGE_START / RANGE_END / DAYS_PER_JOB) for you.
#SBATCH --job-name=mse_clim
#SBATCH --cpus-per-task=2         
#SBATCH --mem=16GB                   
#SBATCH --time=08:00:00            
#SBATCH --output=logs/compute_%A_%a.out
#SBATCH --error=logs/compute_%A_%a.err


: "${RANGE_START:?RANGE_START not set - use submit_range.sh}"
: "${RANGE_END:?RANGE_END not set - use submit_range.sh}"
DAYS_PER_JOB=${DAYS_PER_JOB:-4}
WORKERS=$SLURM_CPUS_PER_TASK

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# This job's slice of the requested range (MM-DD strings compare correctly)
DAYS=$(python - <<PY
import mse_climatology as m
d = [x.strftime('%m-%d') for x in m.calendar_days()
     if "${RANGE_START}" <= x.strftime('%m-%d') <= "${RANGE_END}"]
i = ${SLURM_ARRAY_TASK_ID} * ${DAYS_PER_JOB}
print(*d[i:i + ${DAYS_PER_JOB}])
PY
)

echo "Range ${RANGE_START}..${RANGE_END}, job ${SLURM_ARRAY_TASK_ID}: days = ${DAYS}  workers = ${WORKERS}"
[ -z "${DAYS}" ] && { echo "No days for this task, exiting."; exit 0; }

python -u create_mse_clim.py compute --workers ${WORKERS} --days ${DAYS}
