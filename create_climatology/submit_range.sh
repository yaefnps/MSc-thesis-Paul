#!/bin/bash
# Usage: ./submit_range.sh START END [DAYS_PER_JOB=4] [MAX_RUNNING=15]
#   e.g. ./submit_range.sh 06-01 06-30
# START/END are inclusive, zero-padded MM-DD, within 02-19 .. 10-10 (default season).
# Run from the directory containing create_mse_clim.py.
set -e
START=$1; END=$2; PER=${3:-4}; MAXRUN=${4:-6}
[ -z "$START" ] || [ -z "$END" ] && { echo "Usage: $0 START END [DAYS_PER_JOB] [MAX_RUNNING]"; exit 1; }

NDAYS=$(python - <<PY
import create_mse_clim as m
print(sum("$START" <= x.strftime('%m-%d') <= "$END" for x in m.calendar_days()))
PY
)
[ "$NDAYS" -eq 0 ] && { echo "No valid days between $START and $END (format MM-DD, inside the season+buffer)."; exit 1; }
NJOBS=$(( (NDAYS + PER - 1) / PER ))

echo "$NDAYS days -> $NJOBS jobs ($PER days each, max $MAXRUN running)"
mkdir -p logs
sbatch --array=0-$((NJOBS-1))%${MAXRUN} \
       --export=ALL,RANGE_START=$START,RANGE_END=$END,DAYS_PER_JOB=$PER \
       run_compute.sh
