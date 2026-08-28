#!/bin/bash
set -euo pipefail

NNCPU_DIR="/mnt/c/Users/Upris/Documents/nncpu"
WORK_DIR="/home/user/nncpu_campaign"
CHAMPSIM_DIR="$WORK_DIR/ChampSim"
TRACE_DIR="$WORK_DIR/traces"
RESULTS_DIR="$WORK_DIR/results_full_20"
COMBINED="$WORK_DIR/dpc3_all_traces.txt"
LOG="$WORK_DIR/campaign_direct.log"
JOBS=$(( $(nproc) - 1 ))
[ "$JOBS" -lt 1 ] && JOBS=1

exec > >(tee -a "$LOG") 2>&1

echo "=== Direct Campaign Launch ==="
echo "Started: $(date -Iseconds)"
echo "JOBS: $JOBS"

cat "$NNCPU_DIR/champsim/dpc3_trace_set.txt" "$NNCPU_DIR/champsim/dpc3_holdout_trace_set.txt" > "$COMBINED"

echo ">>> Traces present: $(ls -1 "$TRACE_DIR"/*.champsimtrace.xz | wc -l)"
echo ">>> Launching campaign.py..."

python3 "$NNCPU_DIR/champsim/campaign.py" \
  --champsim "$CHAMPSIM_DIR" \
  --trace-dir "$TRACE_DIR" \
  --trace-list "$COMBINED" \
  --output "$RESULTS_DIR" \
  --warmup 50000000 --simulation 200000000 \
  --jobs "$JOBS"

echo ""
echo "=== CAMPAIGN COMPLETE: $(date -Iseconds) ==="

WINDOWS_RESULTS="$NNCPU_DIR/results/champsim_full_20"
mkdir -p "$WINDOWS_RESULTS"
cp -r "$RESULTS_DIR"/* "$WINDOWS_RESULTS/"
echo ">>> Results copied to $WINDOWS_RESULTS"
