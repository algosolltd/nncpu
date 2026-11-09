#!/bin/bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

NNCPU_DIR="/mnt/c/Users/Upris/Documents/nncpu"
WORK_DIR="$HOME/nncpu_campaign"
CHAMPSIM_DIR="$WORK_DIR/ChampSim"
TRACE_DIR="$WORK_DIR/traces"
RESULTS_DIR="$WORK_DIR/results_full_20"
CHAMPSIM_COMMIT="e6530c7293a4d93857f634447554281a3e582516"
JOBS=$(( $(nproc) - 1 ))
[ "$JOBS" -lt 1 ] && JOBS=1

echo "=== NNCPU ChampSim Full Campaign ==="
echo "NNCPU_DIR:    $NNCPU_DIR"
echo "WORK_DIR:     $WORK_DIR"
echo "CHAMPSIM_DIR: $CHAMPSIM_DIR"
echo "TRACE_DIR:    $TRACE_DIR"
echo "RESULTS_DIR:  $RESULTS_DIR"
echo "JOBS:         $JOBS"
echo "Started:      $(date -Iseconds)"
echo ""

mkdir -p "$WORK_DIR"

# Step 1: Clone ChampSim
if [ ! -d "$CHAMPSIM_DIR" ]; then
  echo ">>> [$(date +%H:%M:%S)] Cloning ChampSim..."
  git clone https://github.com/ChampSim/ChampSim.git "$CHAMPSIM_DIR"
  cd "$CHAMPSIM_DIR"
  git checkout "$CHAMPSIM_COMMIT"
  git submodule update --init vcpkg
else
  echo ">>> ChampSim already cloned"
  cd "$CHAMPSIM_DIR"
fi

# Step 2: Bootstrap vcpkg
if [ ! -x "$CHAMPSIM_DIR/vcpkg/vcpkg" ]; then
  echo ">>> [$(date +%H:%M:%S)] Bootstrapping vcpkg..."
  cd "$CHAMPSIM_DIR"
  ./vcpkg/bootstrap-vcpkg.sh
  ./vcpkg/vcpkg install
else
  echo ">>> vcpkg already bootstrapped"
fi

# Step 3: Build ChampSim with nncpu external module
cd "$CHAMPSIM_DIR"
if [ ! -f "$CHAMPSIM_DIR/bin/champsim" ]; then
  echo ">>> [$(date +%H:%M:%S)] Building ChampSim..."
  mkdir -p .csconfig/external
  make -j"$JOBS" EXTERNAL_MODULE_DIR="$NNCPU_DIR/champsim"
else
  echo ">>> ChampSim binary already built"
fi

echo ">>> Verifying binary..."
"$CHAMPSIM_DIR/bin/champsim" --help 2>&1 | head -3 || true

# Step 4: Download ALL traces (dev + holdout) — skip if already present
mkdir -p "$TRACE_DIR"
TRACE_COUNT=$(ls -1 "$TRACE_DIR"/*.champsimtrace.xz 2>/dev/null | wc -l)
if [ "$TRACE_COUNT" -ge 20 ]; then
  echo ">>> All $TRACE_COUNT traces already present, skipping download"
else
  echo ">>> [$(date +%H:%M:%S)] Downloading development traces (11)..."
  bash "$NNCPU_DIR/champsim/fetch_dpc3.sh" "$TRACE_DIR" 4 "$NNCPU_DIR/champsim/dpc3_trace_set.txt"
  echo ">>> [$(date +%H:%M:%S)] Downloading holdout traces (9)..."
  bash "$NNCPU_DIR/champsim/fetch_dpc3.sh" "$TRACE_DIR" 4 "$NNCPU_DIR/champsim/dpc3_holdout_trace_set.txt"
fi

echo ">>> Traces present:"
ls -1 "$TRACE_DIR"/*.champsimtrace.xz 2>/dev/null | wc -l
echo "trace files"

# Step 5: Create combined trace list
COMBINED="$WORK_DIR/dpc3_all_traces.txt"
cat "$NNCPU_DIR/champsim/dpc3_trace_set.txt" "$NNCPU_DIR/champsim/dpc3_holdout_trace_set.txt" > "$COMBINED"

# Step 6: Run the full campaign
echo ""
echo "=== [$(date +%H:%M:%S)] LAUNCHING FULL CAMPAIGN (20 traces x 4 configs = 80 runs) ==="
echo "This will take several hours."
echo ""

python3 "$NNCPU_DIR/champsim/campaign.py" \
  --champsim "$CHAMPSIM_DIR" \
  --trace-dir "$TRACE_DIR" \
  --trace-list "$COMBINED" \
  --output "$RESULTS_DIR" \
  --warmup 50000000 --simulation 200000000 \
  --jobs "$JOBS"

echo ""
echo "=== [$(date +%H:%M:%S)] CAMPAIGN COMPLETE ==="

# Step 7: Copy results back to Windows
WINDOWS_RESULTS="$NNCPU_DIR/results/champsim_full_20"
mkdir -p "$WINDOWS_RESULTS"
cp -r "$RESULTS_DIR"/* "$WINDOWS_RESULTS/"
echo ">>> Results copied to $WINDOWS_RESULTS"
ls -la "$WINDOWS_RESULTS"
