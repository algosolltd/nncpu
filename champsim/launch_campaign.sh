#!/bin/bash
set -euo pipefail
LOG="/home/user/nncpu_campaign/campaign.log"
SCRIPT="/mnt/c/Users/Upris/Documents/nncpu/champsim/run_full_campaign.sh"
echo "Launching campaign at $(date)" | tee "$LOG"
bash "$SCRIPT" >> "$LOG" 2>&1
echo "Campaign finished at $(date)" >> "$LOG"
echo "EXIT_CODE=$?" >> "$LOG"
