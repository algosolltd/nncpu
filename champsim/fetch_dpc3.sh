#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "usage: $0 TRACE_DIRECTORY [PARALLEL_DOWNLOADS] [TRACE_LIST]" >&2
  exit 2
fi

trace_directory=$1
parallel_downloads=${2:-4}
script_directory=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
trace_list=${3:-$script_directory/dpc3_trace_set.txt}
base_url=https://dpc3.compas.cs.stonybrook.edu/champsim-traces/speccpu

if ! [[ $parallel_downloads =~ ^[1-9][0-9]*$ ]]; then
  echo "PARALLEL_DOWNLOADS must be a positive integer" >&2
  exit 2
fi
if [[ ! -f $trace_list ]]; then
  echo "TRACE_LIST does not exist: $trace_list" >&2
  exit 2
fi

mkdir -p "$trace_directory"
trace_names() {
  sed 's/#.*//' "$trace_list" | awk 'NF {$1=$1; print}'
}

trace_names \
  | xargs -P "$parallel_downloads" -I '{}' bash -c '
      trace_name=$1
      trace_directory=$2
      base_url=$3
      final_path="$trace_directory/$trace_name"
      partial_path="$final_path.partial"
      if [[ -s "$final_path" ]]; then
        exit 0
      fi
      curl -fL --retry 3 --continue-at - \
        --output "$partial_path" "$base_url/$trace_name"
      mv "$partial_path" "$final_path"
    ' _ '{}' "$trace_directory" "$base_url"

while IFS= read -r trace_name; do
  xz --list "$trace_directory/$trace_name" >/dev/null
  if [[ ${NNCPU_FULL_TRACE_VERIFY:-0} == 1 ]]; then
    xz -t "$trace_directory/$trace_name"
  fi
done < <(trace_names)

echo "downloaded and validated the fixed DPC-3 trace set in $trace_directory"
