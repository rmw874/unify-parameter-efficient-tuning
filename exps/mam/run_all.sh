#!/usr/bin/env bash
# Launch 11 MAM SST-2 runs
# one experiment at a time per visible GPU
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
COMMAND=${1:-plan}
GPU_IDS=${GPU_IDS:-0}

if [[ $COMMAND != plan && $COMMAND != run ]]; then
  echo "Usage: GPU_IDS='0 1' bash exps/table2_mam/run_all.sh [plan|run]" >&2
  exit 2
fi

read -r -a GPUS <<< "$GPU_IDS"
if (( ${#GPUS[@]} == 0 )); then
  echo "GPU_IDS cannot be empty." >&2
  exit 2
fi

TASKS=(
  base:42 base:2 base:4 base:6 base:8
  scale1:42 scale1:2 scale1:4
  capacity_ffn24:42 capacity_ffn24:2 capacity_ffn24:4
)

# assigns experiments
# each GPU processes its queue sequentially
for i in "${!GPUS[@]}"; do
  printf 'GPU %s:' "${GPUS[$i]}"
  for j in "${!TASKS[@]}"; do
    if (( j % ${#GPUS[@]} == i )); then printf ' %s' "${TASKS[$j]}"; fi
  done
  echo
done
[[ $COMMAND == plan ]] && exit 0

pids=()
for i in "${!GPUS[@]}"; do
  (
    for j in "${!TASKS[@]}"; do
      if (( j % ${#GPUS[@]} != i )); then continue; fi
      task=${TASKS[$j]}
      CUDA_VISIBLE_DEVICES="${GPUS[$i]}" \
        bash "$HERE/run_mam.sh" "${task%%:*}" "${task##*:}"
    done
  ) &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if (( failed )); then
  echo "At least one GPU worker failed; inspect the run logs." >&2
  exit 1
fi
echo 'All 11 MAM runs completed (or were already complete).'
