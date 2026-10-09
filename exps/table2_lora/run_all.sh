#!/usr/bin/env bash
# 5 seeds in parallel yupppp!
#
#   bash exps/table2_lora/run_all.sh sst2
#   bash exps/table2_lora/run_all.sh mnli
#   SEEDS="42 2" WEIGHT_DECAY=0 bash exps/table2_lora/run_all.sh sst2
#   SEEDS="42 2 4" LORA_ALPHA=16 bash exps/table2_lora/run_all.sh sst2   # ablation, own directory
#   PER_GPU=5 bash exps/table2_lora/run_all.sh sst2    # one big GPU: all seeds at once

set -euo pipefail

TASK=${1:?usage: run_all.sh <sst2|mnli>}
SEEDS=${SEEDS:-"42 2 4 6 8"}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/common.sh"
export OUT_ROOT CACHE_DIR MODEL WEIGHT_DECAY LORA_R LORA_ALPHA LORA_DROPOUT RUN_NAME PYTHON

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
else
    # MIG slices on ucloud are separate CUDA devices
    # that must be addressed by UUID; a process can only see one of them.
    mapfile -t GPUS < <(nvidia-smi -L | grep -oE "MIG-[0-9a-fA-F-]+")
    if [ ${#GPUS[@]} -eq 0 ]; then
        mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
    fi
fi
[ ${#GPUS[@]} -gt 0 ] || { echo "no GPUs found" >&2; exit 1; }

# download the dataset, metric and model once, before the seeds race for them.
bash "$HERE/prefetch.sh" "$TASK"

# w PER_GPU=1 (default) a GPU never holds more than one run (a MIG slice fits exactly one)
PER_GPU=${PER_GPU:-1}
n_slots=$(( ${#GPUS[@]} * PER_GPU ))
declare -a slot_seeds
i=0
for seed in $SEEDS; do
    slot=$(( i % n_slots ))
    slot_seeds[$slot]+="$seed "
    i=$(( i + 1 ))
done

mkdir -p "$OUT_ROOT/$RUN_NAME"
: > "$OUT_ROOT/$RUN_NAME/pids.txt"
for slot in $(seq 0 $(( n_slots - 1 ))); do
    [ -n "${slot_seeds[$slot]:-}" ] || continue
    gpu=${GPUS[$(( slot % ${#GPUS[@]} ))]}
    CUDA_VISIBLE_DEVICES=$gpu nohup bash -c \
        "for s in ${slot_seeds[$slot]}; do bash '$HERE/run_lora.sh' '$TASK' \$s; done" \
        > "$OUT_ROOT/$RUN_NAME/slot${slot}.out" 2>&1 &
    echo "pid=$! gpu=$gpu seeds: ${slot_seeds[$slot]}" | tee -a "$OUT_ROOT/$RUN_NAME/pids.txt"
done
echo
echo "Launched $i runs on ${#GPUS[@]} GPU(s), at most $PER_GPU per GPU at a time."
echo "Follow:    tail -f $OUT_ROOT/$RUN_NAME/seed*/log.txt"
echo "Summarise: $PYTHON $HERE/collect_results.py $OUT_ROOT/$RUN_NAME"
