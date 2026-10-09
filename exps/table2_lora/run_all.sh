#!/usr/bin/env bash
# 5 seeds in parallel yupppp!
#
#   bash exps/table2_lora/run_all.sh sst2
#   bash exps/table2_lora/run_all.sh mnli
#   SEEDS="42 2" WEIGHT_DECAY=0 bash exps/table2_lora/run_all.sh sst2
#   SEEDS="42 2 4" LORA_ALPHA=16 bash exps/table2_lora/run_all.sh sst2   # ablation, own directory
#   PER_GPU=5 bash exps/table2_lora/run_all.sh sst2    # override the per-GPU concurrency
#   WAIT=1 bash exps/table2_lora/run_all.sh sst2       # block until done, then summarise

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

# Default PER_GPU: 1 on MIG slices (a 23 GB slice fits one ~15 GB run), otherwise
# one run per 20 GB of GPU memory, so a full B200 takes all seeds at once.
if [ -z "${PER_GPU:-}" ]; then
    if nvidia-smi -L | grep -q "MIG"; then
        PER_GPU=1
    else
        mem_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1)
        PER_GPU=$(( mem_mib / 20000 ))
        [ "$PER_GPU" -ge 1 ] || PER_GPU=1
    fi
fi
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
    # Paths go in as arguments, never pasted into the command, so any folder name works.
    CUDA_VISIBLE_DEVICES=$gpu nohup bash -c \
        'for s in $3; do bash "$1/run_lora.sh" "$2" "$s"; done' _ "$HERE" "$TASK" "${slot_seeds[$slot]}" \
        > "$OUT_ROOT/$RUN_NAME/slot${slot}.out" 2>&1 &
    echo "pid=$! gpu=$gpu seeds: ${slot_seeds[$slot]}" | tee -a "$OUT_ROOT/$RUN_NAME/pids.txt"
done
echo
echo "Launched $i runs on ${#GPUS[@]} GPU(s), at most $PER_GPU per GPU at a time."
echo "Follow:    tail -f $OUT_ROOT/$RUN_NAME/seed*/log.txt"
echo "Status and results (rerun any time): $PYTHON $HERE/collect_results.py $OUT_ROOT/$RUN_NAME"

# WAIT=1: block until every run has finished, then print the summary. Needed when
# this script is the job itself (UCloud "run a script"), which ends when it exits.
if [ "${WAIT:-0}" = 1 ]; then
    echo "Waiting for the runs to finish..."
    wait
    echo "All runs finished at $(date)."
    "$PYTHON" "$HERE/collect_results.py" "$OUT_ROOT/$RUN_NAME"
fi
