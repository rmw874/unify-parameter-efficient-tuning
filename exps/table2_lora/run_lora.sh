#!/usr/bin/env bash
# uno lora run por favor. this is what run_all calls.
#
#   bash exps/table2_lora/run_lora.sh <sst2|mnli> <seed>
#
# hyperparams follow Table 8 of the paper (lr 1e-4, batch 32, 10 epochs,
# weight decay 0.1, 6% warmup, max grad norm 1, max length 512) and the LoRA
# configuration of the README / exps/run_glue.sh: rank 16 on W_q and W_v,
# alpha 32, dropout 0.1, which is 0.47% of RoBERTa-base ("LoRA (0.5%)").
# every setting can be overridden through the environment, e.g.
#   WEIGHT_DECAY=0 bash exps/table2_lora/run_lora.sh sst2 42
#   LORA_ALPHA=16 bash exps/table2_lora/run_lora.sh sst2 42       # scaling 1 instead of 2
#   MAX_STEPS=200 bash exps/table2_lora/run_lora.sh mnli 42      # timing run
#   RESUME=1 bash exps/table2_lora/run_lora.sh mnli 42           # continue from last checkpoint
set -euo pipefail

TASK=${1:?usage: run_lora.sh <sst2|mnli> <seed>}
SEED=${2:?usage: run_lora.sh <sst2|mnli> <seed>}
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

LR=${LR:-1e-4}
BSZ=${BSZ:-32}
NUM_EPOCHS=${NUM_EPOCHS:-10}
WARMUP_RATIO=${WARMUP_RATIO:-0.06}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-512}
UNFREEZE=${UNFREEZE:-ef_}              # "ef_,classifier" also trains the classification head
MAX_STEPS=${MAX_STEPS:--1}             # >0 stops after that many steps (timing runs)
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-}   # smoke tests only
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-}     # empty = the full dev set
RESUME=${RESUME:-0}
EXTRA_ARGS=${EXTRA_ARGS:-}             # appended verbatim to the run_glue.py command line
SAVE=${SAVE:-$OUT_ROOT/$RUN_NAME/seed$SEED}

extra=()
[ -n "$MAX_TRAIN_SAMPLES" ] && extra+=(--max_train_samples "$MAX_TRAIN_SAMPLES")
[ -n "$MAX_EVAL_SAMPLES" ] && extra+=(--max_eval_samples "$MAX_EVAL_SAMPLES")
[ "$RESUME" = 1 ] || extra+=(--overwrite_output_dir)
# shellcheck disable=SC2206
extra+=($EXTRA_ARGS)

mkdir -p "$SAVE"
echo "task=$TASK seed=$SEED weight_decay=$WEIGHT_DECAY -> $SAVE"
cd "$REPO"
"$PYTHON" -u examples/pytorch/text-classification/run_glue.py \
    --model_name_or_path "$MODEL" \
    --task_name "$TASK" \
    --do_train --do_eval \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --per_device_train_batch_size "$BSZ" \
    --per_device_eval_batch_size "$BSZ" \
    --gradient_accumulation_steps 1 \
    --max_tokens_per_batch 0 \
    --learning_rate "$LR" \
    --lr_scheduler_type polynomial \
    --warmup_ratio "$WARMUP_RATIO" \
    --warmup_steps 0 \
    --weight_decay "$WEIGHT_DECAY" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --adam_beta1 0.9 --adam_beta2 0.98 --adam_epsilon 1e-6 \
    --num_train_epochs "$NUM_EPOCHS" \
    --max_steps "$MAX_STEPS" \
    --attn_mode lora --attn_option none --attn_composition add --attn_bn "$LORA_R" \
    --ffn_mode none --ffn_option none --ffn_adapter_layernorm_option none \
    --ffn_adapter_init_option bert --ffn_adapter_scalar 1 --ffn_bn "$LORA_R" \
    --mid_dim 800 \
    --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" --lora_init lora \
    --unfreeze_params "$UNFREEZE" \
    --seed "$SEED" \
    --fp16 \
    --evaluation_strategy epoch --save_strategy epoch --save_total_limit 2 \
    --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
    --logging_steps 50 \
    --report_to none \
    --disable_tqdm True \
    --ddp_find_unused_parameter False \
    --output_dir "$SAVE" \
    "${extra[@]}" \
    2>&1 | tee "$SAVE/log.txt"
