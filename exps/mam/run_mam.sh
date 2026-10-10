#!/usr/bin/env bash
# Run one MAM experiment on GLUE SST-2 with the original run_glue.py
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
PYTHON=${PYTHON:-python3}
VERIFY_SETUP=${VERIFY_SETUP:-1}
case "$VERIFY_SETUP" in
  1) MODEL=${MODEL:-$ROOT/checkpoints/table2_mam/pretrained_roberta_base} ;;
  0) MODEL=${MODEL:-roberta-base} ;;
  *) echo "VERIFY_SETUP must be 0 or 1 (got: $VERIFY_SETUP)" >&2; exit 2 ;;
esac
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/checkpoints/table2_mam}

if (( $# != 2 )); then
  echo "Usage: bash exps/table2_mam/run_mam.sh {base|scale1|capacity_ffn24} SEED" >&2
  exit 2
fi

CONDITION=$1
SEED=$2
if [[ ! $SEED =~ ^[0-9]+$ ]]; then
  echo "Seed must be a nonnegative integer: $SEED" >&2
  exit 2
fi

case "$CONDITION" in
  base)           ATTN_BN=16; FFN_BN=16; FFN_SCALE=2 ;;
  scale1)         ATTN_BN=16; FFN_BN=16; FFN_SCALE=1 ;;
  capacity_ffn24) ATTN_BN=8;  FFN_BN=24; FFN_SCALE=2 ;;
  *) echo "Unknown condition: $CONDITION" >&2; exit 2 ;;
esac

OUT_DIR="$OUTPUT_ROOT/$CONDITION/seed$SEED"
MODEL_DIR="$OUT_DIR/model"
SCRIPT="$ROOT/examples/pytorch/text-classification/run_glue.py"

# keeps these arguments aligned with the original MAM SST-2 runs
ARGS=(
  --model_name_or_path "$MODEL"
  --task_name sst2
  --do_train true
  --do_eval true
  --max_seq_length 512
  --pad_to_max_length true
  --per_device_train_batch_size 8
  --gradient_accumulation_steps 4
  --per_device_eval_batch_size 8
  --max_tokens_per_batch 0
  --seed "$SEED"
  --num_train_epochs 10
  --max_steps -1
  --learning_rate 1e-4
  --lr_scheduler_type polynomial
  --warmup_ratio 0.06
  --warmup_steps 0
  --adam_beta1 0.9
  --adam_beta2 0.98
  --adam_epsilon 1e-6
  --weight_decay 0.1
  --max_grad_norm 1.0
  --label_smoothing_factor 0.0
  --fp16 true
  --attn_mode prefix
  --attn_option concat
  --attn_composition add
  --attn_bn "$ATTN_BN"
  --mid_dim 800
  --prefix_dropout 0.0
  --ffn_mode adapter
  --ffn_option parallel
  --ffn_bn "$FFN_BN"
  --ffn_adapter_layernorm_option none
  --ffn_adapter_init_option lora
  --ffn_adapter_scalar "$FFN_SCALE"
  --unfreeze_params ef_
  --lora_alpha 0.0
  --lora_dropout 0.0
  --lora_init lora
  --evaluation_strategy epoch
  --save_strategy epoch
  --load_best_model_at_end true
  --metric_for_best_model accuracy
  --greater_is_better true
  --save_total_limit 2
  --logging_steps 50
  --disable_tqdm true
  --report_to none
  --output_dir "$MODEL_DIR"
)

if [[ ${DRY_RUN:-0} == 1 ]]; then
  printf '%q ' "$PYTHON" -u "$SCRIPT" "${ARGS[@]}"
  echo
  exit 0
fi

if [[ ! -f "$SCRIPT" ]]; then
  echo "Missing authors' script: $SCRIPT" >&2
  exit 1
fi

if [[ -f "$OUT_DIR/finished.ok" && -f "$MODEL_DIR/eval_results.json" ]]; then
  echo "SKIP completed: $CONDITION seed $SEED"
  exit 0
fi
if [[ -d "$OUT_DIR" && -n "$(ls -A "$OUT_DIR")" ]]; then
  echo "Incomplete or unverified output: $OUT_DIR" >&2
  echo "Move or remove it before restarting from scratch." >&2
  exit 1
fi

# verify the setup is identical
# a strict verification is enabled by default. can be skipped with VERIFY_SETUP=0.
# in strict mode, you first need to run 'verify_setup.py prepare' once before the first experiment
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$VERIFY_SETUP" == 1 ]]; then
  MODEL="$MODEL" "$PYTHON" "$HERE/verify_setup.py" check
fi

mkdir -p "$OUT_DIR"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_SCRIPTS_VERSION=1.11.0
export TOKENIZERS_PARALLELISM=false
if [[ "$VERIFY_SETUP" == 1 ]]; then
  # use only the pretrained model and SST-2 content verified above
  export HF_DATASETS_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
else
  # allow standard Hugging Face model and dataset downloads
  unset HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE HF_HUB_OFFLINE
fi

printf '%q ' "$PYTHON" -u "$SCRIPT" "${ARGS[@]}" > "$OUT_DIR/command.txt"
echo >> "$OUT_DIR/command.txt"
echo "START $CONDITION seed=$SEED | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
"$PYTHON" -u "$SCRIPT" "${ARGS[@]}" 2>&1 | tee "$OUT_DIR/log.txt"

# marks complete after successful training and evaluation
for file in train_results.json eval_results.json; do
  if [[ ! -f "$MODEL_DIR/$file" ]]; then
    echo "Missing $MODEL_DIR/$file; not marking run complete." >&2
    exit 1
  fi
done
touch "$OUT_DIR/finished.ok"
echo "DONE $CONDITION seed=$SEED | metrics: $MODEL_DIR/eval_results.json"
