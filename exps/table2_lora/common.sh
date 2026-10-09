# Shared defaults
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${PYTHON:-$REPO/.venv/bin/python}
OUT_ROOT=${OUT_ROOT:-$REPO/checkpoints/table2_lora}
CACHE_DIR=${CACHE_DIR:-$REPO/checkpoints/hf_cache}
MODEL=${MODEL:-roberta-base}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}

export TRANSFORMERS_CACHE=$CACHE_DIR
export HF_DATASETS_CACHE=$CACHE_DIR
export HF_METRICS_CACHE=$CACHE_DIR
export TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
