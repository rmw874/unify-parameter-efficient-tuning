#!/usr/bin/env bash
# glued up
#   bash exps/table2_lora/prefetch.sh <sst2|mnli>
set -euo pipefail
TASK=${1:?usage: prefetch.sh <sst2|mnli>}
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

"$PYTHON" - "$TASK" "$MODEL" <<'PY'
import sys
from datasets import load_dataset, load_metric
from transformers import AutoConfig, AutoTokenizer
from transformers.file_utils import WEIGHTS_NAME, cached_path, hf_bucket_url
task, model = sys.argv[1:3]
load_dataset("glue", task)
load_metric("glue", task)
AutoConfig.from_pretrained(model)
AutoTokenizer.from_pretrained(model)
cached_path(hf_bucket_url(model, WEIGHTS_NAME))  # weights only; the fork's model classes need the LoRA config
print(f"prefetched glue/{task} and {model} into {__import__('os').environ['TRANSFORMERS_CACHE']}")
PY
