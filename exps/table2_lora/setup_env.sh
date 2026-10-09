#!/usr/bin/env bash
# Create the Python environment for the Table 2 LoRA runs.
#
#   bash exps/table2_lora/setup_env.sh                 # B200 / H100 / A100: CUDA 12.8 wheels
#   TORCH_CUDA=cu126 bash exps/table2_lora/setup_env.sh  # older GPUs (Pascal)
#
# Creates <repo>/.venv (override with VENV=...) with Python 3.9, torch 2.8,
# this transformers fork (editable) and the pinned requirements.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
VENV=${VENV:-$REPO/.venv}
TORCH_CUDA=${TORCH_CUDA:-cu128}
PYTHON_VERSION=3.9
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-$REPO/.uv-python}

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found, installing it to ~/.local/bin"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv python install "$PYTHON_VERSION"
uv venv --clear --python-preference only-managed --python "$PYTHON_VERSION" "$VENV"
uv pip install --python "$VENV/bin/python" \
    --index-url "https://download.pytorch.org/whl/$TORCH_CUDA" "torch==2.8.*"
uv pip install --python "$VENV/bin/python" \
    -r "$REPO/exps/table2_lora/requirements.txt" -e "$REPO"

"$VENV/bin/python" - <<'PY'
import torch, transformers, datasets
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available(),
      "| gpus:", torch.cuda.device_count())
print("transformers", transformers.__version__, "| datasets", datasets.__version__)
PY
echo "Environment ready: $VENV"
