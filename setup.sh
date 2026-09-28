#!/usr/bin/env bash
# Create the project virtualenv and install dependencies.
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d .venv ]; then
    python3 -m venv .venv
fi

.venv/bin/pip install --upgrade pip

# torch/torchvision: the default PyPI wheels are the CUDA build. On a machine with no
# NVIDIA GPU that still works (it just runs on CPU), but it drags in ~3 GB of unused
# CUDA libraries, so install the slim CPU wheels there instead.
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA GPU detected -> CUDA build of torch"
    .venv/bin/pip install torch torchvision
else
    echo "No NVIDIA GPU detected -> CPU-only build of torch"
    .venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

.venv/bin/pip install -r requirements.txt

echo
echo "Done. Activate with: source .venv/bin/activate"
.venv/bin/python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
