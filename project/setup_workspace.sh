#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Repairs the Udacity Cloud Workspace environment for this project.
#
# Two things are wrong with the stock lab image:
#
#   1. torch is 2.8.0+cpu, so the T4 that nvidia-smi reports is unusable.
#   2. transformers is 4.46.3, which predates Gemma 3 (added in 4.50.0).
#      google/gemma-3-270m-it cannot be loaded at all -- it fails with an
#      unrecognised-architecture error, no matter what hyperparameters you set.
#
# Run once per fresh workspace session:   bash setup_workspace.sh
# Lab sessions are ephemeral, so expect to re-run it after the box is recycled.
# ---------------------------------------------------------------------------
set -uo pipefail

say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

say "Before"
python -c "import torch, transformers; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| transformers', transformers.__version__)" 2>/dev/null || true
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"
df -h / | tail -1

# A CUDA torch wheel is ~2.5 GB unpacked. Bail early rather than half-installing.
AVAIL_KB=$(df -k / | tail -1 | awk '{print $4}')
if [ "$AVAIL_KB" -lt 6000000 ]; then
  echo "WARNING: under ~6 GB free. The CUDA torch wheel may not fit."
fi

# The lab runs as labsuser against a root-owned /usr/local site-packages, so
# fall back to a user-local install if the first attempt is denied.
pip_install() {
  python -m pip install --quiet "$@" || {
    echo "  (retrying with --user)"
    python -m pip install --quiet --user "$@"
  }
}

say "Upgrading pip"
pip_install --upgrade pip

say "Installing CUDA-enabled torch (T4 = sm_75, cu126 wheels support it)"
python -m pip install --quiet --upgrade --index-url https://download.pytorch.org/whl/cu126 \
  torch torchvision || {
  echo "  cu126 failed, falling back to cu121"
  python -m pip install --quiet --upgrade --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision
}

say "Upgrading the HF stack to versions that know about Gemma 3"
pip_install --upgrade \
  "transformers>=4.56,<6" \
  "trl>=0.23" \
  "peft>=0.15" \
  "accelerate>=1.2" \
  "datasets>=3,<6" \
  "huggingface_hub>=0.34"

say "After"
python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "| cuda build", torch.version.cuda,
      "| cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0),
          "| bf16", torch.cuda.is_bf16_supported())
print("transformers", transformers.__version__)
for m in ("trl", "peft", "datasets"):
    try:
        print(m, __import__(m).__version__)
    except Exception as e:
        print(m, "MISSING", e)
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES as C
gemmas = sorted(k for k in C if "gemma" in k)
print("gemma architectures registered:", gemmas)
print("gemma3 supported:", any("gemma3" in g for g in gemmas))
PY

say "Next"
cat <<'EOF'
  huggingface-cli login          # Gemma is a gated repo; accept the license first
  python env_check.py            # should now be all green except Ollama
  python starter_sft.py --smoke  # ~2 min pipeline check
  python starter_sft.py          # the real run
EOF
