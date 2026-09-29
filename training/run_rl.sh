#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
exec python -m torch.distributed.run --standalone --nproc_per_node="${NPROC:-8}" -m openjev.training.run --config "${CONFIG:-training/rl.json}" "$@"
