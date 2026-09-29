#!/usr/bin/env bash
set -euo pipefail
# Prepare work/init and work/data first.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
NPROC="${NPROC:-8}"
python -m openjev.training.cache tokens --config training/sft.json
python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" -m openjev.training.run --config training/sft.json
SFT_CHECKPOINT="$(python -c 'import json; print(json.load(open("work/sft/TRAINING_COMPLETE.json"))["checkpoint"])')"
python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" -m openjev.training.run --config training/sft.json --export-checkpoint "$SFT_CHECKPOINT" --export-output work/sft-model
python -m openjev.training.cache tokens --config training/rl.json
python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" -m openjev.training.cache reference --config training/rl.json
python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" -m openjev.training.run --config training/rl.json
RL_CHECKPOINT="$(python -c 'import json; print(json.load(open("work/rl/TRAINING_COMPLETE.json"))["checkpoint"])')"
python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" -m openjev.training.run --config training/rl.json --export-checkpoint "$RL_CHECKPOINT" --export-output work/openjev-model
