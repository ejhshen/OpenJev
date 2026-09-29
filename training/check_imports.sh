#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
python - "$PROJECT_ROOT" <<'PY'
import importlib, importlib.metadata, sys
from pathlib import Path
for name in ['openjev','torch','transformers','verl']:
    module=importlib.import_module(name)
    print(name, module.__file__)
    if name=='openjev':
        assert Path(module.__file__).resolve().is_relative_to(Path(sys.argv[1])/'src')
from openjev.integrations.verl.engine import DecisionTrainingEngine
from verl.utils.fsdp_utils import apply_fsdp2, fsdp2_clip_grad_norm_
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions
from openjev.models.qwen35.kernels import configure_upstream_fla
import torch
assert importlib.metadata.version("fla-core") == "0.5.2"
if torch.cuda.is_available():
    print(configure_upstream_fla())
else:
    print("FLA package verified; CUDA operator check deferred to GPU execution")
for name in ['torch','transformers','fla-core']:
    print(name, importlib.metadata.version(name))
print('Training imports OK')
PY
