# Training OpenJev

The released training code retains the final model's SFT and REINFORCE-Analysis implementations. Training updates the whole backbone and decision head. It samples decisions, not generated token sequences.

## Environment

Start with the official [verl SGLang Docker environment](https://verl.readthedocs.io/en/latest/start/install.html#install-from-docker-image). The following example uses the official `verlai/verl` image:

```bash
docker pull verlai/verl:sgl055.latest
docker run --rm -it --gpus all --shm-size=10g \
  -v "$PWD:/workspace/openjev" -w /workspace/openjev \
  verlai/verl:sgl055.latest bash
```

OpenJev's tested dependency versions are Python 3.12, PyTorch 2.11.0+cu129, Transformers 5.12.1 and fla-core 0.5.2. Use a compatible PyTorch/CUDA stack in the official environment and install the OpenJev dependencies below. The launch scripts default to eight GPUs.

The FSDP2 wrapping, gradient clipping, checkpoint manager and sequence balancing are imported from verl. The validated source is [verl-agent](https://github.com/langfengQ/verl-agent) at commit `20bd331bdbc9026a5668e11362178e10ab7400c8`. Those imported utility files are unchanged from that upstream revision. For a new checkout:

```bash
git clone https://github.com/langfengQ/verl-agent.git third_party/verl
git -C third_party/verl checkout 20bd331bdbc9026a5668e11362178e10ab7400c8
export PYTHONPATH="$PWD/src:$PWD/third_party/verl:${PYTHONPATH:-}"
python -m pip install "transformers==5.12.1" "fla-core==0.5.2"
python -m pip install -e ".[train]"
bash training/check_imports.sh
```

The import check reports the loaded modules and dependency versions and verifies the training helpers and FLA kernel. The verl checkout remains an external dependency. Inference does not depend on verl.

## Load the starting model

Training loads a complete OpenJev model artifact, including the backbone, decision head, tokenizer and configuration. To fine-tune the released model:

```bash
hf download shenjunhao/OpenJev-4B --local-dir work/base-model
python -m openjev.cli validate-artifact work/base-model
```

Set `recipe.model_artifact` in `training/sft.json` to use another complete OpenJev checkpoint. The two-stage pipeline starts SFT from that artifact and starts RL from the resulting SFT model.

## Prepare training data

[OpenJevData-140k](https://huggingface.co/datasets/shenjunhao/OpenJevData-140k) supplies the compact `SFT` and `RL` splits. Each row contains state, question, options, task/category, optional score values, and a hard or soft answer distribution. Conversion preserves option names and criteria, and resolves a specific Hub revision before downloading.

```bash
python -m openjev.training.prepare_data \
  --dataset shenjunhao/OpenJevData-140k \
  --mandatory-replay training/mandatory-replay.json \
  --output work/data
```

The converter writes:

- `sft.jsonl` and `sft-manifest.json`: complete targets for SFT.
- `replay-selection.json`: a fixed, seeded 25% subset of SFT, stratified by broad category and hard/soft target type. Mandatory contexts enter this subset first.
- `rl-contexts.jsonl` and `rl-manifest.json`: the RL split plus selected SFT contexts, with no answer fields exposed to the actor.
- `outcomes.jsonl`: the feedback environment's independently sampled hidden outcomes, reproducible from the seed.
- `PREPARED.json`: data revision, counts and content hashes.

The **25% is a fraction of the SFT dataset**, not a fraction of each RL batch. All selected SFT contexts receive the same REINFORCE-A and reference-KL update as the original RL contexts; there is no supervised replay loss. The required replay list uses hashes of public input content. For a different dataset, supply its own mandatory list or omit that argument.

The public release has **100,345 SFT** and **46,393 RL** rows. With 25% replay it selects **25,086 SFT** rows and trains RL on **71,479 contexts**. At global batch 1,024, these sizes yield **98 SFT steps and 70 RL steps**, calculated by the runner.

## Run SFT and RL

The default configs use relative paths under `work/`. From the repository root:

```bash
NPROC=8 bash training/run_two_stage.sh
```

The fixed sequence is token cache → SFT → SFT artifact export → RL token cache → frozen SFT reference cache → RL → final artifact export. The exported model is `work/openjev-model`, loadable by the same inference example as the published weights.

Individual stages can also be launched explicitly:

```bash
python -m openjev.training.cache tokens --config training/sft.json
NPROC=8 bash training/run_sft.sh

# Export after SFT finishes; the completion file contains the final checkpoint.
SFT_CHECKPOINT="$(python -c 'import json; print(json.load(open("work/sft/TRAINING_COMPLETE.json"))["checkpoint"])')"
NPROC=8 bash training/run_sft.sh --export-checkpoint "$SFT_CHECKPOINT" --export-output work/sft-model

python -m openjev.training.cache tokens --config training/rl.json
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m openjev.training.cache reference --config training/rl.json
NPROC=8 bash training/run_rl.sh
```

| Setting | SFT | REINFORCE-A |
| --- | --- | --- |
| Global decision batch | 1,024 | 1,024 |
| Maximum decisions per microbatch | 32 | 32 |
| Expanded padded-token target per microbatch | 65,536 | 65,536 |
| Backbone / head learning rate | 4e-6 / 4e-5 | 7.5e-7 / 7.5e-6 |
| Warmup steps | 10 | 5 |
| Schedule after warmup | Constant | Constant |
| Samples per decision | — | 16, with replacement |
| Uniform exploration | — | 10% |
| Analytic/outcome mixture coefficient | — | 0.5 |
| Frozen-reference KL coefficient | — | 0.01 |

The microbatch target groups whole decisions by expanded length and candidate count. A single complete decision can exceed the packing target; the serializer still enforces its separate branch and logical token limits. SFT makes one unique shuffled pass, with zero-loss padding only to align the final batch across ranks. RL fixes each context's rank ownership and pads the final local batch with a few repeated contexts. This keeps outcome-tape cursors consistent across resume.

## Validation, checkpointing and resume

To evaluate during training, set `recipe.eval_manifest` to your held-out manifest and `recipe.eval_split` to its split name in both configs. The manifest uses the same supervised JSONL schema as `sft-manifest.json`. By default, `eval_every` and `checkpoint_every` are 10; validation also runs at the end. Without an evaluation manifest the runner trains and saves checkpoints without silently creating a new evaluation split. MMDM and JevBench remain external evaluations.

Metrics are appended to `work/sft/metrics.jsonl` and `work/rl/metrics.jsonl`; checkpoints are under `checkpoints/step-XXXXXXXX/`. Resume with the same dataset, batch, world size and recipe:

```bash
NPROC=8 bash training/run_sft.sh --resume work/sft/checkpoints/step-00000010
NPROC=8 bash training/run_rl.sh --resume work/rl/checkpoints/step-00000010
```

Model, optimizer, scheduler, RNG state, sampler position, sampled-action generator and per-context outcome cursors are restored. Token and reference caches reuse existing entries and validate their model/input fingerprints. `--stop-after` saves an early checkpoint for a controlled pause; it does not redefine the full dataset budget. The all-in-one shell is a fresh-run sequence; use individual resume commands after an interruption.

## Execution optimizations

- **Complete-decision microbatches:** length-aware packing reduces padding; all ranks agree on the number of forward/backward microsteps.
- **Token balancing:** SFT distributes expanded branch work across ranks using verl's sequence balancing helper.
- **FSDP2 accumulation:** retain gathered parameters between microsteps and defer gradient synchronization until the final microstep. Decoder blocks and the FP32 head have explicit precision policies; backbone compute is BF16 with FP32 reductions.
- **Activation checkpointing and aligned branches:** checkpoint decoder activations and round physical branch batches to multiples of eight.
- **Verified kernels:** use upstream FLA's gated delta rule at version 0.5.2 while retaining the Transformers normalization and convolution paths.
- **Offline caches:** tokenize inputs once; precompute frozen SFT reference log-probabilities once, so RL does not perform another reference forward on every update.

The model and optimizer use the same mathematical objectives as the final training run. This repository adds portable dataset conversion, sample-count-derived schedules and launch scripts around those optimized components.
