"""Resumable token and frozen SFT reference caches for the release pipeline."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from openjev.data.hf_loader import load_manifest, load_context_manifest
from openjev.data.token_cache import TokenCache, decision_key
from openjev.models.decision.serializer import DecisionSerializer


def serializer_for(recipe):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(Path(recipe['model_artifact'])/'tokenizer', local_files_only=True)
    config = json.loads((Path(recipe['model_artifact'])/'openjev_config.json').read_text())
    args = {k:v for k,v in config['serializer'].items() if k != 'version'}
    args.update(max_branch_length=recipe['max_branch_length'], max_decision_tokens=recipe['max_decision_tokens'])
    return tok, DecisionSerializer(tok, **args)


def prepare_tokens(config, workers):
    recipe = config['recipe']; _, serializer = serializer_for(recipe)
    cache = TokenCache(recipe['token_cache'], serializer, writable=True)
    loader = load_manifest if config['stage'] == 'sft' else load_context_manifest
    rows = loader(recipe['train_manifest'], split='train')
    if recipe.get('eval_manifest'):
        rows += load_manifest(recipe['eval_manifest'], split=recipe.get('eval_split', 'dev'))
    print(json.dumps(cache.populate(rows, workers=workers, progress=lambda x: print(json.dumps(x), flush=True))))
    cache.db.close()


def prepare_reference(config, device, rank, world):
    import torch
    import torch.distributed as dist
    from openjev.artifacts import load_decision_artifact
    from openjev.data.collator import collate_serialized, pack_decisions
    from openjev.rlcd.reference_cache import ReferenceCache, reference_contract
    recipe = config['recipe']; tok, serializer = serializer_for(recipe)
    tokens = TokenCache(recipe['token_cache'], serializer)
    cache = ReferenceCache(config['reference_cache'], reference_contract(recipe['model_artifact'], tokens.fingerprint, recipe['kernel_backend']), writable=True)
    rows = load_context_manifest(recipe['train_manifest'], split='train')
    local = [r for r in rows if int(hashlib.sha256(r.id.encode()).hexdigest()[:8],16) % world == rank]
    existing = {r[0] for r in cache.db.execute('SELECT key FROM predictions WHERE namespace=?', (cache.fingerprint,))}
    pending = [r for r in local if decision_key(r.decision) not in existing]
    model, _, _, _ = load_decision_artifact(recipe['model_artifact'], device=device, dtype='bfloat16')
    model.eval(); model.branch_microbatch_size = None
    with torch.inference_mode():
        for start in range(0, len(pending), 256):
            groups = pack_decisions([('primary', r, tokens(r.decision)) for r in pending[start:start+256]], recipe['microbatch_decisions'], recipe['max_padded_tokens_per_gpu'])
            pairs = []
            for group in groups:
                pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
                batch = collate_serialized([x[2] for x in group], pad)
                batch = {k:v.to(device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
                logs = model(batch, temperature=1.).log_probs.float().cpu().tolist()
                pairs.extend((r.decision, logs[i][:len(r.decision.options)]) for i, (_,r,_) in enumerate(group))
            cache.put(pairs)
            print(json.dumps({'rank': rank, 'cached': min(start+256,len(pending)), 'pending':len(pending)}), flush=True)
    del model; torch.cuda.empty_cache(); dist.barrier()
    # Validate coverage by key, not merely the number of cached rows.
    for r in local:
        cache.get(r.decision)
    dist.barrier(); cache.db.close(); tokens.db.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('kind', choices=['tokens','reference']); p.add_argument('--config', required=True)
    p.add_argument('--workers',type=int,default=16); a=p.parse_args()
    config=json.loads(Path(a.config).read_text())
    if a.kind=='tokens':
        prepare_tokens(config,a.workers); return
    import torch
    import torch.distributed as dist
    torch.cuda.set_device(int(os.environ['LOCAL_RANK'])); dist.init_process_group('nccl')
    try:
        prepare_reference(config,torch.device('cuda',int(os.environ['LOCAL_RANK'])),dist.get_rank(),dist.get_world_size())
    finally:
        dist.destroy_process_group()

if __name__=='__main__':
    main()
