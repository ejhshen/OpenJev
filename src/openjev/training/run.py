"""Distributed SFT or REINFORCE-A with the optimized FSDP2 execution engine."""
import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
from openjev.artifacts import load_decision_artifact, save_decision_artifact, sha256_file, write_json
from openjev.data.hf_loader import load_manifest, load_context_manifest
from openjev.integrations.verl.engine import SFTRecipe, DecisionTrainingEngine
from openjev.integrations.verl.checkpoint import DecisionCheckpointManager, training_contract
from openjev.training.sampling import FullPassStream


def export(engine, recipe, destination, step, contract):
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
    full = get_model_state_dict(engine.model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    status = [None]
    if dist.get_rank() == 0:
        try:
            model, tokenizer, _, _ = load_decision_artifact(recipe.model_artifact, device='cpu', dtype='float32')
            model.load_state_dict(full, strict=True)
            model.adapter.backbone.to(dtype=torch.bfloat16); model.head.float()
            model.openjev_config = engine.config; model.eval()
            save_decision_artifact(model, tokenizer, engine.config, destination,
                                   stage='decision-rlcd' if contract['stage']=='rl' else 'decision-sft',
                                   provenance={'global_step':step,'training_contract':contract}, temperature=1.)
        except Exception as exc:
            status[0] = repr(exc)
    dist.broadcast_object_list(status, src=0)
    if status[0]:
        raise RuntimeError(status[0])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--resume')
    p.add_argument('--export-checkpoint');p.add_argument('--export-output')
    p.add_argument('--stop-after',type=int,help='Save and stop early; resume retains the full-pass schedule')
    a=p.parse_args(); config=json.loads(Path(a.config).read_text()); stage=config['stage']
    if stage not in ('sft','rl'):
        raise ValueError('stage must be sft or rl')
    recipe=SFTRecipe(**config['recipe'])
    rank,world,local=(int(os.environ[k]) for k in ('RANK','WORLD_SIZE','LOCAL_RANK'))
    torch.cuda.set_device(local);dist.init_process_group('nccl',timeout=timedelta(minutes=40))
    device=torch.device('cuda',local)
    try:
        loader=load_manifest if stage=='sft' else load_context_manifest
        rows=loader(recipe.train_manifest,split=recipe.train_split)
        recipe.global_decision_batch=recipe.global_decision_batch or world
        recipe.max_steps=math.ceil(len(rows)/recipe.global_decision_batch)
        if config.get('epochs',1)!=1:
            raise ValueError('this release recipe uses one full pass per stage')
        recipe.validate(world)
        dev=load_manifest(recipe.eval_manifest,split=recipe.eval_split) if recipe.eval_manifest else None
        random.seed(recipe.seed);torch.manual_seed(recipe.seed);torch.cuda.manual_seed_all(recipe.seed)
        contract=training_contract(recipe)|{'stage':stage,'rows':len(rows),'full_pass_steps':recipe.max_steps,
                   'balance_tokens':config.get('balance_tokens',False),'runner_sha256':sha256_file(__file__),
                   'sampler_sha256':sha256_file(Path(__file__).with_name('sampling.py'))}
        if dev:
            contract['eval_manifest_sha256']=sha256_file(recipe.eval_manifest)
        if stage=='rl':
            contract.update(tape_sha256=sha256_file(config['tape']),alpha=.5,epsilon=.1,group_size=16,kl_coefficient=.01)
            from openjev.training.rl import RLEngine
            engine=RLEngine(recipe,device,'rl',config['tape'],config.get('reference_cache'))
            if engine.reference_cache:
                contract['reference_cache_fingerprint']=engine.reference_cache.fingerprint
        else:
            engine=DecisionTrainingEngine(recipe,device)
            engine.config=dataclasses.replace(engine.config,serializer={**engine.config.serializer,
                'max_branch_length':recipe.max_branch_length,'max_decision_tokens':recipe.max_decision_tokens})
            engine.model.openjev_config=engine.config
        contract['token_cache_fingerprint']=getattr(engine.token_cache,'fingerprint',None)
        stream=FullPassStream(rows,stage=stage,world=world,batch=recipe.global_decision_batch,seed=recipe.seed,contract=contract)
        manager=DecisionCheckpointManager(engine,training_contract=contract)
        resume=a.export_checkpoint or a.resume
        step=manager.load(resume,stream) if resume else 0
        if resume and stage=='rl':
            engine.load_state_dict(manager.loaded_extra_state)
        output=Path(recipe.output_dir);output.mkdir(parents=True,exist_ok=True)
        def emit(event):
            if rank==0:
                line=json.dumps(event,allow_nan=False);print(line,flush=True)
                with (output/'metrics.jsonl').open('a') as f:f.write(line+'\n')
        if rank==0:
            write_json(output/'run-contract.json',{'recipe':dataclasses.asdict(recipe),'contract':contract})
        if a.export_checkpoint:
            if not a.export_output:raise ValueError('--export-output is required')
            export(engine,recipe,a.export_output,step,contract);emit({'event':'export','step':step,'output':a.export_output});return
        emit({'event':'start','stage':stage,'step':step,'max_steps':recipe.max_steps,'rows':len(rows),'world_size':world})
        while step<recipe.max_steps:
            entries=stream.next_global();per_rank=len(entries)//world
            local_entries=entries[rank*per_rank:(rank+1)*per_rank]
            if stage=='sft' and config.get('balance_tokens',False):
                from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions
                costs=[]
                for _,idx in entries:
                    encoded=(engine.token_cache or engine.serializer)(rows[idx].decision)
                    costs.append(len(encoded.branches)*max(len(b.input_ids) for b in encoded.branches))
                partitions=get_seqlen_balanced_partitions(costs,k_partitions=world,equal_size=True)
                local_entries=[entries[i] for i in partitions[rank]]
            items=[(role,dataclasses.replace(rows[i],sample_weight=0.) if role=='sft_pad' else rows[i]) for role,i in local_entries]
            result=engine.train_step([r for _,r in items]) if stage=='sft' else engine.mixed_step(items)
            step+=1;emit({'event':'train','stage':stage,'step':step,**result})
            final=step==recipe.max_steps;stop=a.stop_after is not None and step>=a.stop_after
            if final or stop or (recipe.checkpoint_every and step%recipe.checkpoint_every==0):
                checkpoint=manager.save(output/'checkpoints'/f'step-{step:08d}',step=step,sampler_state=stream.state_dict(),
                     extra_state=engine.state_dict() if stage=='rl' else None)
                emit({'event':'checkpoint','step':step,'path':str(checkpoint)})
            if dev and (final or (recipe.eval_every and step%recipe.eval_every==0)):
                result=engine.evaluate(dev,prediction_path=output/'eval'/f'step-{step:08d}-rank-{rank:04d}.jsonl')
                emit({'event':'eval','step':step,**result})
            if stop and not final:
                emit({'event':'stopped','step':step});return
        if rank==0:
            write_json(output/'TRAINING_COMPLETE.json',{'step':step,'unique_contexts':len(rows),'checkpoint':str(checkpoint),'stage':stage})
        emit({'event':'complete','stage':stage,'step':step})
    finally:
        dist.destroy_process_group()

if __name__=='__main__':
    main()
