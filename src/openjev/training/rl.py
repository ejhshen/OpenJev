"""Released REINFORCE-A actor: every primary and replay context uses RL + KL."""
import dataclasses
import torch
import torch.distributed as dist
from openjev.artifacts import load_decision_artifact
from openjev.data.schema import DecisionExample
from openjev.integrations.verl.engine import DecisionTrainingEngine
from openjev.rlcd.tape import FrozenTapeEnvironment
from openjev.rlcd.behavior import sample_actions
from openjev.rlcd.stage2 import reinforce_a_primary

class RLEngine(DecisionTrainingEngine):
 def __init__(self,recipe,device,stage,tape,reference_cache=None):
  super().__init__(recipe,device);self.stage=stage;self.policy_updates=0
  self.config=dataclasses.replace(self.config,serializer={**self.config.serializer,'max_branch_length':recipe.max_branch_length,'max_decision_tokens':recipe.max_decision_tokens});self.model.openjev_config=self.config
  self.action_generator=torch.Generator(device=device).manual_seed(recipe.seed+200003*self.rank)
  self.environment=FrozenTapeEnvironment(tape) if stage=='rl' else None
  self.reference=None;self.reference_cache=None
  if stage=='rl' and reference_cache:
   from openjev.rlcd.reference_cache import ReferenceCache,reference_contract
   self.reference_cache=ReferenceCache(reference_cache,reference_contract(recipe.model_artifact,self.token_cache.fingerprint,recipe.kernel_backend))
  if stage=='rl' and not reference_cache:
   self.reference,_,_,_=load_decision_artifact(recipe.model_artifact,device=device,dtype='bfloat16');self.reference.eval();self.reference.requires_grad_(False);self.reference.branch_microbatch_size=None
 def collate_cpu(self,examples):
  if isinstance(examples[0],DecisionExample):return super().collate_cpu(examples)
  batch=self.collator.collate_contexts(examples)
  assert 'targets' not in batch
  if batch['input_ids'].numel()>self.recipe.max_padded_tokens_per_gpu:raise ValueError('RL decision exceeds padded token budget')
  return batch
 def state_dict(self):return {'version':1,'policy_updates':self.policy_updates,'action_generator':self.action_generator.get_state().tolist(),'environment':self.environment.state_dict() if self.environment else None}
 def load_state_dict(self,s):
  if not s or s['version']!=1:raise ValueError('missing Stage 2 rank state')
  self.policy_updates=s['policy_updates'];self.action_generator.set_state(torch.tensor(s['action_generator'],dtype=torch.uint8))
  if self.environment:self.environment.load_state_dict(s['environment'])
 def validate_items(self,items):
  if self.reference_cache:
   for role,row in items:
    if role=='primary':self.reference_cache.get(row.decision)
 def mixed_step(self,items):
  assert self.stage=='rl' and all(role=='primary' and not isinstance(row,DecisionExample) and row.sample_weight==1. for role,row in items)
  stats=torch.zeros(3,device=self.device,dtype=torch.float64);before=self.environment.query_count
  def objective(out,batch,group):
   assert 'targets' not in batch
   if self.reference_cache:
    ref=torch.full_like(out.log_probs,-torch.inf,device='cpu')
    for i,(_,row,_) in enumerate(group):
     v=self.reference_cache.get(row.decision);ref[i,:len(v)]=torch.tensor(v)
    ref=ref.to(self.device,non_blocking=True)
   else:
    with torch.no_grad():ref=self.reference(batch,temperature=1.).log_probs.detach()
   sampled=sample_actions(out.probabilities,16,epsilon=.1,option_mask=out.option_mask,generator=self.action_generator)
   feedback=self.environment.feedback_batch([row.id for _,row,_ in group],sampled.actions.tolist())
   c=torch.tensor(feedback,device=self.device,dtype=out.probabilities.dtype)
   loss,_=reinforce_a_primary(out.probabilities,out.log_probs,sampled.actions,c,sampled.behavior_action_probs,ref,out.option_mask,kl_coefficient=.01)
   numerator=loss*len(group);stats[0]+=numerator.detach().double();stats[1]+=len(group);stats[2]+=c.mean(-1).double().sum()
   return numerator
  result=self.run_update(items,objective);self.policy_updates+=1;dist.all_reduce(stats)
  queries=torch.tensor(self.environment.query_count-before,device=self.device);dist.all_reduce(queries)
  result.pop('nll');result.update(rl_surrogate=float(stats[0]/stats[1]),feedback_correct_rate=float(stats[2]/stats[1]),feedback_queries=int(queries),rl_decisions=int(stats[1]),supervised_replay_decisions=0,surrogate_loss_is_population_objective=False)
  return result
