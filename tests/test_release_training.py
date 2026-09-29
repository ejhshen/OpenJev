import itertools
import json
import torch
import pytest
from openjev.training.prepare_data import convert, prepare, select_replay
from openjev.training.sampling import FullPassStream
from openjev.data.hf_loader import load_manifest, load_context_manifest
from openjev.rlcd.reinforce import analytic_probability_term_loss
from openjev.rlcd.tape import FrozenTapeEnvironment


def rows(n, split='SFT'):
    return [convert({'state':f'case {i}', 'question':'choose',
        'options':[{'id':'a','name':'A','criteria':None},{'id':'b','name':'B','criteria':'alternative'}],
        'category':'logic' if i%2 else 'routing', 'task':'choice','score_values':None,
        'answer':{'kind':'soft' if i%3==0 else 'hard','probabilities':[.3,.7] if i%3==0 else [1.,0.]}}, split, i) for i in range(n)]


def test_fixed_replay_has_only_rl_actor_inputs(tmp_path):
    sft=rows(16);rl=rows(8,'RL');required={sft[0].group_id}
    report=prepare(sft,rl,tmp_path/'prepared',mandatory=required,episodes=4)
    assert report['sft_replayed']==4 and report['rl_total']==12
    contexts=load_context_manifest(tmp_path/'prepared/rl-manifest.json',split='train')
    assert sft[0].id in {r.id for r in contexts}
    assert all('target' not in r.to_dict() for r in contexts)
    assert len(load_manifest(tmp_path/'prepared/sft-manifest.json',split='train'))==16
    tape=FrozenTapeEnvironment(tmp_path/'prepared/outcomes.jsonl')
    feedback=tape.feedback_batch([contexts[0].id],[[0,1,0,1]])[0]
    assert feedback[0]==feedback[2] and feedback[1]==feedback[3] and feedback[0]+feedback[1]==1
    assert select_replay(sft,.25,42,required)==select_replay(sft,.25,42,required)


@pytest.mark.parametrize('stage',['sft','rl'])
def test_full_coverage_fixed_rl_ownership_and_resume(stage):
    data=rows(19);kwargs=dict(stage=stage,world=2,batch=8,seed=42,contract={})
    s=FullPassStream(data,**kwargs);first=s.next_global();saved=s.state_dict()
    restored=FullPassStream(data,**kwargs);restored.load_state_dict(saved)
    second=s.next_global();assert second==restored.next_global()
    last=s.next_global();batches=[first,second,last]
    assert {i for b in batches for role,i in b if role!='sft_pad'}==set(range(19))
    if stage=='sft':
        assert sum(role=='sft_pad' for b in batches for role,i in b)==1
        assert sum(role=='sft' for b in batches for role,i in b)==19
    else:
        owners={}
        for batch in batches:
            for j,(_,i) in enumerate(batch):
                rank=j//4
                assert owners.setdefault(i,rank)==rank
    with pytest.raises(StopIteration):s.next_global()


def test_reinforce_a_exact_expected_gradient():
    # Enumerate both possible outcomes and all pairs of sampled actions.
    logits=torch.tensor([.4,-.2],dtype=torch.float64,requires_grad=True)
    p=logits.softmax(-1);logs=logits.log_softmax(-1);q=torch.tensor([.3,.7],dtype=torch.float64)
    mu=(.9*p+.1/2).detach();expected=torch.zeros_like(logits)
    for y in range(2):
        for a in itertools.product(range(2),repeat=2):
            actions=torch.tensor([a]);feedback=torch.tensor([[float(x==y) for x in a]],dtype=torch.float64)
            loss=analytic_probability_term_loss(p[None],logs[None],actions,feedback,mu[actions],alpha=.5).loss
            grad=torch.autograd.grad(loss,logits,retain_graph=True)[0]
            expected+=q[y]*mu[a[0]]*mu[a[1]]*grad
    objective=.5*(-q*logs).sum()+.25*(p-q).square().sum()
    torch.testing.assert_close(expected,torch.autograd.grad(objective,logits)[0],atol=1e-7,rtol=1e-6)
