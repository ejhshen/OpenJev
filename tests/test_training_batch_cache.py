import dataclasses,json
from pathlib import Path
import pytest,torch
from openjev.data.schema import DecisionExample
from openjev.models.decision.serializer import DecisionSerializer
from openjev.data.collator import pack_decisions,split_to_microsteps,collate_serialized
from openjev.data.token_cache import TokenCache
from openjev.rlcd.reference_cache import ReferenceCache
from openjev.rlcd.tape import FrozenTapeEnvironment
from openjev.training.supervised import decision_cross_entropy

class Tokenizer:
    bos_token_id=None
    def encode(self,text,add_special_tokens=False):return [ord(c) for c in text]
    def get_vocab(self):return {'dummy':0}

def rows():
    return [DecisionExample.from_dict({'id':str(i),'group_id':str(i),'state':'s'*(10+i*20),
       'question':'choose','options':[{'id':str(j),'name':'x'*j+'a'} for j in range(k)],
       'target':{'kind':'hard','option_id':str(k-1)},'sample_weight':w}) for i,(k,w) in enumerate([(2,1.),(5,2.),(3,.5),(4,0.)])]

def test_packing_preserves_complete_decisions_and_weights():
    data=rows();serialize=DecisionSerializer(Tokenizer())
    entries=[('primary' if i==0 else 'replay',r,serialize(r.decision)) for i,r in enumerate(data)]
    groups=pack_decisions(entries,3,1800)
    groups=split_to_microsteps(groups,4)
    assert sorted(x[1].id for group in groups for x in group)==['0','1','2','3']
    assert all(len({x[0]=='primary' for x in group})==1 for group in groups)
    z=torch.randn(4,5,requires_grad=True);mask=torch.arange(5)[None,:]<torch.tensor([2,5,3,4])[:,None]
    logs=z.masked_fill(~mask,-torch.inf).log_softmax(-1);target=torch.zeros_like(z)
    for i,r in enumerate(data):target[i,:len(r.decision.options)]=torch.tensor(r.target.vector(r.decision))
    weights=torch.tensor([r.sample_weight for r in data]);full=(decision_cross_entropy(logs,target,mask,reduction='none')*weights).sum()/weights.sum()
    parts=0.
    for group in groups:
        ix=[int(x[1].id) for x in group]
        parts+=(decision_cross_entropy(logs[ix],target[ix],mask[ix],reduction='none')*weights[ix]).sum()/weights.sum()
    torch.testing.assert_close(torch.autograd.grad(parts,z,retain_graph=True)[0],torch.autograd.grad(full,z)[0])

def test_token_cache_exact_resume_and_input_invalidation(tmp_path):
    serializer=DecisionSerializer(Tokenizer());data=rows();path=tmp_path/'tokens.sqlite'
    cache=TokenCache(path,serializer,writable=True);assert cache.populate(data[:2],workers=2)['new']==2
    assert cache.populate(data,workers=2)['new']==2
    reader=TokenCache(path,serializer)
    for r in data:assert reader(r.decision)==serializer(r.decision)
    changed=dataclasses.replace(data[0].decision,state='changed input with the same ID')
    with pytest.raises(KeyError):reader(changed)
    changed_recipe=TokenCache(path,DecisionSerializer(Tokenizer(),max_branch_length=1024))
    with pytest.raises(KeyError):changed_recipe(data[0].decision)

def test_reference_cache_is_bound_to_frozen_model_and_options(tmp_path):
    data=rows();path=tmp_path/'ref.sqlite';cache=ReferenceCache(path,{'model':'one'},writable=True)
    cache.put([(data[0].decision,[-.7,-.69])]);reader=ReferenceCache(path,{'model':'one'})
    assert reader.get(data[0].decision)==[-.7,-.69]
    with pytest.raises(KeyError):ReferenceCache(path,{'model':'two'}).get(data[0].decision)
    with pytest.raises(KeyError):reader.get(dataclasses.replace(data[0].decision,options=tuple(reversed(data[0].decision.options))))

def test_tape_batch_matches_sequential_feedback_and_resume(tmp_path):
    p=tmp_path/'tape.jsonl';p.write_text(json.dumps({'id':'a','num_options':2,'outcomes':[1,0,1]})+'\n')
    a=FrozenTapeEnvironment(p);b=FrozenTapeEnvironment(p);actions=[[0,1,1],[1,0,0]]
    actual=a.feedback_batch(['a','a'],actions);expected=[]
    for chosen in actions:
        e=b.start_episode('a');expected.append(b.feedback(e.episode_id,chosen));b.close_episode(e.episode_id)
    assert actual==expected;assert a.state_dict()==b.state_dict()
    restored=FrozenTapeEnvironment(p);restored.load_state_dict(a.state_dict())
    assert restored.feedback_batch(['a'],[[0,1]])==a.feedback_batch(['a'],[[0,1]])


def test_branch_shape_buckets_preserve_forward_and_parameter_gradients():
    import copy
    from openjev.models.decision.model import OpenJevDecisionModel
    from openjev.models.decision.head import DecisionHead
    from openjev.config import OpenJevConfig
    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__();self.emb=torch.nn.Embedding(128,12);self.linear=torch.nn.Linear(12,12)
        def get_hidden_size(self):return 12
        def forward_hidden(self,input_ids,attention_mask,position_ids=None):
            return self.linear(self.emb(input_ids).cumsum(1))
    torch.manual_seed(7)
    a=OpenJevDecisionModel(Adapter(),DecisionHead(12,set_dim=16,set_layers=1,set_heads=4,ffn_dim=32));a.openjev_config=OpenJevConfig(hidden_size=12)
    b=copy.deepcopy(a);b.branch_batch_multiple=8
    decisions=[DecisionExample.from_dict({'id':str(i),'group_id':str(i),'state':'abc','question':'choose',
       'options':[{'id':str(j),'name':'a'} for j in range(k)],'target':{'kind':'hard','option_id':'0'}}) for i,k in enumerate([8,9])]
    serializer=DecisionSerializer(Tokenizer());batch=collate_serialized([serializer(x.decision) for x in decisions],0)
    x=a(batch).probabilities;y=b(batch).probabilities
    torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-5)
    x[:,0].sum().backward();y[:,0].sum().backward()
    for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=2e-5)
    assert b.padded_branch_count(17)==20
