"""Immutable pre-sampled outcomes owned only by the feedback environment."""
from dataclasses import dataclass
import hashlib,json
from pathlib import Path
from .environment import EpisodeHandle

class FrozenTapeEnvironment:
    def __init__(self, path):
        p=Path(path);self._fingerprint=hashlib.sha256(p.read_bytes()).hexdigest()
        self._records={}
        for line in p.read_text().splitlines():
            row=json.loads(line);key=row['id'];n=row['num_options'];values=row['outcomes']
            if key in self._records or not isinstance(n,int) or isinstance(n,bool) or n<2:
                raise ValueError('duplicate or invalid tape record')
            if not values or any(not isinstance(v,int) or isinstance(v,bool) or not 0<=v<n for v in values):
                raise ValueError('invalid frozen outcome')
            self._records[key]=(n,tuple(values))
        self._positions={key:0 for key in self._records};self._episodes={};self._queries=0;self._next=0
    @property
    def query_count(self):return self._queries
    def start_episode(self, context_id, *, episode_id=None):
        if episode_id is None:episode_id=f'tape-{self._next}';self._next+=1
        if episode_id in self._episodes:
            key,y=self._episodes[episode_id]
            if key!=context_id:raise ValueError('episode context changed')
            return EpisodeHandle(episode_id,context_id,self._records[key][0])
        n,values=self._records[context_id];pos=self._positions[context_id]
        if pos>=len(values):raise RuntimeError('outcome tape exhausted; never silently recycle')
        self._episodes[episode_id]=(context_id,values[pos]);self._positions[context_id]=pos+1
        return EpisodeHandle(episode_id,context_id,n)
    def feedback(self, episode_id, actions):
        key,y=self._episodes[episode_id];n=self._records[key][0];actions=tuple(actions)
        if not actions or any(not isinstance(a,int) or isinstance(a,bool) or not 0<=a<n for a in actions):raise ValueError('invalid action')
        self._queries+=len(actions);return tuple(float(a==y) for a in actions)
    def feedback_batch(self, context_ids, actions):
        """Batch transport, preserving the existing per-episode cursor and query semantics."""
        if len(context_ids) != len(actions):
            raise ValueError('feedback batch shape mismatch')
        result = []
        for key, selected in zip(context_ids, actions):
            episode = self.start_episode(key)
            result.append(self.feedback(episode.episode_id, selected))
            self.close_episode(episode.episode_id)
        return result
    def close_episode(self,episode_id):del self._episodes[episode_id]
    def state_dict(self):
        return {'version':1,'tape_sha256':self._fingerprint,'positions':dict(self._positions),'episodes':dict(self._episodes),'queries':self._queries,'next_episode':self._next}
    def load_state_dict(self,state):
        if state.get('version')!=1 or state.get('tape_sha256')!=self._fingerprint:raise ValueError('tape mismatch')
        positions=state['positions']
        if set(positions)!=set(self._records) or any(not isinstance(v,int) or isinstance(v,bool) or not 0<=v<=len(self._records[k][1]) for k,v in positions.items()):raise ValueError('invalid tape position')
        episodes={k:tuple(v) for k,v in state['episodes'].items()}
        for key,y in episodes.values():
            if key not in self._records or not isinstance(y,int) or not 0<=y<self._records[key][0]:raise ValueError('invalid episode')
        if any(not isinstance(state[k],int) or isinstance(state[k],bool) or state[k]<0 for k in ('queries','next_episode')):raise ValueError('invalid tape counters')
        self._positions=dict(positions);self._episodes=episodes;self._queries=state['queries'];self._next=state['next_episode']
