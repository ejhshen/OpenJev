"""Full-pass schedules from the final recipe, generalized to dataset size."""
import hashlib
import json
import math
import random


class FullPassStream:
    def __init__(self, rows, *, stage, world, batch, seed, contract):
        if stage not in ('sft', 'rl') or len(rows) < world or batch % world:
            raise ValueError('need at least one example per rank and a world-divisible batch')
        self.stage, self.world, self.batch = stage, world, batch
        self.order = sorted(range(len(rows)), key=lambda i: rows[i].id)
        random.Random(seed).shuffle(self.order)
        self.local = [self.order[r::world] for r in range(world)]
        self.steps = math.ceil(len(rows)/batch)
        self.step = self.total_seen = 0
        self.fingerprint = hashlib.sha256(json.dumps([stage, world, batch, seed, contract, [r.id for r in rows]], sort_keys=True).encode()).hexdigest()

    def next_global(self):
        if self.step >= self.steps:
            raise StopIteration('complete full pass')
        if self.stage == 'sft':
            indices = self.order[self.total_seen:self.total_seen+self.batch]
            result = [('sft', i) for i in indices]
            self.total_seen += len(indices)
            result += [('sft_pad', self.order[i]) for i in range((-len(result)) % self.world)]
        else:
            n = self.batch//self.world
            result = [('primary', order[(self.step*n+j) % len(order)]) for order in self.local for j in range(n)]
            self.total_seen += len(result)
        self.step += 1
        return result

    def state_dict(self):
        return {'fingerprint': self.fingerprint, 'step': self.step, 'total_seen': self.total_seen}

    def load_state_dict(self, state):
        if state['fingerprint'] != self.fingerprint or not 0 <= state['step'] <= self.steps:
            raise ValueError('resume dataset or schedule mismatch')
        self.step, self.total_seen = state['step'], state['total_seen']
