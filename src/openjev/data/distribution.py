"""Bridge distribution-only Stage 2 rows into the existing decision model.

Option identity is its frozen vector position. All supervision, including
one-hot labels, enters the same distribution CE and never a second trainer.
"""
from collections.abc import Mapping
import math

def normalize_distribution_row(row: Mapping, *, actor=False):
    if not isinstance(row, Mapping):
        raise ValueError('row must be an object')
    if 'metadata' not in row:
        return row
    if actor and any(k in row for k in ('target_distribution','target','posterior','oracle','q_exact','outcomes')):
        raise ValueError('actor context cannot contain supervision or outcomes')
    metadata = dict(row['metadata'])
    if not actor:
        metadata['target_contract'] = 'distribution-only-v1'
        if metadata.get('probability_origin') == 'program_exact':
            metadata['posterior_kind'] = 'known-posterior'
    options = [{'id': f'o{i}', 'name': o['name'], 'criteria': o.get('criteria')}
               for i, o in enumerate(row['options'])]
    result = {'id': row['id'], 'group_id': row['group_id'], 'state': row['state'],
              'question': row['question'], 'options': options,
              'source': {'dataset': metadata.get('seed_source', metadata.get('source', 'stage2')),
                         'split': metadata.get('split')},
              'provenance': metadata, 'primitive': metadata.get('primitive','choice'),
              'sample_weight': row.get('sample_weight',1.0)}
    if metadata.get('score_values') is not None:
        result['score_values'] = metadata['score_values']
    if not actor:
        q = row['target_distribution']
        if len(q) != len(options) or any(isinstance(p,bool) or not isinstance(p,(int,float))
                or not math.isfinite(p) or p < 0 for p in q):
            raise ValueError('one finite nonnegative target probability per option is required')
        if not math.isclose(math.fsum(q), 1, abs_tol=1e-6, rel_tol=0):
            raise ValueError('target_distribution must sum to one')
        result['target'] = {'kind': 'soft', 'probabilities': {o['id']:p for o,p in zip(options,q)}}
    elif any(k in metadata for k in ('q_exact','target_distribution','gold_index','option_semantic_indices','q_bin','probability_origin')):
        raise ValueError('actor metadata contains oracle information')
    return result
