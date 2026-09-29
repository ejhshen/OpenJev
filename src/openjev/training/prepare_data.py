"""Convert the compact public dataset to SFT, actor-only RL, and outcome files."""
import argparse
import collections
import hashlib
import json
import math
import random
from pathlib import Path
from openjev.artifacts import sha256_file, write_json
from openjev.data.schema import DecisionExample, DecisionContext


def content_key(row):
    fields = {k: row.get(k) for k in ('state', 'question', 'options', 'task', 'score_values')}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def convert(row, split, index):
    options = row['options']
    probabilities = row['answer']['probabilities']
    if len(probabilities) != len(options):
        raise ValueError('answer probabilities must align with options')
    if any(not math.isfinite(p) or p < 0 for p in probabilities) or not math.isclose(sum(probabilities), 1, abs_tol=1e-6):
        raise ValueError('invalid target distribution')
    kind = row['answer']['kind']
    if kind == 'hard':
        if sum(p == 1 for p in probabilities) != 1 or any(p not in (0, 1) for p in probabilities):
            raise ValueError('hard answer must be one-hot')
        target = {'kind': kind, 'option_id': options[probabilities.index(1)]['id']}
    elif kind == 'soft':
        target = {'kind': kind, 'probabilities': dict(zip([o['id'] for o in options], probabilities))}
    else:
        raise ValueError('answer.kind must be hard or soft')
    key = content_key(row)
    data = dict(id=f'{split}-{index:07d}-{key[:16]}', group_id=key, state=row['state'],
                question=row['question'], options=options, primitive=row.get('task', 'choice'),
                target=target, source={'category': row['category']}, score_values=row.get('score_values'))
    return DecisionExample.from_dict(data)


def select_replay(rows, fraction, seed, mandatory=()):
    """A fixed subset of SFT rows; category/kind quotas, not a batch mixture."""
    if not 0 <= fraction <= 1:
        raise ValueError('replay fraction must lie in [0, 1]')
    quota = int(len(rows) * fraction)
    mandatory = set(mandatory)
    available = {r.group_id for r in rows}
    if not mandatory <= available:
        raise ValueError('mandatory replay examples missing from SFT')
    selected = {i for i, r in enumerate(rows) if r.group_id in mandatory}
    if len(selected) > quota:
        raise ValueError('mandatory replay exceeds the requested SFT fraction')
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r.source.get('category'), r.target.kind)].append(i)
    rng = random.Random(seed)
    # Choose the most underrepresented stratum at each draw, after mandatory inclusion.
    pools = {}; counts = {}
    for key, indices in sorted(groups.items()):
        pools[key] = [i for i in indices if i not in selected]
        rng.shuffle(pools[key])
        counts[key] = sum(i in selected for i in indices)
    while len(selected) < quota:
        key = max((k for k in pools if pools[k]), key=lambda k: len(groups[k])*fraction-counts[k])
        selected.add(pools[key].pop()); counts[key] += 1
    return sorted(selected)


def write_rows(path, rows):
    with path.open('w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n')


def prepare(sft, rl, output, *, fraction=.25, seed=20260923, mandatory=(), episodes=256, provenance=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'prepared data already exists: {output}; reuse it or choose a new output')
    if not sft or not rl or episodes < 1:
        raise ValueError('both training splits and a positive episode budget are required')
    selected = select_replay(sft, fraction, seed, mandatory)
    combined = list(rl) + [sft[i] for i in selected]
    if len({x.id for x in combined}) != len(combined):
        raise ValueError('duplicate training record IDs')
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output/'sft.jsonl', [r.to_dict() for r in sft])
    contexts = []
    for r in combined:
        d = r.to_dict(); d.pop('target')
        contexts.append(DecisionContext.from_dict(d).to_dict())
    write_rows(output/'rl-contexts.jsonl', contexts)
    def outcomes():
        for r in combined:
            rng = random.Random(f'{seed}:{r.id}')
            yield {'id': r.id, 'num_options': len(r.decision.options),
                   'outcomes': rng.choices(range(len(r.decision.options)), weights=r.target.vector(r.decision), k=episodes)}
    write_rows(output/'outcomes.jsonl', outcomes())
    for stage, name, n in [('sft', 'sft.jsonl', len(sft)), ('rl', 'rl-contexts.jsonl', len(combined))]:
        write_json(output/f'{stage}-manifest.json', {'format_version': 1, 'shards': [
            {'path': name, 'split': 'train', 'rows': n, 'sha256': sha256_file(output/name)}]})
    write_json(output/'replay-selection.json', {'seed': seed, 'fraction_of_sft': fraction,
               'sft_indices': selected, 'ids': [sft[i].id for i in selected], 'mandatory_contexts': len(mandatory)})
    report = {'sft': len(sft), 'rl_original': len(rl), 'sft_replayed': len(selected), 'rl_total': len(combined),
              'replay_objective': 'REINFORCE-A + KL', 'seed': seed, 'tape_episodes': episodes,
              'source': provenance, 'files': {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()}}
    write_json(output/'PREPARED.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', default='shenjunhao/OpenJevData-140k')
    p.add_argument('--revision', default='main')
    p.add_argument('--output', default='work/data')
    p.add_argument('--replay-fraction', type=float, default=.25)
    p.add_argument('--seed', type=int, default=20260923)
    p.add_argument('--mandatory-replay', help='JSON list of content hashes that must enter RL')
    a = p.parse_args()
    from datasets import load_dataset
    local = Path(a.dataset)
    if local.is_dir():
        ds = load_dataset('parquet', data_files={s: str(local/'data'/f'{s}.parquet') for s in ('SFT', 'RL')})
        revision = {s: sha256_file(local/'data'/f'{s}.parquet') for s in ('SFT', 'RL')}
    else:
        from huggingface_hub import HfApi
        revision = HfApi().dataset_info(a.dataset, revision=a.revision).sha
        ds = load_dataset(a.dataset, revision=revision)
    rows = {s: [convert(r, s, i) for i, r in enumerate(ds[s])] for s in ('SFT', 'RL')}
    mandatory = json.loads(Path(a.mandatory_replay).read_text()) if a.mandatory_replay else []
    print(json.dumps(prepare(rows['SFT'], rows['RL'], a.output, fraction=a.replay_fraction,
                            seed=a.seed, mandatory=mandatory, provenance={'dataset': a.dataset, 'revision': revision}), indent=2))

if __name__ == '__main__':
    main()
