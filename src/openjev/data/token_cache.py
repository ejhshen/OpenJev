"""Content-addressed token cache using SQLite; serialization remains authoritative."""
from dataclasses import asdict
from functools import lru_cache
import hashlib, inspect, json, sqlite3, zlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from openjev.models.decision.serializer import DecisionSerializer, SerializedDecision, SerializedBranch


def decision_key(decision):
    return hashlib.sha256(json.dumps(asdict(decision), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


_BUILD_SERIALIZER = None

def _init_builder(serializer):
    global _BUILD_SERIALIZER
    _BUILD_SERIALIZER = serializer

def _encode_payload(decision):
    x = _BUILD_SERIALIZER(decision)
    compact = [x.decision_id, x.option_ids, x.state_ids, x.question_ids, x.option_token_ids]
    return zlib.compress(json.dumps(compact, separators=(',', ':')).encode(), 1)


class TokenCache:
    def __init__(self, path, serializer, *, writable=False):
        self.serializer = serializer
        tok = serializer.tokenizer
        tokenizer = tok.backend_tokenizer.to_str() if hasattr(tok, 'backend_tokenizer') else repr(tok.get_vocab())
        contract = {'tokenizer': tokenizer, 'bos': serializer.add_bos_token,
                    'max_branch_length': serializer.max_branch_length,
                    'max_decision_tokens': serializer.max_decision_tokens,
                    'source': inspect.getsource(DecisionSerializer)}
        self.fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        path = Path(path)
        if writable:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(path, timeout=120)
            self.db.execute('PRAGMA journal_mode=WAL')
            self.db.execute('CREATE TABLE IF NOT EXISTS tokens (namespace TEXT, key TEXT, payload BLOB, PRIMARY KEY(namespace,key))')
            self.db.commit()
        else:
            self.db = sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True, timeout=120)
        self.writable = writable

    @staticmethod
    def decode(payload):
        name, options, state, question, suffixes = json.loads(zlib.decompress(payload))
        state, question = tuple(state), tuple(question)
        suffixes = tuple(tuple(x) for x in suffixes)
        prefix = state + question
        branches = tuple(SerializedBranch(prefix+s, len(prefix)-1, len(prefix)+len(s)-1) for s in suffixes)
        return SerializedDecision(name, tuple(options), state, question, suffixes, branches,
                                  len(prefix)+sum(map(len, suffixes)))

    @lru_cache(maxsize=1024)
    def __call__(self, decision):
        key = decision_key(decision)
        row = self.db.execute('SELECT payload FROM tokens WHERE namespace=? AND key=?', (self.fingerprint, key)).fetchone()
        if row is None:
            raise KeyError(f'token cache missing input {decision.id}; build with the same tokenizer/serializer')
        return self.decode(row[0])

    def populate(self, rows, *, workers=16, progress=None):
        if not self.writable:
            raise RuntimeError('read-only token cache')
        existing = {x[0] for x in self.db.execute('SELECT key FROM tokens WHERE namespace=?', (self.fingerprint,))}
        done = 0
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=_init_builder, initargs=(self.serializer,)) as pool:
            for start in range(0, len(rows), 512):
                pending = []
                for row in rows[start:start+512]:
                    d = row.decision; key = decision_key(d)
                    if key not in existing:
                        existing.add(key); pending.append((key, d))
                payloads = list(pool.map(_encode_payload, [d for _, d in pending], chunksize=16))
                with self.db:
                    self.db.executemany('INSERT OR IGNORE INTO tokens VALUES (?,?,?)',
                                        [(self.fingerprint, key, value) for (key, _), value in zip(pending, payloads)])
                done += len(pending)
                if progress and start % 8192 == 0:
                    progress({'processed': min(start+512, len(rows)), 'total': len(rows), 'new': done})
        self.db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        return {'rows': len(rows), 'new': done, 'fingerprint': self.fingerprint}
