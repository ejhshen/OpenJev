"""Frozen model predictions, content-addressed independently of outcome labels."""
import hashlib, json, sqlite3
from pathlib import Path
from functools import lru_cache
from openjev.data.token_cache import decision_key

class ReferenceCache:
    def __init__(self, path, contract, *, writable=False):
        self.fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        path = Path(path)
        if writable:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(path, timeout=120)
            self.db.execute('PRAGMA journal_mode=WAL')
            self.db.execute('CREATE TABLE IF NOT EXISTS predictions (namespace TEXT, key TEXT, logs TEXT, PRIMARY KEY(namespace,key))')
            self.db.commit()
        else:
            self.db = sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True, timeout=120)

    @lru_cache(maxsize=2048)
    def get(self, decision):
        row = self.db.execute('SELECT logs FROM predictions WHERE namespace=? AND key=?',
                              (self.fingerprint, decision_key(decision))).fetchone()
        if row is None:
            raise KeyError(f'reference prediction missing for {decision.id}')
        result = json.loads(row[0])
        if len(result) != len(decision.options):
            raise ValueError('reference option count mismatch')
        return result

    def put(self, pairs):
        with self.db:
            self.db.executemany('INSERT OR IGNORE INTO predictions VALUES (?,?,?)',
                [(self.fingerprint, decision_key(d), json.dumps(logs, allow_nan=False)) for d,logs in pairs])


def reference_contract(artifact, token_fingerprint, kernel_backend):
    from openjev.artifacts import sha256_file
    return {'artifact': sha256_file(Path(artifact)/'manifest.json'), 'tokens': token_fingerprint,
            'kernel_backend': kernel_backend, 'temperature': 1.0, 'format': 'fp32-log-prob-v1'}
