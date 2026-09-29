"""Load immutable local JSONL shards; Hub support is an optional transport."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator

from .schema import DecisionContext, DecisionExample


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_jsonl(path: str | Path) -> Iterator[DecisionExample]:
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield DecisionExample.from_dict(json.loads(line))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc


def load_jsonl(path: str | Path) -> list[DecisionExample]:
    rows, seen = [], set()
    for row in iter_jsonl(path):
        if row.id in seen:
            raise ValueError(f"{path}: duplicate decision id: {row.id}")
        seen.add(row.id)
        rows.append(row)
    return rows


def load_context_jsonl(path: str | Path) -> list[DecisionContext]:
    path = Path(path)
    rows, seen = [], set()
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                context = DecisionContext.from_dict(json.loads(line))
                if context.id in seen:
                    raise ValueError(f"duplicate context id: {context.id}")
                seen.add(context.id)
                rows.append(context)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return rows


def load_context_manifest(path: str | Path, *, split: str | None = "rlcd-train") -> list[DecisionContext]:
    """Read dedicated context shards or the actor-context view in a pilot export.

    A supervised shard is never accepted as an actor context shard: the context
    schema rejects target fields instead of silently removing them at runtime.
    """
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError("unsupported context manifest format")
    context_views = [entry for entry in manifest.get("oracles", []) if entry.get("kind") == "actor-contexts"]
    shards = [{"split": "rlcd-train", **entry} for entry in context_views] if context_views else manifest.get("shards", [])
    selected = [entry for entry in shards if split is None or entry.get("split") == split]
    if not selected:
        raise ValueError(f"manifest contains no actor context shards for {split!r}")
    contexts, seen = [], set()
    for shard in selected:
        if not all(key in shard for key in ("path", "sha256", "rows", "split")):
            raise ValueError("context shards require path, sha256, rows and split")
        shard_path = path.parent / shard["path"]
        if sha256_file(shard_path) != shard["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {shard_path}")
        rows = load_context_jsonl(shard_path)
        if len(rows) != shard["rows"]:
            raise ValueError(f"row count mismatch for {shard_path}")
        for row in rows:
            if row.id in seen or row.source.get("split") not in (None, shard["split"]):
                raise ValueError(f"duplicate context or mismatched split for {row.id}")
            seen.add(row.id)
            contexts.append(row)
    return contexts


def load_manifest(path: str | Path, *, split: str | None = None) -> list[DecisionExample]:
    """Read ``{format_version:1, shards:[{path,sha256,split,rows}]}``.

    Every selected file is hash checked before its first row is accepted. Paths
    are relative to the manifest. A stable revision is supplied by the manifest
    itself, rather than silently resolving a moving remote dataset branch.
    """
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1 or not isinstance(manifest.get("shards"), list):
        raise ValueError("unsupported or malformed dataset manifest")
    shards = [shard for shard in manifest["shards"] if split is None or shard.get("split") == split]
    if not shards:
        raise ValueError(f"manifest contains no shards for split {split!r}")
    examples, seen = [], set()
    for shard in shards:
        if not all(key in shard for key in ("path", "sha256", "rows", "split")):
            raise ValueError("each shard needs path, sha256, rows and split")
        shard_path = path.parent / shard["path"]
        if sha256_file(shard_path) != shard["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {shard_path}")
        shard_rows = load_jsonl(shard_path)
        if len(shard_rows) != shard["rows"]:
            raise ValueError(f"row count mismatch for {shard_path}")
        for example in shard_rows:
            if example.id in seen:
                raise ValueError(f"duplicate decision id: {example.id}")
            if example.source.get("split") not in (None, shard["split"]):
                raise ValueError(f"row {example.id} split does not match its shard")
            seen.add(example.id)
            examples.append(example)
    return examples


def load_hub_manifest(repo_id: str, *, revision: str, filename: str = "manifest.json",
                      split: str | None = None, cache_dir: str | None = None) -> list[DecisionExample]:
    """Download an immutable dataset snapshot using optional huggingface_hub.

    Local JSONL training does not import or require any Hugging Face library.
    """
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision.lower()):
        raise ValueError("Hub data revision must be an immutable 40-character commit SHA")
    from huggingface_hub import snapshot_download

    root = snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision, cache_dir=cache_dir)
    return load_manifest(Path(root) / filename, split=split)
