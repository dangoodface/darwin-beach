"""Mirror $ARENA_DATA/out/ to a target, uploading only what is not there yet.

Order matters for the page: content-addressed files (genomes, kernels) go first,
indexes last, so an index never names a file the target does not have. Indexes
are always candidates but skipped when byte-identical to what was last sent.

Targets: a local folder (the dry-run mirror) or R2 through boto3. R2 credentials
come from environment variables only; a systemd EnvironmentFile that lives
outside the repo and is filled from `pass` sets them (see systemd/README.md).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .store import data_root

INDEX_NAMES = {"latest.json", "curve.json", "tree.json", "tree-full.json", "ladder.json", "index.json"}
IMMUTABLE = "public, max-age=31536000, immutable"
SHORT = "public, max-age=60"
TYPES = {".json": "application/json", ".bin": "application/octet-stream", ".wasm": "application/wasm"}


class Target(Protocol):
    def has(self, key: str, sha256: str, immutable: bool) -> bool: ...
    def put(self, key: str, data: bytes, content_type: str, cache_control: str) -> None: ...


class LocalTarget:
    def __init__(self, folder: Path) -> None:
        self.folder = folder

    def has(self, key: str, sha256: str, immutable: bool) -> bool:
        p = self.folder / key
        return p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == sha256

    def put(self, key: str, data: bytes, content_type: str, cache_control: str) -> None:
        p = self.folder / key
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, p)


class R2Target:
    """Cloudflare R2 over the S3 API. `client` is injectable for tests."""

    def __init__(self, bucket: str, client, state_path: Path) -> None:
        self.bucket, self.client, self.state_path = bucket, client, state_path
        try:
            self.state: dict[str, str] = json.loads(state_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            self.state = {}

    @classmethod
    def from_env(cls, root: Path | None = None) -> "R2Target":
        try:
            bucket, endpoint = os.environ["ARENA_R2_BUCKET"], os.environ["ARENA_R2_ENDPOINT"]
            key_id, secret = os.environ["ARENA_R2_KEY_ID"], os.environ["ARENA_R2_SECRET"]
        except KeyError as e:
            raise SystemExit(f"missing environment variable {e}; see systemd/README.md") from None
        import boto3  # optional extra: uv sync --extra r2

        client = boto3.client("s3", endpoint_url=endpoint, aws_access_key_id=key_id,
                              aws_secret_access_key=secret, region_name="auto")
        return cls(bucket, client, (root or data_root()) / f"upload-state-{bucket}.json")

    def has(self, key: str, sha256: str, immutable: bool) -> bool:
        if self.state.get(key) == sha256:
            return True
        if immutable:  # named by its hash, so presence is enough
            try:
                self.client.head_object(Bucket=self.bucket, Key=key)
            except Exception:
                return False
            self.state[key] = sha256
            return True
        return False

    def put(self, key: str, data: bytes, content_type: str, cache_control: str) -> None:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type,
                               CacheControl=cache_control)
        self.state[key] = hashlib.sha256(data).hexdigest()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state))


@dataclass
class Summary:
    uploaded: list[str] = field(default_factory=list)
    skipped: int = 0


def _is_index(path: Path) -> bool:
    return path.name in INDEX_NAMES


def sync(src: Path, target: Target, dry_run: bool = False) -> Summary:
    files = [p for p in sorted(src.rglob("*")) if p.is_file() and ".tmp" not in p.name]
    files.sort(key=lambda p: (_is_index(p), p.relative_to(src).as_posix()))  # content first, indexes last
    summary = Summary()
    for p in files:
        key = p.relative_to(src).as_posix()
        data = p.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        immutable = not _is_index(p)
        if target.has(key, sha, immutable):
            summary.skipped += 1
            continue
        if not dry_run:
            target.put(key, data, TYPES.get(p.suffix, "application/octet-stream"),
                       IMMUTABLE if immutable else SHORT)
        summary.uploaded.append(key)
    return summary


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Mirror $ARENA_DATA/out to a folder or to R2.")
    ap.add_argument("--src", type=Path, help="default: $ARENA_DATA/out")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--to", type=Path, help="local folder to mirror into (the dry-run mirror)")
    g.add_argument("--r2", action="store_true", help="upload to R2 (ARENA_R2_* environment variables)")
    ap.add_argument("--dry-run", action="store_true", help="list what would be uploaded; write nothing")
    args = ap.parse_args(argv)
    src = args.src or data_root() / "out"
    if not src.is_dir():
        raise SystemExit(f"nothing to upload: {src} does not exist")
    target: Target = LocalTarget(args.to) if args.to else R2Target.from_env()
    s = sync(src, target, args.dry_run)
    verb = "would upload" if args.dry_run else "uploaded"
    print(f"{verb} {len(s.uploaded)} files, {s.skipped} already there")
    for k in s.uploaded[:20]:
        print("  ", k)
