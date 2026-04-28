"""Integrity helpers for repository downloads."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 hex digest for *data*."""

    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 hex digest for a local file."""

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_oid_bytes(data: bytes, algo: str = "sha1") -> str:
    """Return the Git blob object id for *data*.

    Git blob OIDs are not plain hashes of the file bytes. They hash the
    byte sequence ``b"blob <size>\\0" + data``.
    """

    h = hashlib.new(algo)
    h.update(f"blob {len(data)}\0".encode("utf-8"))
    h.update(data)
    return h.hexdigest()


def git_blob_oid_file(path: Path, algo: str = "sha1", chunk_size: int = 1024 * 1024) -> str:
    """Return the Git blob object id for a local file."""

    size = path.stat().st_size
    h = hashlib.new(algo)
    h.update(f"blob {size}\0".encode("utf-8"))
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()
