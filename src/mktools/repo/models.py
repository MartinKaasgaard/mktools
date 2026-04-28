"""Typed data models for :mod:`mktools.repo`.

The models intentionally avoid provider-specific fields except for a small
number of normalized identifiers. This keeps the public API stable while
allowing GitHub, GitLab, local git, or other providers to be added later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ProviderName = Literal["github", "generic_git"]
EntryKind = Literal["file", "dir", "symlink", "submodule"]
SelectionKind = Literal["repo", "dir", "file"]
TransportKind = Literal["auto", "raw", "clone", "sparse_clone"]


@dataclass(frozen=True)
class RepoSpec:
    """Normalized repository reference."""

    url: str
    provider: ProviderName
    host: str
    owner: str | None = None
    name: str | None = None
    ref: str = "HEAD"


@dataclass(frozen=True)
class RepoEntry:
    """A normalized entry in a repository tree."""

    path: str
    kind: EntryKind
    size: int | None = None
    oid: str | None = None
    oid_kind: str | None = None  # e.g. git_blob_sha1, git_tree_sha1
    mode: str | None = None
    download_url: str | None = None


@dataclass(frozen=True)
class RepoTree:
    """A browse result for a repository, directory, or file."""

    spec: RepoSpec
    requested_ref: str
    resolved_commit: str
    root_path: str
    recursive: bool
    entries: tuple[RepoEntry, ...]


@dataclass(frozen=True)
class SelectionPlan:
    """Concrete download plan produced by :func:`mktools.repo.select`."""

    tree: RepoTree
    selection_kind: SelectionKind
    selection_path: str
    files: tuple[RepoEntry, ...]
    transport: TransportKind = "auto"


@dataclass(frozen=True)
class FetchResult:
    """Summary returned by :func:`mktools.repo.get`."""

    dest: Path
    downloaded: int
    skipped: int
    verified: int
    resolved_commit: str
    manifest_path: Path | None = None
