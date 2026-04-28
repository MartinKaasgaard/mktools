"""Provider interface for :mod:`mktools.repo`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..models import RepoEntry, RepoSpec, RepoTree


class RepoProvider(ABC):
    """Abstract base class for repository providers."""

    def __init__(self, spec: RepoSpec) -> None:
        self.spec = spec

    @abstractmethod
    def resolve_ref(self, ref: str) -> str:
        """Resolve a branch, tag, or commit-ish to an immutable commit id."""

    @abstractmethod
    def browse(self, ref: str, path: str = "", recursive: bool = False) -> RepoTree:
        """Browse a repository path."""

    @abstractmethod
    def file_meta(self, ref: str, path: str) -> RepoEntry:
        """Return normalized metadata for a single file."""

    @abstractmethod
    def download_file(self, ref: str, path: str, dest: Path) -> None:
        """Download a single file to *dest*."""
