"""Provider registry for :mod:`mktools.repo`."""

from __future__ import annotations

from ..models import RepoSpec
from .base import RepoProvider
from .generic_git import GenericGitProvider
from .github import GitHubProvider


def get_provider(spec: RepoSpec) -> RepoProvider:
    """Return the provider implementation for *spec*."""

    if spec.provider == "github":
        return GitHubProvider(spec)
    if spec.provider == "generic_git":
        return GenericGitProvider(spec)
    raise ValueError(f"Unsupported repository provider: {spec.provider}")


__all__ = ["RepoProvider", "GitHubProvider", "GenericGitProvider", "get_provider"]
