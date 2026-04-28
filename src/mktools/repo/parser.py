"""Repository URL parsing."""

from __future__ import annotations

import re
from urllib.parse import urlparse

from .errors import RepoParseError
from .models import RepoSpec

_GITHUB_PATH_RE = re.compile(r"^/([^/]+)/([^/]+?)(?:\.git)?/?$")
_GITHUB_SSH_RE = re.compile(r"^git@github\.com:([^/]+)/(.+?)(?:\.git)?$")


def parse_repo_url(url: str, ref: str = "HEAD") -> RepoSpec:
    """Parse a repository URL into a :class:`RepoSpec`.

    v1 supports a first-class GitHub provider and uses ``generic_git`` as a
    fallback for other git remotes.
    """

    if not url or not isinstance(url, str):
        raise RepoParseError("Repository URL must be a non-empty string")

    ssh_match = _GITHUB_SSH_RE.match(url)
    if ssh_match:
        owner, name = ssh_match.groups()
        return RepoSpec(
            url=url,
            provider="github",
            host="github.com",
            owner=owner,
            name=name,
            ref=ref,
        )

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https", "ssh", "git"}:
        raise RepoParseError(f"Unsupported repository URL scheme: {url}")

    host = parsed.netloc.lower()
    if host in {"github.com", "www.github.com"}:
        match = _GITHUB_PATH_RE.match(parsed.path)
        if not match:
            raise RepoParseError(f"Could not parse GitHub repository URL: {url}")
        owner, name = match.groups()
        return RepoSpec(
            url=url,
            provider="github",
            host="github.com",
            owner=owner,
            name=name,
            ref=ref,
        )

    return RepoSpec(url=url, provider="generic_git", host=host or "unknown", ref=ref)
