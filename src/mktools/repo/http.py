"""Small stdlib HTTP helpers used by repository providers."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .errors import RepoProviderError

DEFAULT_TIMEOUT = 60


def _headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "mktools.repo/0.1",
    }
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if extra:
        headers.update(extra)
    return headers


def http_get_json(url: str, headers: dict[str, str] | None = None, timeout: int = DEFAULT_TIMEOUT) -> Any:
    """GET *url* and decode JSON."""

    req = urllib.request.Request(url, headers=_headers(headers))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RepoProviderError(f"HTTP {exc.code} for {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RepoProviderError(f"HTTP request failed for {url}: {exc}") from exc


def http_download(
    url: str,
    dest: Path,
    headers: dict[str, str] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    chunk_size: int = 1024 * 1024,
) -> None:
    """Download *url* to *dest* using streaming writes."""

    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers=_headers(headers))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, dest.open("wb") as fh:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                fh.write(chunk)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RepoProviderError(f"HTTP {exc.code} for {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RepoProviderError(f"HTTP request failed for {url}: {exc}") from exc
