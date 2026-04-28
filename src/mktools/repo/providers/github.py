"""GitHub provider for :mod:`mktools.repo`."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

from ..errors import RepoNotFoundError, RepoProviderError
from ..http import http_download, http_get_json
from ..models import RepoEntry, RepoSpec, RepoTree
from .base import RepoProvider


class GitHubProvider(RepoProvider):
    """Provider for public or token-accessible GitHub repositories."""

    api_base = "https://api.github.com"

    def __init__(self, spec: RepoSpec) -> None:
        super().__init__(spec)
        if not spec.owner or not spec.name:
            raise ValueError("GitHubProvider requires owner and repository name")

    @property
    def repo_api(self) -> str:
        return f"{self.api_base}/repos/{self.spec.owner}/{self.spec.name}"

    @lru_cache(maxsize=64)
    def resolve_ref(self, ref: str) -> str:
        data = http_get_json(f"{self.repo_api}/commits/{quote(ref, safe='')}")
        try:
            return data["sha"]
        except KeyError as exc:
            raise RepoProviderError(f"Could not resolve ref {ref!r}") from exc

    def browse(self, ref: str, path: str = "", recursive: bool = False) -> RepoTree:
        resolved = self.resolve_ref(ref)
        normalized_path = path.strip("/")

        if recursive:
            return self._browse_recursive(ref=ref, resolved=resolved, path=normalized_path)
        return self._browse_contents(ref=ref, resolved=resolved, path=normalized_path)

    def _browse_recursive(self, ref: str, resolved: str, path: str) -> RepoTree:
        data = http_get_json(f"{self.repo_api}/git/trees/{resolved}?recursive=1")
        if data.get("truncated"):
            # Keep v1 explicit: callers can fall back to generic_git or refine path.
            raise RepoProviderError(
                "GitHub tree response was truncated. Try browsing a narrower path "
                "or use the generic git provider/sparse checkout path."
            )

        entries: list[RepoEntry] = []
        prefix = f"{path}/" if path else ""
        for item in data.get("tree", []):
            item_path = item["path"]
            if path and item_path != path and not item_path.startswith(prefix):
                continue

            raw_type = item.get("type")
            mode = item.get("mode")
            if raw_type == "blob":
                kind = "file"
                oid_kind = "git_blob_sha1"
            elif raw_type == "tree":
                kind = "dir"
                oid_kind = "git_tree_sha1"
            elif raw_type == "commit" and mode == "160000":
                kind = "submodule"
                oid_kind = "git_commit_sha1"
            else:
                kind = "file"
                oid_kind = None

            entries.append(
                RepoEntry(
                    path=item_path,
                    kind=kind,  # type: ignore[arg-type]
                    size=item.get("size"),
                    oid=item.get("sha"),
                    oid_kind=oid_kind,
                    mode=mode,
                )
            )

        return RepoTree(
            spec=self.spec,
            requested_ref=ref,
            resolved_commit=resolved,
            root_path=path,
            recursive=True,
            entries=tuple(entries),
        )

    def _browse_contents(self, ref: str, resolved: str, path: str) -> RepoTree:
        encoded_path = quote(path, safe="/")
        if encoded_path:
            url = f"{self.repo_api}/contents/{encoded_path}?ref={quote(resolved, safe='')}"
        else:
            url = f"{self.repo_api}/contents?ref={quote(resolved, safe='')}"

        data = http_get_json(url)
        items = data if isinstance(data, list) else [data]
        entries: list[RepoEntry] = []

        for item in items:
            item_type = item.get("type")
            if item_type == "file":
                kind = "file"
                oid_kind = "git_blob_sha1"
            elif item_type == "dir":
                kind = "dir"
                oid_kind = None
            elif item_type == "symlink":
                kind = "symlink"
                oid_kind = None
            elif item_type == "submodule":
                kind = "submodule"
                oid_kind = "git_commit_sha1"
            else:
                kind = "file"
                oid_kind = None

            entries.append(
                RepoEntry(
                    path=item["path"],
                    kind=kind,  # type: ignore[arg-type]
                    size=item.get("size"),
                    oid=item.get("sha"),
                    oid_kind=oid_kind,
                    download_url=item.get("download_url"),
                )
            )

        return RepoTree(
            spec=self.spec,
            requested_ref=ref,
            resolved_commit=resolved,
            root_path=path,
            recursive=False,
            entries=tuple(entries),
        )

    def file_meta(self, ref: str, path: str) -> RepoEntry:
        resolved = self.resolve_ref(ref)
        encoded_path = quote(path.strip("/"), safe="/")
        data = http_get_json(f"{self.repo_api}/contents/{encoded_path}?ref={quote(resolved, safe='')}")
        if isinstance(data, list) or data.get("type") != "file":
            raise RepoNotFoundError(f"Not a file: {path}")
        return RepoEntry(
            path=data["path"],
            kind="file",
            size=data.get("size"),
            oid=data.get("sha"),
            oid_kind="git_blob_sha1",
            download_url=data.get("download_url"),
        )

    def download_file(self, ref: str, path: str, dest: Path) -> None:
        resolved = self.resolve_ref(ref)
        encoded_path = quote(path.strip("/"), safe="/")
        url = f"{self.repo_api}/contents/{encoded_path}?ref={quote(resolved, safe='')}"
        http_download(url, dest, headers={"Accept": "application/vnd.github.raw"})
