"""Public API for repository browsing, selection, and retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from .errors import RepoIntegrityError, RepoNotFoundError
from .integrity import git_blob_oid_file, sha256_file
from .manifest import read_manifest, write_manifest
from .models import FetchResult, RepoEntry, RepoTree, SelectionPlan, TransportKind
from .parser import parse_repo_url
from .providers import get_provider

TargetKind = Literal["repo", "dir", "file"]
ExistsPolicy = Literal["verify", "overwrite", "error"]


def browse(url: str, ref: str = "HEAD", path: str = "", recursive: bool = False) -> RepoTree:
    """Browse a repository URL.

    Parameters
    ----------
    url:
        Repository URL, for example ``https://github.com/user/repo``.
    ref:
        Branch, tag, commit SHA, or ``HEAD``.
    path:
        Optional repository path to inspect.
    recursive:
        Whether to return recursive entries below *path*.
    """

    spec = parse_repo_url(url, ref=ref)
    provider = get_provider(spec)
    return provider.browse(ref=ref, path=path, recursive=recursive)


def select(tree: RepoTree, target: TargetKind = "repo", path: str = "") -> SelectionPlan:
    """Select a repo, directory, or file from a :class:`RepoTree`.

    ``select`` does not perform network or filesystem operations. It only
    turns a browse result into a concrete file list for :func:`get`.
    """

    normalized_path = path.strip("/")

    if target == "repo":
        files = tuple(entry for entry in tree.entries if entry.kind == "file")
        if not files:
            raise RepoNotFoundError("No files found in tree. Try browse(..., recursive=True).")
        return SelectionPlan(tree=tree, selection_kind="repo", selection_path="", files=files)

    if target == "dir":
        prefix = f"{normalized_path}/" if normalized_path else ""
        files = tuple(
            entry for entry in tree.entries if entry.kind == "file" and (not prefix or entry.path.startswith(prefix))
        )
        if not files:
            raise RepoNotFoundError(f"No files found under directory: {normalized_path!r}")
        return SelectionPlan(tree=tree, selection_kind="dir", selection_path=normalized_path, files=files)

    if target == "file":
        files = tuple(entry for entry in tree.entries if entry.kind == "file" and entry.path == normalized_path)
        if len(files) != 1:
            raise RepoNotFoundError(f"File not found in tree: {normalized_path!r}")
        return SelectionPlan(tree=tree, selection_kind="file", selection_path=normalized_path, files=files)

    raise ValueError(f"Unsupported target: {target!r}")


def plan(tree: RepoTree, target: TargetKind = "repo", path: str = "") -> SelectionPlan:
    """Alias for :func:`select` for users who prefer explicit planning language."""

    return select(tree=tree, target=target, path=path)


def get(
    plan: SelectionPlan,
    dest: str | Path,
    transport: TransportKind = "auto",
    on_exists: ExistsPolicy = "verify",
) -> FetchResult:
    """Download files described by a :class:`SelectionPlan`.

    Existing files are verified before downloading. The default policy,
    ``on_exists='verify'``, skips files that match the current provider object
    id or the prior local manifest hash.
    """

    if on_exists not in {"verify", "overwrite", "error"}:
        raise ValueError("on_exists must be one of: 'verify', 'overwrite', 'error'")

    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    provider = get_provider(plan.tree.spec)
    manifest = read_manifest(dest_path)

    if manifest and manifest.get("resolved_commit") == plan.tree.resolved_commit:
        fast_verified = _verify_against_manifest(dest_path, plan.files, manifest)
        if fast_verified == len(plan.files):
            return FetchResult(
                dest=dest_path,
                downloaded=0,
                skipped=len(plan.files),
                verified=fast_verified,
                resolved_commit=plan.tree.resolved_commit,
                manifest_path=dest_path / ".mkrepo.manifest.json",
            )

    downloaded = 0
    skipped = 0
    verified = 0
    file_records: dict[str, dict[str, object]] = {}

    for entry in plan.files:
        local_path = dest_path / entry.path
        same = False

        if local_path.exists():
            if on_exists == "error":
                raise FileExistsError(f"Local file already exists: {local_path}")
            if on_exists == "verify":
                same = _local_file_matches(entry=entry, local_path=local_path, manifest=manifest)
                if same:
                    skipped += 1
                    verified += 1

        if not same:
            provider.download_file(plan.tree.resolved_commit, entry.path, local_path)
            downloaded += 1
            _verify_downloaded_file(entry=entry, local_path=local_path)

        file_records[entry.path] = _manifest_record(entry=entry, local_path=local_path)

    manifest_path = write_manifest(
        dest_path,
        {
            "version": 1,
            "repo_url": plan.tree.spec.url,
            "provider": plan.tree.spec.provider,
            "requested_ref": plan.tree.requested_ref,
            "resolved_commit": plan.tree.resolved_commit,
            "selection_kind": plan.selection_kind,
            "selection_path": plan.selection_path,
            "transport": transport,
            "files": file_records,
        },
    )

    return FetchResult(
        dest=dest_path,
        downloaded=downloaded,
        skipped=skipped,
        verified=verified,
        resolved_commit=plan.tree.resolved_commit,
        manifest_path=manifest_path,
    )


def fetch(
    plan: SelectionPlan,
    dest: str | Path,
    transport: TransportKind = "auto",
    on_exists: ExistsPolicy = "verify",
) -> FetchResult:
    """Alias for :func:`get`."""

    return get(plan=plan, dest=dest, transport=transport, on_exists=on_exists)


def _verify_against_manifest(dest: Path, files: tuple[RepoEntry, ...], manifest: dict) -> int:
    verified = 0
    records = manifest.get("files", {})
    for entry in files:
        local_path = dest / entry.path
        record = records.get(entry.path)
        if not local_path.exists() or not record:
            break
        if record.get("sha256") != sha256_file(local_path):
            break
        verified += 1
    return verified


def _local_file_matches(entry: RepoEntry, local_path: Path, manifest: dict | None) -> bool:
    if not local_path.is_file():
        return False

    if entry.size is not None and local_path.stat().st_size != entry.size:
        return False

    if entry.oid and entry.oid_kind == "git_blob_sha1":
        return git_blob_oid_file(local_path) == entry.oid

    if manifest:
        record = manifest.get("files", {}).get(entry.path)
        if record and record.get("sha256") == sha256_file(local_path):
            return True

    return False


def _verify_downloaded_file(entry: RepoEntry, local_path: Path) -> None:
    if not local_path.exists() or not local_path.is_file():
        raise RepoIntegrityError(f"Downloaded file is missing: {local_path}")

    if entry.size is not None and local_path.stat().st_size != entry.size:
        raise RepoIntegrityError(
            f"Size mismatch for {entry.path}: expected {entry.size}, got {local_path.stat().st_size}"
        )

    if entry.oid and entry.oid_kind == "git_blob_sha1":
        local_oid = git_blob_oid_file(local_path)
        if local_oid != entry.oid:
            raise RepoIntegrityError(f"Git blob OID mismatch for {entry.path}: expected {entry.oid}, got {local_oid}")


def _manifest_record(entry: RepoEntry, local_path: Path) -> dict[str, object]:
    return {
        "size": local_path.stat().st_size,
        "sha256": sha256_file(local_path),
        "provider_oid": entry.oid,
        "provider_oid_kind": entry.oid_kind,
    }
