"""Generic git CLI provider for :mod:`mktools.repo`.

This provider intentionally uses the git executable rather than reimplementing
Git protocols in Python. It is a fallback for non-GitHub remotes and for cases
where a sparse/partial clone is the most reliable transport.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from ..errors import RepoNotFoundError, RepoProviderError
from ..models import RepoEntry, RepoSpec, RepoTree
from .base import RepoProvider


class GenericGitProvider(RepoProvider):
    """Provider backed by the local ``git`` command."""

    def __init__(self, spec: RepoSpec) -> None:
        super().__init__(spec)
        if shutil.which("git") is None:
            raise RepoProviderError("The generic git provider requires the 'git' executable")

    def _run(self, args: list[str], cwd: str | None = None) -> str:
        proc = subprocess.run(
            args,
            cwd=cwd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            raise RepoProviderError(proc.stderr.strip() or f"Command failed: {' '.join(args)}")
        return proc.stdout

    def _temp_clone(self, ref: str) -> Path:
        tmpdir = Path(tempfile.mkdtemp(prefix="mkrepo_"))
        try:
            # Fast path for branch/tag refs.
            self._run(
                [
                    "git",
                    "clone",
                    "--depth=1",
                    "--filter=blob:none",
                    "--no-checkout",
                    "--branch",
                    ref,
                    self.spec.url,
                    str(tmpdir),
                ]
            )
        except RepoProviderError:
            shutil.rmtree(tmpdir, ignore_errors=True)
            tmpdir = Path(tempfile.mkdtemp(prefix="mkrepo_"))
            self._run(["git", "clone", "--filter=blob:none", "--no-checkout", self.spec.url, str(tmpdir)])

        self._run(["git", "-C", str(tmpdir), "checkout", ref])
        return tmpdir

    def resolve_ref(self, ref: str) -> str:
        candidates = [ref]
        if ref not in {"HEAD"} and not ref.startswith("refs/"):
            candidates.extend([f"refs/heads/{ref}", f"refs/tags/{ref}"])

        for candidate in candidates:
            out = self._run(["git", "ls-remote", self.spec.url, candidate])
            if out.strip():
                return out.split()[0]

        out = self._run(["git", "ls-remote", self.spec.url, "HEAD"])
        if not out.strip():
            raise RepoProviderError(f"Could not resolve ref {ref!r}")
        return out.split()[0]

    def browse(self, ref: str, path: str = "", recursive: bool = False) -> RepoTree:
        normalized_path = path.strip("/")
        tmpdir = self._temp_clone(ref)
        try:
            resolved = self._run(["git", "-C", str(tmpdir), "rev-parse", "HEAD"]).strip()
            entries = self._ls_tree(tmpdir=tmpdir, path=normalized_path, recursive=recursive)
            return RepoTree(
                spec=self.spec,
                requested_ref=ref,
                resolved_commit=resolved,
                root_path=normalized_path,
                recursive=recursive,
                entries=tuple(entries),
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _path_type(self, tmpdir: Path, path: str) -> str | None:
        if not path:
            return "tree"
        proc = subprocess.run(
            ["git", "-C", str(tmpdir), "cat-file", "-t", f"HEAD:{path}"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            return None
        return proc.stdout.strip()

    def _ls_tree(self, tmpdir: Path, path: str, recursive: bool) -> list[RepoEntry]:
        path_type = self._path_type(tmpdir, path)
        if path_type is None:
            raise RepoNotFoundError(f"Path not found: {path}")

        if path_type == "tree" and path and not recursive:
            cmd = ["git", "-C", str(tmpdir), "ls-tree", "--long", f"HEAD:{path}"]
            prefix = f"{path}/"
        else:
            cmd = ["git", "-C", str(tmpdir), "ls-tree", "--full-tree", "--long"]
            if recursive:
                cmd.append("-r")
            cmd.extend(["HEAD"])
            if path:
                cmd.extend(["--", path])
            prefix = ""

        out = self._run(cmd)
        entries: list[RepoEntry] = []
        for line in out.splitlines():
            # Format: "<mode> <type> <object> <size>\t<path>"
            left, raw_path = line.split("\t", 1)
            parts = left.split()
            if len(parts) < 4:
                continue
            mode, raw_type, oid, size_raw = parts[:4]
            item_path = prefix + raw_path
            size = None if size_raw == "-" else int(size_raw)

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
                    size=size,
                    oid=oid,
                    oid_kind=oid_kind,
                    mode=mode,
                )
            )
        return entries

    def file_meta(self, ref: str, path: str) -> RepoEntry:
        tree = self.browse(ref=ref, path=path, recursive=False)
        for entry in tree.entries:
            if entry.path == path.strip("/") and entry.kind == "file":
                return entry
        raise RepoNotFoundError(f"File not found: {path}")

    def download_file(self, ref: str, path: str, dest: Path) -> None:
        normalized_path = path.strip("/")
        tmpdir = self._temp_clone(ref)
        try:
            # Fetch the requested blob into the working tree.
            self._run(["git", "-C", str(tmpdir), "checkout", "HEAD", "--", normalized_path])
            src = tmpdir / normalized_path
            if not src.exists() or not src.is_file():
                raise RepoNotFoundError(f"Downloaded file not found: {path}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
