from __future__ import annotations

import os
from pathlib import Path

import pytest

from mktools.repo import api
from mktools.repo.errors import RepoNotFoundError
from mktools.repo.integrity import git_blob_oid_bytes, git_blob_oid_file, sha256_bytes, sha256_file
from mktools.repo.manifest import read_manifest, write_manifest
from mktools.repo.models import RepoEntry, RepoSpec, RepoTree
from mktools.repo.parser import parse_repo_url


class FakeProvider:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads
        self.downloads: list[str] = []

    def download_file(self, ref: str, path: str, dest: Path) -> None:  # noqa: ARG002
        try:
            payload = self.payloads[path]
        except KeyError as exc:
            raise FileNotFoundError(path) from exc
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
        self.downloads.append(path)


def _tree_for_payloads(payloads: dict[str, bytes], root_path: str = "dataset") -> RepoTree:
    spec = RepoSpec(
        url="https://github.com/example/project",
        provider="github",
        host="github.com",
        owner="example",
        name="project",
        ref="main",
    )
    entries = tuple(
        RepoEntry(
            path=path,
            kind="file",
            size=len(payload),
            oid=git_blob_oid_bytes(payload),
            oid_kind="git_blob_sha1",
        )
        for path, payload in payloads.items()
    )
    return RepoTree(
        spec=spec,
        requested_ref="main",
        resolved_commit="abc123",
        root_path=root_path,
        recursive=True,
        entries=entries,
    )


def test_parse_github_https_url() -> None:
    spec = parse_repo_url("https://github.com/alecuba16/fuhrlander", ref="master")

    assert spec.provider == "github"
    assert spec.host == "github.com"
    assert spec.owner == "alecuba16"
    assert spec.name == "fuhrlander"
    assert spec.ref == "master"


def test_parse_github_https_git_suffix() -> None:
    spec = parse_repo_url("https://github.com/alecuba16/fuhrlander.git")

    assert spec.provider == "github"
    assert spec.owner == "alecuba16"
    assert spec.name == "fuhrlander"


def test_parse_github_ssh_url() -> None:
    spec = parse_repo_url("git@github.com:alecuba16/fuhrlander.git")

    assert spec.provider == "github"
    assert spec.host == "github.com"
    assert spec.owner == "alecuba16"
    assert spec.name == "fuhrlander"


def test_parse_generic_git_url() -> None:
    spec = parse_repo_url("https://git.example.com/team/project.git")

    assert spec.provider == "generic_git"
    assert spec.host == "git.example.com"
    assert spec.owner is None
    assert spec.name is None


def test_integrity_helpers_roundtrip(tmp_path: Path) -> None:
    payload = b"hello\n"
    path = tmp_path / "hello.txt"
    path.write_bytes(payload)

    assert sha256_file(path) == sha256_bytes(payload)
    assert git_blob_oid_file(path) == git_blob_oid_bytes(payload)
    assert git_blob_oid_bytes(payload) == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_manifest_read_write(tmp_path: Path) -> None:
    manifest = {"version": 1, "files": {"a.txt": {"sha256": "abc"}}}

    path = write_manifest(tmp_path, manifest)

    assert path.exists()
    assert read_manifest(tmp_path) == manifest


def test_select_directory_filters_files() -> None:
    payloads = {
        "dataset/a.txt": b"a",
        "dataset/nested/b.txt": b"b",
        "other/c.txt": b"c",
    }
    tree = _tree_for_payloads(payloads)

    plan = api.select(tree, target="dir", path="dataset")

    assert plan.selection_kind == "dir"
    assert plan.selection_path == "dataset"
    assert sorted(entry.path for entry in plan.files) == ["dataset/a.txt", "dataset/nested/b.txt"]


def test_select_file_requires_exact_match() -> None:
    tree = _tree_for_payloads({"dataset/a.txt": b"a"})

    plan = api.select(tree, target="file", path="dataset/a.txt")

    assert len(plan.files) == 1
    assert plan.files[0].path == "dataset/a.txt"


def test_select_file_missing_raises() -> None:
    tree = _tree_for_payloads({"dataset/a.txt": b"a"})

    with pytest.raises(RepoNotFoundError):
        api.select(tree, target="file", path="dataset/missing.txt")


def test_get_downloads_writes_manifest_and_skips_identical_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payloads = {
        "dataset/a.txt": b"alpha",
        "dataset/b.txt": b"bravo",
    }
    provider = FakeProvider(payloads)
    tree = _tree_for_payloads(payloads)
    plan = api.select(tree, target="dir", path="dataset")

    monkeypatch.setattr(api, "get_provider", lambda spec: provider)

    first = api.get(plan, dest=tmp_path)
    assert first.downloaded == 2
    assert first.skipped == 0
    assert first.verified == 0
    assert sorted(provider.downloads) == ["dataset/a.txt", "dataset/b.txt"]
    assert (tmp_path / "dataset/a.txt").read_bytes() == b"alpha"
    assert first.manifest_path is not None
    assert first.manifest_path.exists()

    second = api.get(plan, dest=tmp_path)
    assert second.downloaded == 0
    assert second.skipped == 2
    assert second.verified == 2
    assert sorted(provider.downloads) == ["dataset/a.txt", "dataset/b.txt"]


def test_get_redownloads_modified_local_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payloads = {"dataset/a.txt": b"alpha"}
    provider = FakeProvider(payloads)
    tree = _tree_for_payloads(payloads)
    plan = api.select(tree, target="file", path="dataset/a.txt")

    monkeypatch.setattr(api, "get_provider", lambda spec: provider)

    first = api.get(plan, dest=tmp_path)
    assert first.downloaded == 1

    (tmp_path / "dataset/a.txt").write_bytes(b"tampered")

    second = api.get(plan, dest=tmp_path)
    assert second.downloaded == 1
    assert second.skipped == 0
    assert (tmp_path / "dataset/a.txt").read_bytes() == b"alpha"
    assert provider.downloads == ["dataset/a.txt", "dataset/a.txt"]


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_MKTOOLS_REPO_INTEGRATION") != "1",
    reason="Set RUN_MKTOOLS_REPO_INTEGRATION=1 to run network integration tests.",
)
def test_github_fuhrlander_dataset_roundtrip(tmp_path: Path) -> None:
    tree = api.browse(
        "https://github.com/alecuba16/fuhrlander",
        ref="master",
        path="dataset",
        recursive=True,
    )
    plan = api.select(tree, target="dir", path="dataset")
    result = api.get(plan, dest=tmp_path)

    downloaded = sorted(path.name for path in (tmp_path / "dataset").iterdir() if path.is_file())

    assert result.downloaded >= 1
    assert "wind_plant_data.json" in downloaded
    assert any(name.endswith(".json.bz2") for name in downloaded)
