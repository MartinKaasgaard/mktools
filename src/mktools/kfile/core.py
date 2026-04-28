from __future__ import annotations

import bz2
import fnmatch
import hashlib
import os
import shutil
import stat
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Sequence

import pandas as pd

from mktools.common.exceptions import PathValidationError, UnsafeArchiveError
from mktools.common.runtime import RuntimeConfig, RuntimeMode

HIDDEN_SYSTEM_NAMES = {"thumbs.db", "desktop.ini", ".ds_store"}


@dataclass(frozen=True)
class ArchiveExtractionReport:
    archive_path: str
    target_dir: str
    total_members: int
    extracted_count: int
    skipped_count: int
    overwritten_count: int
    extracted_members: list[str]
    skipped_members: list[str]
    overwritten_members: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FileSystemOps:
    """Validated, reusable filesystem operations."""

    @staticmethod
    def coerce_path(value: str | os.PathLike[str], *, parameter_name: str = "path") -> Path:
        if isinstance(value, Path):
            return value.expanduser()
        if isinstance(value, os.PathLike):
            return Path(value).expanduser()
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                raise PathValidationError(f"{parameter_name} must not be empty.")
            return Path(stripped).expanduser()
        raise TypeError(f"{parameter_name} must be a str or Path-like object.")

    @staticmethod
    def normalize_extensions(
        extensions: Sequence[str] | set[str] | None,
    ) -> set[str] | None:
        if extensions is None:
            return None
        normalized: set[str] = set()
        for ext in extensions:
            if not isinstance(ext, str):
                raise TypeError("Each extension must be a string.")
            item = ext.strip().lower()
            if not item:
                raise ValueError("Empty extension values are not allowed.")
            if not item.startswith("."):
                item = f".{item}"
            normalized.add(item)
        return normalized

    @classmethod
    def ensure_exists(
        cls,
        path: str | os.PathLike[str],
        *,
        must_be_file: bool | None = None,
        must_be_dir: bool | None = None,
        parameter_name: str = "path",
    ) -> Path:
        p = cls.coerce_path(path, parameter_name=parameter_name)
        if not p.exists():
            raise FileNotFoundError(f"{parameter_name} does not exist: {p}")
        if must_be_file is True and not p.is_file():
            raise PathValidationError(f"{parameter_name} must be an existing file: {p}")
        if must_be_dir is True and not p.is_dir():
            raise PathValidationError(f"{parameter_name} must be an existing directory: {p}")
        return p.resolve()

    @classmethod
    def ensure_directory(
        cls,
        path: str | os.PathLike[str],
        *,
        create: bool = False,
        parameter_name: str = "path",
    ) -> Path:
        p = cls.coerce_path(path, parameter_name=parameter_name)
        if p.exists() and not p.is_dir():
            raise PathValidationError(f"{parameter_name} exists but is not a directory: {p}")
        if not p.exists():
            if not create:
                raise FileNotFoundError(f"{parameter_name} does not exist: {p}")
            p.mkdir(parents=True, exist_ok=True)
        return p.resolve()

    @staticmethod
    def is_hidden(path: Path) -> bool:
        return path.name.startswith(".") or path.name.lower() in HIDDEN_SYSTEM_NAMES

    @staticmethod
    def file_hash(
        path: str | os.PathLike[str],
        *,
        algorithm: str = "sha256",
        chunk_size: int = 1024 * 1024,
    ) -> str:
        p = FileSystemOps.ensure_exists(path, must_be_file=True)
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        try:
            hasher = hashlib.new(algorithm)
        except Exception as exc:
            raise ValueError(f"Unsupported hash algorithm: {algorithm!r}") from exc
        with p.open("rb") as fh:
            while True:
                chunk = fh.read(chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()


class DirectoryScanner:
    """List and summarize directory contents with strong validation."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        extensions: Sequence[str] | set[str] | None = None,
        include_hidden: bool = False,
        recurse: bool = False,
        follow_symlinks: bool = False,
        runtime: RuntimeConfig | None = None,
    ) -> None:
        self.root = FileSystemOps.ensure_directory(root)
        self.extensions = FileSystemOps.normalize_extensions(extensions)
        self.include_hidden = bool(include_hidden)
        self.recurse = bool(recurse)
        self.follow_symlinks = bool(follow_symlinks)
        self.runtime = runtime or RuntimeConfig(mode=RuntimeMode.DEVELOPMENT, logger_name="mktools.kfile")

    def _iter_entries(self):
        if self.recurse:
            yield from self.root.rglob("*")
        else:
            yield from self.root.iterdir()

    def _include(self, path: Path) -> bool:
        if not self.include_hidden and FileSystemOps.is_hidden(path):
            return False
        if path.is_file() and self.extensions is not None:
            return path.suffix.lower() in self.extensions
        return True

    def to_frame(self) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        for entry in self._iter_entries():
            try:
                if not self._include(entry):
                    continue
                is_dir = entry.is_dir()
                is_file = entry.is_file()
                if not (is_dir or is_file):
                    continue
                if entry.is_symlink() and not self.follow_symlinks:
                    continue
                st = entry.stat()
                records.append(
                    {
                        "path": str(entry.resolve()),
                        "relative_path": str(entry.relative_to(self.root)),
                        "name": entry.name,
                        "parent": str(entry.parent.resolve()),
                        "suffix": entry.suffix.lower(),
                        "is_dir": is_dir,
                        "is_file": is_file,
                        "is_hidden": FileSystemOps.is_hidden(entry),
                        "size_bytes": st.st_size if is_file else 0,
                        "modified_time": datetime.fromtimestamp(st.st_mtime),
                        "created_time": datetime.fromtimestamp(st.st_ctime),
                        "mode": stat.filemode(st.st_mode),
                        "depth": len(entry.relative_to(self.root).parts),
                    }
                )
            except PermissionError as exc:
                self.runtime.log(f"Skipping unreadable path {entry}: {exc}")
        frame = pd.DataFrame(records)
        if not frame.empty:
            frame = frame.sort_values(["is_dir", "relative_path"], ascending=[False, True]).reset_index(drop=True)
        return frame

    def brief(self, *, max_files: int = 10) -> dict[str, Any]:
        if not isinstance(max_files, int) or max_files <= 0:
            raise ValueError("max_files must be a positive integer.")
        frame = self.to_frame()
        dirs = frame.loc[frame["is_dir"], "relative_path"].tolist() if not frame.empty else []
        files = frame.loc[frame["is_file"], "relative_path"].tolist() if not frame.empty else []
        return {
            "root": str(self.root),
            "dir_count": len(dirs),
            "file_count": len(files),
            "dirs": dirs,
            "files": files[:max_files],
        }

    def summary(self) -> dict[str, Any]:
        frame = self.to_frame()
        files = frame[frame["is_file"]].copy() if not frame.empty else pd.DataFrame()
        extension_counts = (
            files["suffix"].value_counts(dropna=False).to_dict() if not files.empty else {}
        )
        files_per_dir = (
            files.assign(directory=files["relative_path"].map(lambda x: str(Path(x).parent)))
            .groupby("directory")
            .size()
            .sort_index()
            .to_dict()
            if not files.empty
            else {}
        )
        return {
            "root": str(self.root),
            "total_entries": int(len(frame)),
            "total_dirs": int((frame["is_dir"].sum()) if not frame.empty else 0),
            "total_files": int((frame["is_file"].sum()) if not frame.empty else 0),
            "extensions": extension_counts,
            "files_per_dir": files_per_dir,
        }

    def tree(self, *, max_depth: int | None = None) -> str:
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 0):
            raise ValueError("max_depth must be None or a non-negative integer.")

        lines = [f"{self.root.name}/"]

        def walk(path: Path, prefix: str = "", depth: int = 0) -> None:
            if max_depth is not None and depth >= max_depth:
                return
            children = [child for child in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())) if self._include(child)]
            for idx, child in enumerate(children):
                connector = "└── " if idx == len(children) - 1 else "├── "
                lines.append(f"{prefix}{connector}{child.name}{'/' if child.is_dir() else ''}")
                if child.is_dir():
                    extension = "    " if idx == len(children) - 1 else "│   "
                    walk(child, prefix + extension, depth + 1)

        walk(self.root)
        return "\n".join(lines)


class FileIndexBuilder:
    """Build a metadata index for files, with optional labels."""

    def __init__(
        self,
        root_dir: str | os.PathLike[str],
        *,
        extensions: Sequence[str] | set[str] | None = None,
        include_hidden: bool = False,
        label_source: Literal["parent_dir", "filename", "custom", "constant"] = "parent_dir",
        label_property: Callable[[Path], str] | None = None,
        constant_label: str | None = None,
        runtime: RuntimeConfig | None = None,
    ) -> None:
        self.root_dir = FileSystemOps.ensure_directory(root_dir)
        self.extensions = FileSystemOps.normalize_extensions(extensions)
        self.include_hidden = bool(include_hidden)
        self.label_source = label_source
        self.label_property = label_property
        self.constant_label = constant_label
        self.runtime = runtime or RuntimeConfig(mode=RuntimeMode.DEVELOPMENT, logger_name="mktools.kfile")
        self._validate_label_config()

    def _validate_label_config(self) -> None:
        allowed = {"parent_dir", "filename", "custom", "constant"}
        if self.label_source not in allowed:
            raise ValueError(f"label_source must be one of {sorted(allowed)}")
        if self.label_source == "custom" and not callable(self.label_property):
            raise ValueError("label_property must be callable when label_source='custom'.")
        if self.label_source == "constant" and (self.constant_label is None or not str(self.constant_label).strip()):
            raise ValueError("constant_label must be provided when label_source='constant'.")

    def _include_file(self, path: Path) -> bool:
        if not path.is_file():
            return False
        if not self.include_hidden and FileSystemOps.is_hidden(path):
            return False
        if self.extensions is None:
            return True
        return path.suffix.lower() in self.extensions

    def _derive_label(self, path: Path) -> str:
        if self.label_source == "parent_dir":
            return path.parent.name
        if self.label_source == "filename":
            return path.stem
        if self.label_source == "custom":
            assert self.label_property is not None
            value = self.label_property(path)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Custom label_property returned invalid label for {path}.")
            return value.strip()
        assert self.constant_label is not None
        return self.constant_label

    def build(self, *, subdir: str | os.PathLike[str] | None = None) -> pd.DataFrame:
        target = self.root_dir if subdir is None else FileSystemOps.ensure_directory(self.root_dir / subdir)
        records: list[dict[str, Any]] = []
        for path in target.rglob("*"):
            if not self._include_file(path):
                continue
            try:
                st = path.stat()
                records.append(
                    {
                        "filepath": str(path.resolve()),
                        "relative_path": str(path.relative_to(self.root_dir)),
                        "filename": path.stem,
                        "basename": path.name,
                        "extension": path.suffix.lower(),
                        "label": self._derive_label(path),
                        "size_bytes": st.st_size,
                        "created_time": datetime.fromtimestamp(st.st_ctime),
                        "modified_time": datetime.fromtimestamp(st.st_mtime),
                    }
                )
            except PermissionError as exc:
                self.runtime.log(f"Skipping unreadable file {path}: {exc}")
        return pd.DataFrame(records)


class ZipHandler:
    """Safe ZIP inspection and extraction with zip-slip protection."""

    def __init__(
        self,
        zip_path: str | os.PathLike[str],
        *,
        runtime: RuntimeConfig | None = None,
    ) -> None:
        self.zip_path = FileSystemOps.ensure_exists(zip_path, must_be_file=True, parameter_name="zip_path")
        if self.zip_path.suffix.lower() != ".zip":
            raise PathValidationError(f"zip_path must point to a .zip file: {self.zip_path}")
        self.runtime = runtime or RuntimeConfig(mode=RuntimeMode.DEVELOPMENT, logger_name="mktools.kfile")
        self._validate_archive()

    def _validate_archive(self) -> None:
        if not zipfile.is_zipfile(self.zip_path):
            raise UnsafeArchiveError(f"File is not a valid ZIP archive: {self.zip_path}")
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            bad_member = zf.testzip()
            if bad_member is not None:
                raise UnsafeArchiveError(f"ZIP archive failed CRC validation at member: {bad_member}")
            names = [info.filename for info in zf.infolist() if not info.is_dir()]
            duplicates = {name for name in names if names.count(name) > 1}
            if duplicates:
                raise UnsafeArchiveError(
                    f"ZIP archive contains duplicate file members, which is unsafe: {sorted(duplicates)[:10]}"
                )
            for info in zf.infolist():
                self._validate_member_name(info.filename)

    @staticmethod
    def _validate_member_name(name: str) -> None:
        candidate = Path(name)
        if candidate.is_absolute():
            raise UnsafeArchiveError(f"Archive member uses an absolute path: {name!r}")
        if ".." in candidate.parts:
            raise UnsafeArchiveError(f"Archive member escapes the target directory: {name!r}")

    def list_members(self) -> pd.DataFrame:
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            records = []
            for info in zf.infolist():
                records.append(
                    {
                        "member": info.filename,
                        "is_dir": info.is_dir(),
                        "compressed_size": info.compress_size,
                        "uncompressed_size": info.file_size,
                        "crc": info.CRC,
                    }
                )
        return pd.DataFrame(records)

    def extract(
        self,
        target_dir: str | os.PathLike[str],
        *,
        if_exists: Literal["skip", "overwrite", "raise"] = "skip",
        members: Sequence[str] | None = None,
        verify: bool = True,
    ) -> ArchiveExtractionReport:
        if if_exists not in {"skip", "overwrite", "raise"}:
            raise ValueError("if_exists must be 'skip', 'overwrite', or 'raise'.")
        target = FileSystemOps.ensure_directory(target_dir, create=True, parameter_name="target_dir")
        member_filter = set(members) if members is not None else None
        extracted: list[str] = []
        skipped: list[str] = []
        overwritten: list[str] = []

        with zipfile.ZipFile(self.zip_path, "r") as zf:
            infos = [info for info in zf.infolist() if not info.is_dir()]
            if member_filter is not None:
                unknown = member_filter - {info.filename for info in infos}
                if unknown:
                    raise ValueError(f"Requested members not found in ZIP: {sorted(unknown)}")
                infos = [info for info in infos if info.filename in member_filter]

            for info in infos:
                self._validate_member_name(info.filename)
                destination = (target / info.filename).resolve()
                if target not in destination.parents and destination != target:
                    raise UnsafeArchiveError(f"Unsafe resolved extraction target: {destination}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    if if_exists == "skip":
                        skipped.append(info.filename)
                        continue
                    if if_exists == "raise":
                        raise FileExistsError(f"Target file already exists: {destination}")
                    overwritten.append(info.filename)
                with zf.open(info, "r") as src, destination.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted.append(info.filename)

            if verify:
                for info in infos:
                    destination = (target / info.filename).resolve()
                    if not destination.exists():
                        raise RuntimeError(f"Extracted file is missing after extraction: {destination}")
                    if destination.stat().st_size != info.file_size:
                        raise RuntimeError(
                            f"Extracted file size mismatch for {destination}: expected {info.file_size}, got {destination.stat().st_size}"
                        )

        return ArchiveExtractionReport(
            archive_path=str(self.zip_path),
            target_dir=str(target),
            total_members=len(infos),
            extracted_count=len(extracted),
            skipped_count=len(skipped),
            overwritten_count=len(overwritten),
            extracted_members=extracted,
            skipped_members=skipped,
            overwritten_members=overwritten,
        )

    def ensure_unzipped(self, target_dir: str | os.PathLike[str]) -> ArchiveExtractionReport:
        return self.extract(target_dir, if_exists="skip", verify=True)


def list_directory(
    path: str | os.PathLike[str],
    *,
    max_files: int = 10,
    exts: Sequence[str] | set[str] | None = None,
    include_hidden: bool = False,
    recurse: bool = False,
    runtime: RuntimeConfig | None = None,
) -> dict[str, Any]:
    scanner = DirectoryScanner(
        path,
        extensions=exts,
        include_hidden=include_hidden,
        recurse=recurse,
        runtime=runtime,
    )
    return scanner.brief(max_files=max_files)


def ensure_unzipped(
    zip_path: str | os.PathLike[str],
    target_dir: str | os.PathLike[str],
    *,
    runtime: RuntimeConfig | None = None,
) -> ArchiveExtractionReport:
    return ZipHandler(zip_path, runtime=runtime).ensure_unzipped(target_dir)


@dataclass(frozen=True)
class DecompressionReport:
    source: str
    output: str
    status: str
    source_size_bytes: int
    output_size_bytes: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Bz2Handler:
    """
    Handler for single-file .bz2 decompression.

    A .bz2 file is normally one compressed file, not a multi-file archive.
    Example:
        turbine_80.json.bz2 -> turbine_80.json
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        runtime: RuntimeConfig | None = None,
    ) -> None:
        self.path = FileSystemOps.ensure_exists(
            path,
            must_be_file=True,
            parameter_name="path",
        )

        if self.path.suffix.lower() != ".bz2":
            raise PathValidationError(f"path must point to a .bz2 file: {self.path}")

        self.runtime = runtime or RuntimeConfig(
            mode=RuntimeMode.DEVELOPMENT,
            logger_name="mktools.kfile",
        )

    def ensure_decompressed(
        self,
        destination: str | os.PathLike[str],
        *,
        preserve_relative_to: str | os.PathLike[str] | None = None,
        overwrite: bool = False,
    ) -> DecompressionReport:
        destination_path = FileSystemOps.ensure_directory(
            destination,
            create=True,
            parameter_name="destination",
        )

        if preserve_relative_to is not None:
            base_path = FileSystemOps.ensure_directory(
                preserve_relative_to,
                parameter_name="preserve_relative_to",
            )
            try:
                relative_path = self.path.relative_to(base_path)
            except ValueError as exc:
                raise PathValidationError(
                    f"Cannot preserve relative path because {self.path} "
                    f"is not inside {base_path}."
                ) from exc

            output_relative_path = relative_path.with_suffix("")
            output_path = destination_path / output_relative_path
        else:
            output_path = destination_path / self.path.with_suffix("").name

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and output_path.stat().st_size > 0 and not overwrite:
            return DecompressionReport(
                source=str(self.path),
                output=str(output_path),
                status="skipped_existing",
                source_size_bytes=self.path.stat().st_size,
                output_size_bytes=output_path.stat().st_size,
            )

        with bz2.open(self.path, "rb") as src, output_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)

        return DecompressionReport(
            source=str(self.path),
            output=str(output_path),
            status="decompressed",
            source_size_bytes=self.path.stat().st_size,
            output_size_bytes=output_path.stat().st_size,
        )


def ensure_decompressed_bz2(
    path: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    preserve_relative_to: str | os.PathLike[str] | None = None,
    overwrite: bool = False,
    runtime: RuntimeConfig | None = None,
) -> DecompressionReport:
    return Bz2Handler(path, runtime=runtime).ensure_decompressed(
        destination,
        preserve_relative_to=preserve_relative_to,
        overwrite=overwrite,
    )
