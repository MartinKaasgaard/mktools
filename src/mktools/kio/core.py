from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from mktools.common.exceptions import PathValidationError, UnsupportedFileTypeError
from mktools.common.runtime import RuntimeConfig, RuntimeMode
from mktools.kfile.core import FileSystemOps

SUPPORTED_EXTENSIONS = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".txt": "text_table",
    ".xlsx": "excel",
    ".xls": "excel",
    ".parquet": "parquet",
    ".json": "json",
    ".jsonl": "jsonl",
    ".pkl": "pickle",
    ".pickle": "pickle",
}


@dataclass
class LoadResult:
    filepath: str
    loader: str
    rows: int
    columns: int
    extension: str
    kwargs_used: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "filepath": self.filepath,
            "loader": self.loader,
            "rows": self.rows,
            "columns": self.columns,
            "extension": self.extension,
            "kwargs_used": self.kwargs_used,
        }


class DataImporter:
    """Robust tabular-data importer with strong path and argument validation."""

    def __init__(self, *, runtime: RuntimeConfig | None = None) -> None:
        self.runtime = runtime or RuntimeConfig(mode=RuntimeMode.DEVELOPMENT, logger_name="mktools.kio")
        self.last_result: LoadResult | None = None

    @staticmethod
    def _ensure_non_empty_file(path: Path) -> None:
        if path.stat().st_size == 0:
            raise EmptyDataError(f"File is empty: {path}")

    @staticmethod
    def _coerce_filepath(filepath: str | Path) -> Path:
        return FileSystemOps.ensure_exists(filepath, must_be_file=True, parameter_name="filepath")

    @staticmethod
    def _resolve_loader(path: Path, file_type: str | None = None) -> str:
        if file_type is not None:
            candidate = str(file_type).strip().lower()
            if not candidate:
                raise ValueError("file_type must not be empty when provided.")
            allowed = set(SUPPORTED_EXTENSIONS.values())
            if candidate not in allowed:
                raise UnsupportedFileTypeError(
                    f"Unsupported file_type {file_type!r}. Supported values: {sorted(allowed)}"
                )
            return candidate
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"Unsupported file extension {suffix!r} for file {path.name!r}. "
                f"Supported extensions: {sorted(SUPPORTED_EXTENSIONS)}"
            )
        return SUPPORTED_EXTENSIONS[suffix]

    @staticmethod
    def _validate_expected_ext(path: Path, expected_ext: str | Sequence[str] | None) -> None:
        if expected_ext is None:
            return
        allowed = {expected_ext} if isinstance(expected_ext, str) else set(expected_ext)
        normalized = FileSystemOps.normalize_extensions(allowed)
        assert normalized is not None
        if path.suffix.lower() not in normalized:
            raise PathValidationError(
                f"Unexpected file extension {path.suffix!r}; expected one of {sorted(normalized)}."
            )

    @staticmethod
    def infer_delimiter(path: Path, *, sample_bytes: int = 8192) -> str:
        if not isinstance(sample_bytes, int) or sample_bytes <= 0:
            raise ValueError("sample_bytes must be a positive integer.")
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            sample = fh.read(sample_bytes)
        if not sample.strip():
            return ","
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
            return dialect.delimiter
        except csv.Error:
            return ","

    def load(
        self,
        filepath: str | Path,
        *,
        file_type: str | None = None,
        expected_ext: str | Sequence[str] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        path = self._coerce_filepath(filepath)
        self._ensure_non_empty_file(path)
        self._validate_expected_ext(path, expected_ext)
        loader = self._resolve_loader(path, file_type=file_type)

        dispatch = {
            "csv": self.load_csv,
            "tsv": self.load_tsv,
            "text_table": self.load_text_table,
            "excel": self.load_excel,
            "parquet": self.load_parquet,
            "json": self.load_json,
            "jsonl": self.load_jsonl,
            "pickle": self.load_pickle,
        }
        return dispatch[loader](path, **kwargs)

    def _finalize(self, path: Path, loader: str, df: pd.DataFrame, kwargs: dict[str, Any]) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Loader {loader!r} did not return a pandas DataFrame.")
        self.last_result = LoadResult(
            filepath=str(path),
            loader=loader,
            rows=int(df.shape[0]),
            columns=int(df.shape[1]),
            extension=path.suffix.lower(),
            kwargs_used=kwargs,
        )
        self.runtime.log(
            f"Loaded {path.name!r} via {loader} with {df.shape[0]} rows and {df.shape[1]} columns."
        )
        return df

    def load_csv(
        self,
        filepath: str | Path,
        *,
        encoding_candidates: Sequence[str] = ("utf-8", "utf-8-sig", "latin-1"),
        infer_delimiter: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        path = self._coerce_filepath(filepath)
        self._validate_expected_ext(path, ".csv")
        read_kwargs = dict(kwargs)
        if infer_delimiter and "sep" not in read_kwargs:
            read_kwargs["sep"] = self.infer_delimiter(path)
        last_exc: Exception | None = None
        for encoding in encoding_candidates:
            try:
                df = pd.read_csv(path, encoding=encoding, **read_kwargs)
                return self._finalize(path, "csv", df, {**read_kwargs, "encoding": encoding})
            except UnicodeDecodeError as exc:
                last_exc = exc
                continue
            except ParserError as exc:
                raise ParserError(f"Failed to parse CSV file {path}: {exc}") from exc
        raise UnicodeDecodeError(
            "unknown",
            b"",
            0,
            1,
            f"Unable to decode {path} using encodings {tuple(encoding_candidates)}; last error: {last_exc}",
        )

    def load_tsv(self, filepath: str | Path, **kwargs: Any) -> pd.DataFrame:
        path = self._coerce_filepath(filepath)
        self._validate_expected_ext(path, ".tsv")
        read_kwargs = {"sep": "\t", **kwargs}
        try:
            df = pd.read_csv(path, **read_kwargs)
        except ParserError as exc:
            raise ParserError(f"Failed to parse TSV file {path}: {exc}") from exc
        return self._finalize(path, "tsv", df, read_kwargs)

    def load_text_table(
        self,
        filepath: str | Path,
        *,
        infer_delimiter: bool = True,
        **kwargs: Any,
    ) -> pd.DataFrame:
        path = self._coerce_filepath(filepath)
        self._validate_expected_ext(path, ".txt")
        read_kwargs = dict(kwargs)
        if infer_delimiter and "sep" not in read_kwargs:
            read_kwargs["sep"] = self.infer_delimiter(path)
        try:
            df = pd.read_csv(path, **read_kwargs)
        except ParserError as exc:
            raise ParserError(f"Failed to parse text table {path}: {exc}") from exc
        return self._finalize(path, "text_table", df, read_kwargs)

    def load_excel(self, filepath: str | Path, **kwargs: Any) -> pd.DataFrame:
        path = self._coerce_filepath(filepath)
        self._validate_expected_ext(path, [".xlsx", ".xls"])
        try:
            df = pd.read_excel(path, **kwargs)
        except ImportError as exc:
            raise ImportError(
                "Reading Excel files requires an engine such as openpyxl or xlrd, depending on the file type."
            ) from exc
        return self._finalize(path, "excel", df, kwargs)

    def load_parquet(self, filepath: str | Path, **kwargs: Any) -> pd.DataFrame:
        path = self._coerce_filepath(filepath)
        self._validate_expected_ext(path, ".parquet")
        try:
            df = pd.read_parquet(path, **kwargs)
        except ImportError as exc:
            raise ImportError("Reading Parquet files requires a parquet engine such as pyarrow or fastparquet.") from exc
        return self._finalize(path, "parquet", df, kwargs)

    def load_json(self, filepath: str | Path, **kwargs: Any) -> pd.DataFrame:
        path = self._coerce_filepath(filepath)
        self._validate_expected_ext(path, ".json")
        try:
            df = pd.read_json(path, **kwargs)
        except ValueError as exc:
            raise ValueError(f"Failed to parse JSON file {path}: {exc}") from exc
        return self._finalize(path, "json", df, kwargs)

    def load_jsonl(self, filepath: str | Path, **kwargs: Any) -> pd.DataFrame:
        path = self._coerce_filepath(filepath)
        self._validate_expected_ext(path, ".jsonl")
        json_kwargs = {"lines": True, **kwargs}
        try:
            df = pd.read_json(path, **json_kwargs)
        except ValueError as exc:
            raise ValueError(f"Failed to parse JSONL file {path}: {exc}") from exc
        return self._finalize(path, "jsonl", df, json_kwargs)

    def load_pickle(self, filepath: str | Path, **kwargs: Any) -> pd.DataFrame:
        path = self._coerce_filepath(filepath)
        self._validate_expected_ext(path, [".pkl", ".pickle"])
        obj = pd.read_pickle(path, **kwargs)
        if not isinstance(obj, pd.DataFrame):
            raise TypeError(
                f"Pickle file {path} does not contain a pandas DataFrame; got {type(obj).__name__}."
            )
        return self._finalize(path, "pickle", obj, kwargs)

    def load_many(
        self,
        filepaths: Sequence[str | Path],
        *,
        on_error: Literal["raise", "skip", "collect"] = "raise",
        **kwargs: Any,
    ) -> dict[str, Any]:
        if on_error not in {"raise", "skip", "collect"}:
            raise ValueError("on_error must be 'raise', 'skip', or 'collect'.")
        if not isinstance(filepaths, Sequence) or isinstance(filepaths, (str, Path)):
            raise TypeError("filepaths must be a sequence of paths.")
        loaded: dict[str, pd.DataFrame] = {}
        errors: dict[str, str] = {}
        for filepath in filepaths:
            try:
                loaded[str(filepath)] = self.load(filepath, **kwargs)
            except Exception as exc:
                if on_error == "raise":
                    raise
                if on_error == "collect":
                    errors[str(filepath)] = str(exc)
        return {"loaded": loaded, "errors": errors}


def load_csv(
    filepath: str | Path,
    *,
    runtime: RuntimeConfig | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    return DataImporter(runtime=runtime).load_csv(filepath, **kwargs)


def load_table(
    filepath: str | Path,
    *,
    runtime: RuntimeConfig | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    return DataImporter(runtime=runtime).load(filepath, **kwargs)
