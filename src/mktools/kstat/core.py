from __future__ import annotations

import difflib
import logging
import re
import warnings
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype, is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype, is_object_dtype, is_string_dtype

from mktools.common.runtime import AnalysisDepth, RuntimeConfig, ensure_runtime_config


@dataclass
class DatasetOverview:
    dataset_name: str
    rows: int
    columns: int
    memory_mb: float
    duplicate_rows: int
    duplicate_row_pct: float
    missing_cells: int
    missing_cell_pct: float
    numeric_columns: int
    categorical_columns: int
    datetime_columns: int
    boolean_columns: int
    timedelta_columns: int
    other_columns: int


class FeatureCatalog:
    VALID_CATEGORIES = {"numerical", "categorical", "datetime", "boolean", "timedelta", "other"}

    def __init__(self, df: pd.DataFrame, excluded: Optional[Iterable[str]] = None) -> None:
        self._validate_df(df)
        self._df = df
        self._excluded = set(excluded or [])
        self._features: dict[str, list[str]] = {key: [] for key in self.VALID_CATEGORIES}
        self.refresh(df)

    @staticmethod
    def _validate_df(df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")

    @staticmethod
    def _infer_category(series: pd.Series) -> str:
        dtype = series.dtype
        if is_bool_dtype(dtype):
            return "boolean"
        if is_datetime64_any_dtype(dtype):
            return "datetime"
        if pd.api.types.is_timedelta64_dtype(dtype):
            return "timedelta"
        if is_numeric_dtype(dtype):
            return "numerical"
        if isinstance(dtype, CategoricalDtype) or is_object_dtype(dtype) or is_string_dtype(dtype):
            return "categorical"
        return "other"

    def refresh(self, df: Optional[pd.DataFrame] = None) -> "FeatureCatalog":
        if df is not None:
            self._validate_df(df)
            self._df = df
        self._features = {key: [] for key in self.VALID_CATEGORIES}
        for col in self._df.columns:
            if col in self._excluded:
                continue
            self._features[self._infer_category(self._df[col])].append(col)
        for key in self._features:
            self._features[key] = sorted(self._features[key])
        return self

    def get(self, category: Optional[str] = None) -> dict[str, list[str]] | list[str]:
        if category is None:
            return {k: list(v) for k, v in self._features.items()}
        if category not in self.VALID_CATEGORIES:
            raise ValueError(f"Unknown category {category!r}.")
        return list(self._features[category])

    def as_frame(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for category, columns in self._features.items():
            for col in columns:
                rows.append({"column": col, "category": category, "dtype": str(self._df[col].dtype)})
        return pd.DataFrame(rows).sort_values(["category", "column"]).reset_index(drop=True)


class KStatProfiler:
    DEFAULT_MISSING_TOKENS = {"", " ", "na", "n/a", "null", "none", "nan", "unknown", "missing", "?", "-", "--"}

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        dataset_name: str = "dataset",
        depth: str | AnalysisDepth = AnalysisDepth.STANDARD,
        runtime: Optional[RuntimeConfig] = None,
        copy_df: bool = True,
        sample_values: int = 3,
        max_similarity_checks: int = 500,
    ) -> None:
        self._validate_dataframe(df)
        self.df = df.copy(deep=True) if copy_df else df
        self.dataset_name = self._validate_dataset_name(dataset_name)
        self.depth = AnalysisDepth.coerce(depth)
        self.runtime = ensure_runtime_config(runtime, logger_name="mktools.kstat")
        self.sample_values = self._validate_positive_int(sample_values, "sample_values", minimum=1)
        self.max_similarity_checks = self._validate_positive_int(max_similarity_checks, "max_similarity_checks", minimum=50)
        self.catalog = FeatureCatalog(self.df)
        self.reports: dict[str, Any] = {}
        self.runtime.log(f"Initialized KStatProfiler(name={self.dataset_name!r}, depth={self.depth.value}, shape={self.df.shape})", level=logging.INFO)

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if df.columns.has_duplicates:
            duplicates = df.columns[df.columns.duplicated()].tolist()
            raise ValueError(f"DataFrame contains duplicate column names: {duplicates[:10]}")

    @staticmethod
    def _validate_dataset_name(name: str) -> str:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("dataset_name must be a non-empty string.")
        return name.strip()

    @staticmethod
    def _validate_positive_int(value: Any, field: str, minimum: int = 0) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{field} must be an integer.")
        if value < minimum:
            raise ValueError(f"{field} must be >= {minimum}.")
        return value

    @staticmethod
    def _validate_ratio(value: float, field: str) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{field} must be a float between 0 and 1.")
        if not 0 <= float(value) <= 1:
            raise ValueError(f"{field} must be between 0 and 1.")

    def _refresh_after_mutation(self) -> None:
        self.catalog.refresh(self.df)
        self.reports.clear()

    def _cache(self, name: str, obj: Any) -> Any:
        self.reports[name] = obj
        return obj

    def profile(self, refresh: bool = True) -> dict[str, Any]:
        if refresh:
            self.reports.clear()
        bundle: dict[str, Any] = {
            "overview": self.overview(),
            "column_summary": self.column_summary(),
            "missingness": self.missingness_summary(),
            "duplicates": self.duplicate_summary(),
            "dtype_audit": self.dtype_audit(),
            "feature_catalog": self.catalog.as_frame(),
        }
        if self.depth in {AnalysisDepth.STANDARD, AnalysisDepth.DEEP}:
            bundle["descriptive_stats"] = self.descriptive_stats()
            bundle["outliers"] = self.outlier_summary()
            bundle["cardinality"] = self.cardinality_summary()
            bundle["constant_columns"] = self.constant_columns()
            bundle["correlation"] = self.correlation_summary()
        if self.depth is AnalysisDepth.DEEP:
            bundle["string_quality"] = self.string_quality_summary()
            bundle["similar_values"] = self.detect_possible_value_inconsistencies()
            bundle["identical_columns"] = self.identical_column_pairs()
            bundle["datetime_summary"] = self.datetime_summary()
        self.reports.update(bundle)
        return bundle

    def overview(self) -> DatasetOverview:
        rows, columns = self.df.shape
        memory_mb = float(self.df.memory_usage(deep=True).sum() / (1024**2))
        missing_cells = int(self.df.isna().sum().sum())
        total_cells = max(rows * columns, 1)
        duplicate_rows = int(self.df.duplicated().sum())
        catalog = self.catalog.get()
        return self._cache("overview", DatasetOverview(
            dataset_name=self.dataset_name,
            rows=rows,
            columns=columns,
            memory_mb=memory_mb,
            duplicate_rows=duplicate_rows,
            duplicate_row_pct=(duplicate_rows / max(rows, 1)) * 100,
            missing_cells=missing_cells,
            missing_cell_pct=(missing_cells / total_cells) * 100,
            numeric_columns=len(catalog["numerical"]),
            categorical_columns=len(catalog["categorical"]),
            datetime_columns=len(catalog["datetime"]),
            boolean_columns=len(catalog["boolean"]),
            timedelta_columns=len(catalog["timedelta"]),
            other_columns=len(catalog["other"]),
        ))

    def column_summary(self, sample_values: Optional[int] = None) -> pd.DataFrame:
        sample_values = self.sample_values if sample_values is None else self._validate_positive_int(sample_values, "sample_values", minimum=1)
        rows = []
        for col in self.df.columns:
            series = self.df[col]
            missing = int(series.isna().sum())
            non_null = int(series.notna().sum())
            unique_non_null = int(series.nunique(dropna=True))
            rows.append({
                "column": col,
                "category": FeatureCatalog._infer_category(series),
                "dtype": str(series.dtype),
                "rows": len(series),
                "non_null": non_null,
                "missing": missing,
                "missing_pct": round((missing / max(len(series), 1)) * 100, 4),
                "unique_non_null": unique_non_null,
                "memory_mb": round(series.memory_usage(deep=True) / (1024**2), 6),
                "sample_values": series.dropna().astype(str).drop_duplicates().head(sample_values).tolist(),
            })
        result = pd.DataFrame(rows).sort_values(["missing_pct", "column"], ascending=[False, True]).reset_index(drop=True)
        return self._cache("column_summary", result)

    def missingness_summary(self) -> pd.DataFrame:
        total_rows = max(len(self.df), 1)
        rows = [{"column": col, "missing_count": int(self.df[col].isna().sum()), "missing_pct": round((int(self.df[col].isna().sum()) / total_rows) * 100, 4)} for col in self.df.columns]
        result = pd.DataFrame(rows).sort_values(["missing_count", "column"], ascending=[False, True]).reset_index(drop=True)
        return self._cache("missingness", result)

    def duplicate_summary(self, subset: Optional[Iterable[str]] = None) -> pd.DataFrame:
        subset_list = list(subset) if subset is not None else None
        if subset_list is not None:
            unknown = [col for col in subset_list if col not in self.df.columns]
            if unknown:
                raise KeyError(f"Unknown columns in subset: {unknown}")
        duplicate_mask = self.df.duplicated(subset=subset_list, keep=False)
        duplicate_count = int(self.df.duplicated(subset=subset_list).sum())
        result = pd.DataFrame([{
            "scope": "all_columns" if subset_list is None else ",".join(subset_list),
            "duplicate_rows": duplicate_count,
            "duplicate_row_pct": round((duplicate_count / max(len(self.df), 1)) * 100, 4),
            "rows_in_duplicate_groups": int(duplicate_mask.sum()),
        }])
        return self._cache("duplicates", result)

    @staticmethod
    def _numeric_parse_rate(series: pd.Series) -> float:
        non_null = series.dropna()
        if non_null.empty:
            return 1.0
        parsed = pd.to_numeric(non_null.astype(str).str.replace(",", "", regex=False), errors="coerce")
        return float(parsed.notna().mean())

    @staticmethod
    def _datetime_parse_rate(series: pd.Series) -> float:
        non_null = series.dropna()
        if non_null.empty:
            return 1.0
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            parsed = pd.to_datetime(non_null.astype(str), errors="coerce")
        return float(parsed.notna().mean())

    @staticmethod
    def _boolean_parse_rate(series: pd.Series) -> float:
        non_null = series.dropna()
        if non_null.empty:
            return 1.0
        normalized = non_null.astype(str).str.strip().str.lower()
        valid = {"true", "false", "yes", "no", "y", "n", "1", "0", "t", "f"}
        return float(normalized.isin(valid).mean())

    def dtype_audit(self, *, numeric_threshold: float = 0.95, datetime_threshold: float = 0.95, boolean_threshold: float = 0.98) -> pd.DataFrame:
        self._validate_ratio(numeric_threshold, "numeric_threshold")
        self._validate_ratio(datetime_threshold, "datetime_threshold")
        self._validate_ratio(boolean_threshold, "boolean_threshold")
        rows = []
        for col in self.df.columns:
            series = self.df[col]
            current_dtype = str(series.dtype)
            numeric_rate = self._numeric_parse_rate(series)
            datetime_rate = self._datetime_parse_rate(series)
            boolean_rate = self._boolean_parse_rate(series)
            if is_bool_dtype(series.dtype):
                recommended_dtype = "boolean"
            elif is_numeric_dtype(series.dtype):
                recommended_dtype = "numeric"
            elif is_datetime64_any_dtype(series.dtype):
                recommended_dtype = "datetime64[ns]"
            elif numeric_rate >= numeric_threshold:
                recommended_dtype = "numeric"
            elif datetime_rate >= datetime_threshold:
                recommended_dtype = "datetime64[ns]"
            elif boolean_rate >= boolean_threshold:
                recommended_dtype = "boolean"
            elif isinstance(series.dtype, CategoricalDtype):
                recommended_dtype = "category"
            else:
                recommended_dtype = "object"
            rows.append({
                "column": col,
                "current_dtype": current_dtype,
                "recommended_dtype": recommended_dtype,
                "numeric_parse_rate": round(numeric_rate, 4),
                "datetime_parse_rate": round(datetime_rate, 4),
                "boolean_parse_rate": round(boolean_rate, 4),
                "flag_inconsistency": recommended_dtype != current_dtype and not (recommended_dtype == "numeric" and current_dtype.startswith(("int", "float"))),
            })
        result = pd.DataFrame(rows).sort_values(["flag_inconsistency", "column"], ascending=[False, True]).reset_index(drop=True)
        return self._cache("dtype_audit", result)

    def cardinality_summary(self, high_cardinality_threshold: int = 50) -> pd.DataFrame:
        self._validate_positive_int(high_cardinality_threshold, "high_cardinality_threshold", minimum=1)
        object_like = self.df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        rows = [{
            "column": col,
            "unique_non_null": int(self.df[col].nunique(dropna=True)),
            "high_cardinality": int(self.df[col].nunique(dropna=True)) > high_cardinality_threshold,
        } for col in object_like]
        result = pd.DataFrame(rows).sort_values(["unique_non_null", "column"], ascending=[False, True]).reset_index(drop=True)
        return self._cache("cardinality", result)

    def constant_columns(self, near_constant_threshold: float = 0.99) -> pd.DataFrame:
        self._validate_ratio(near_constant_threshold, "near_constant_threshold")
        rows = []
        for col in self.df.columns:
            s = self.df[col].dropna()
            if s.empty:
                rows.append({"column": col, "top_value": np.nan, "top_ratio": np.nan, "is_constant": False, "is_near_constant": False})
                continue
            vc = s.value_counts(dropna=False)
            top_ratio = float(vc.iloc[0] / len(s))
            rows.append({
                "column": col,
                "top_value": vc.index[0],
                "top_ratio": round(top_ratio, 4),
                "is_constant": vc.shape[0] == 1,
                "is_near_constant": top_ratio >= near_constant_threshold,
            })
        result = pd.DataFrame(rows).sort_values(["is_constant", "top_ratio", "column"], ascending=[False, False, True]).reset_index(drop=True)
        return self._cache("constant_columns", result)

    def descriptive_stats(self) -> pd.DataFrame:
        if self.df.empty:
            return self._cache("descriptive_stats", pd.DataFrame())
        try:
            result = self.df.describe(include="all", datetime_is_numeric=True).T
        except TypeError:
            result = self.df.describe(include="all").T
        return self._cache("descriptive_stats", result)

    def outlier_summary(self) -> pd.DataFrame:
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        rows = []
        for col in numeric_cols:
            s = pd.to_numeric(self.df[col], errors="coerce").dropna()
            if s.empty:
                continue
            q1 = float(s.quantile(0.25))
            q3 = float(s.quantile(0.75))
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            mask = (s < lower) | (s > upper)
            rows.append({
                "column": col,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_limit": lower,
                "upper_limit": upper,
                "outlier_count": int(mask.sum()),
                "outlier_pct": round(float(mask.mean() * 100), 4),
            })
        result = pd.DataFrame(rows)
        if not result.empty:
            result = result.sort_values(["outlier_count", "column"], ascending=[False, True]).reset_index(drop=True)
        return self._cache("outliers", result)

    def correlation_summary(self, method: str = "pearson") -> pd.DataFrame:
        method = str(method).strip().lower()
        if method not in {"pearson", "spearman", "kendall"}:
            raise ValueError("method must be one of: 'pearson', 'spearman', 'kendall'")
        numeric = self.df.select_dtypes(include=[np.number])
        if numeric.shape[1] < 2:
            return self._cache("correlation", pd.DataFrame())
        return self._cache("correlation", numeric.corr(method=method))

    def string_quality_summary(self) -> pd.DataFrame:
        object_like = self.df.select_dtypes(include=["object", "string"]).columns.tolist()
        rows = []
        for col in object_like:
            s = self.df[col].astype("string")
            non_null = s.dropna()
            stripped = non_null.str.strip()
            normalized = stripped.str.lower()
            rows.append({
                "column": col,
                "leading_or_trailing_spaces": int((non_null != stripped).sum()) if not non_null.empty else 0,
                "empty_strings": int((stripped == "").sum()) if not non_null.empty else 0,
                "case_variant_collisions": int(non_null.nunique(dropna=True) - normalized.nunique(dropna=True)) if not non_null.empty else 0,
            })
        result = pd.DataFrame(rows)
        if not result.empty:
            result = result.sort_values(["leading_or_trailing_spaces", "empty_strings", "column"], ascending=[False, False, True]).reset_index(drop=True)
        return self._cache("string_quality", result)

    def detect_possible_value_inconsistencies(self, column: Optional[str] = None, exempt_column: Optional[str] = None, cutoff: float = 0.85) -> pd.DataFrame:
        self._validate_ratio(cutoff, "cutoff")
        object_columns = self.df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        if column is not None and exempt_column is not None:
            raise ValueError("Use either 'column' or 'exempt_column', not both.")
        if column is not None:
            columns_to_check = [column]
        elif exempt_column is not None:
            columns_to_check = [c for c in object_columns if c != exempt_column]
        else:
            columns_to_check = object_columns
        rows = []
        for col in columns_to_check:
            unique_values = self.df[col].dropna().astype(str).str.strip().str.lower().drop_duplicates().tolist()
            if len(unique_values) <= 1 or len(unique_values) > self.max_similarity_checks:
                continue
            value_set = sorted(unique_values)
            for value in value_set:
                matches = [m for m in difflib.get_close_matches(value, value_set, n=3, cutoff=cutoff) if m != value]
                if matches:
                    rows.append({"column": col, "value": value, "potential_matches": matches})
        result = pd.DataFrame(rows)
        if not result.empty:
            result = result.sort_values(["column", "value"]).reset_index(drop=True)
        return self._cache("similar_values", result)

    def identical_column_pairs(self) -> pd.DataFrame:
        cols = self.df.columns.tolist()
        pairs = []
        for i, left in enumerate(cols):
            for right in cols[i+1:]:
                if self.df[left].equals(self.df[right]):
                    pairs.append({"left_column": left, "right_column": right})
        return self._cache("identical_columns", pd.DataFrame(pairs))

    def datetime_summary(self) -> pd.DataFrame:
        dt_cols = self.df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
        rows = []
        for col in dt_cols:
            s = self.df[col].dropna()
            rows.append({
                "column": col,
                "min": s.min() if not s.empty else pd.NaT,
                "max": s.max() if not s.empty else pd.NaT,
                "range_days": (s.max() - s.min()).days if len(s) > 1 else np.nan,
                "missing_count": int(self.df[col].isna().sum()),
            })
        result = pd.DataFrame(rows)
        if not result.empty:
            result = result.sort_values("column").reset_index(drop=True)
        return self._cache("datetime_summary", result)

    def compare_data_dictionary(self, expected: Mapping[str, Any] | Iterable[str], *, normalize: bool = True) -> dict[str, list[str]]:
        expected_columns = list(expected.keys()) if isinstance(expected, Mapping) else list(expected)
        actual_columns = self.df.columns.astype(str).tolist()
        if normalize:
            norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
            expected_norm_map = {norm(c): c for c in expected_columns}
            actual_norm_map = {norm(c): c for c in actual_columns}
            expected_norm = set(expected_norm_map)
            actual_norm = set(actual_norm_map)
            matched = sorted(expected_norm_map[n] for n in expected_norm & actual_norm)
            missing = sorted(expected_norm_map[n] for n in expected_norm - actual_norm)
            extra = sorted(actual_norm_map[n] for n in actual_norm - expected_norm)
        else:
            expected_set = set(expected_columns)
            actual_set = set(actual_columns)
            matched = sorted(expected_set & actual_set)
            missing = sorted(expected_set - actual_set)
            extra = sorted(actual_set - expected_set)
        result = {"matched": matched, "missing": missing, "extra": extra}
        self.reports["data_dictionary_comparison"] = result
        return result

    def clean_column_names(self, *, inplace: bool = False, lower: bool = False, replace_spaces_with: str = "_", strip: bool = True) -> pd.DataFrame | "KStatProfiler":
        target = self if inplace else self.copy()
        new_columns = []
        seen = set()
        for col in target.df.columns:
            new_col = str(col)
            if strip:
                new_col = new_col.strip()
            new_col = re.sub(r"\s+", replace_spaces_with, new_col)
            if lower:
                new_col = new_col.lower()
            candidate = new_col
            suffix = 1
            while candidate in seen:
                suffix += 1
                candidate = f"{new_col}{replace_spaces_with}{suffix}"
            seen.add(candidate)
            new_columns.append(candidate)
        target.df.columns = new_columns
        target._refresh_after_mutation()
        return target if inplace else target.df

    def strip_string_values(self, *, columns: Optional[Iterable[str]] = None, inplace: bool = False) -> pd.DataFrame | "KStatProfiler":
        target = self if inplace else self.copy()
        cols = list(columns) if columns is not None else target.df.select_dtypes(include=["object", "string"]).columns.tolist()
        for col in cols:
            if col not in target.df.columns:
                raise KeyError(f"Column {col!r} not found in DataFrame.")
            target.df[col] = target.df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
        target._refresh_after_mutation()
        return target if inplace else target.df

    def standardize_missing_tokens(self, *, columns: Optional[Iterable[str]] = None, missing_tokens: Optional[Iterable[str]] = None, inplace: bool = False) -> pd.DataFrame | "KStatProfiler":
        tokens = {str(x).strip().lower() for x in (missing_tokens or self.DEFAULT_MISSING_TOKENS)}
        target = self if inplace else self.copy()
        cols = list(columns) if columns is not None else target.df.columns.tolist()
        for col in cols:
            if is_object_dtype(target.df[col]) or is_string_dtype(target.df[col]):
                target.df[col] = target.df[col].map(lambda x: np.nan if isinstance(x, str) and x.strip().lower() in tokens else x)
        target._refresh_after_mutation()
        return target if inplace else target.df

    def drop_duplicate_rows(self, *, subset: Optional[Iterable[str]] = None, keep: str = "first", inplace: bool = False) -> pd.DataFrame | "KStatProfiler":
        if keep not in {"first", "last", False}:
            raise ValueError("keep must be one of {'first', 'last', False}.")
        target = self if inplace else self.copy()
        subset_list = list(subset) if subset is not None else None
        target.df = target.df.drop_duplicates(subset=subset_list, keep=keep).reset_index(drop=True)
        target._refresh_after_mutation()
        return target if inplace else target.df

    def coerce_recommended_dtypes(self, *, inplace: bool = False, numeric_threshold: float = 0.95, datetime_threshold: float = 0.95, boolean_threshold: float = 0.98, convert_to_category: bool = False) -> pd.DataFrame | "KStatProfiler":
        audit = self.dtype_audit(numeric_threshold=numeric_threshold, datetime_threshold=datetime_threshold, boolean_threshold=boolean_threshold)
        target = self if inplace else self.copy()
        for _, row in audit.iterrows():
            col = row["column"]
            recommended = row["recommended_dtype"]
            if recommended == "numeric" and not is_numeric_dtype(target.df[col].dtype):
                target.df[col] = pd.to_numeric(target.df[col].astype(str).str.replace(",", "", regex=False), errors="coerce")
            elif recommended == "datetime64[ns]" and not is_datetime64_any_dtype(target.df[col].dtype):
                target.df[col] = pd.to_datetime(target.df[col], errors="coerce")
            elif recommended == "boolean" and not is_bool_dtype(target.df[col].dtype):
                mapper = {"true": True, "t": True, "yes": True, "y": True, "1": True, "false": False, "f": False, "no": False, "n": False, "0": False}
                target.df[col] = target.df[col].astype("string").str.strip().str.lower().map(mapper).astype("boolean")
            elif recommended == "object" and convert_to_category:
                nunique = target.df[col].nunique(dropna=True)
                if nunique > 0 and nunique <= max(50, len(target.df) * 0.05):
                    target.df[col] = target.df[col].astype("category")
        target._refresh_after_mutation()
        return target if inplace else target.df

    def prepare_for_eda(self, *, inplace: bool = False, clean_column_names: bool = True, lower_column_names: bool = False, strip_string_values: bool = True, standardize_missing_tokens: bool = True, drop_duplicate_rows: bool = False, coerce_dtypes: bool = False) -> pd.DataFrame | "KStatProfiler":
        target = self if inplace else self.copy()
        if clean_column_names:
            target.clean_column_names(inplace=True, lower=lower_column_names)
        if strip_string_values:
            target.strip_string_values(inplace=True)
        if standardize_missing_tokens:
            target.standardize_missing_tokens(inplace=True)
        if drop_duplicate_rows:
            target.drop_duplicate_rows(inplace=True)
        if coerce_dtypes:
            target.coerce_recommended_dtypes(inplace=True)
        return target if inplace else target.df

    def copy(self) -> "KStatProfiler":
        return KStatProfiler(self.df.copy(deep=True), dataset_name=self.dataset_name, depth=self.depth, runtime=self.runtime, copy_df=False, sample_values=self.sample_values, max_similarity_checks=self.max_similarity_checks)
