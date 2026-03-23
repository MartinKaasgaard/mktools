from __future__ import annotations

import difflib
import logging
import re
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import pandas as pd
from pandas.api.types import (
    CategoricalDtype,
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EDAHandoff:
    dataset_name: str
    original_shape: tuple[int, int]
    prepared_shape: tuple[int, int]
    feature_catalog: pd.DataFrame
    issue_register: pd.DataFrame
    recommended_actions: pd.DataFrame
    target_candidates: pd.DataFrame
    identifier_candidates: pd.DataFrame
    prep_report: pd.DataFrame
    dtype_conversion_candidates: pd.DataFrame
    prepared_df: Optional[pd.DataFrame] = None

    def to_dict(self, *, include_prepared_df: bool = False) -> dict[str, Any]:
        payload = {
            "dataset_name": self.dataset_name,
            "original_shape": self.original_shape,
            "prepared_shape": self.prepared_shape,
            "feature_catalog": self.feature_catalog.copy(),
            "issue_register": self.issue_register.copy(),
            "recommended_actions": self.recommended_actions.copy(),
            "target_candidates": self.target_candidates.copy(),
            "identifier_candidates": self.identifier_candidates.copy(),
            "prep_report": self.prep_report.copy(),
            "dtype_conversion_candidates": self.dtype_conversion_candidates.copy(),
        }
        if include_prepared_df:
            payload["prepared_df"] = None if self.prepared_df is None else self.prepared_df.copy(deep=True)
        return payload


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
    DEFAULT_ID_NAME_HINTS = ("id", "uuid", "guid", "key", "code", "number", "no")

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
        self.runtime.log(
            f"Initialized KStatProfiler(name={self.dataset_name!r}, depth={self.depth.value}, shape={self.df.shape})",
            level=logging.INFO,
        )

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

    def _ensure_columns_exist(self, columns: Iterable[str], *, label: str = "columns") -> list[str]:
        cols = list(columns)
        missing = [col for col in cols if col not in self.df.columns]
        if missing:
            raise KeyError(f"Unknown {label}: {missing}")
        return cols

    @staticmethod
    def _safe_divide(numerator: float, denominator: float) -> float:
        return 0.0 if denominator == 0 else float(numerator / denominator)

    @staticmethod
    def _top_values(series: pd.Series, n: int = 5) -> list[Any]:
        if n <= 0:
            return []
        s = series.dropna()
        if s.empty:
            return []
        return s.value_counts(dropna=False).head(n).index.tolist()

    @staticmethod
    def _series_python_type_names(series: pd.Series) -> list[str]:
        names = sorted({type(x).__name__ for x in series.dropna().tolist()})
        return names

    @staticmethod
    def _name_has_hint(column: str, hints: Iterable[str]) -> bool:
        normalized = str(column).strip().lower()
        parts = re.split(r"[^a-z0-9]+", normalized)
        return any(h in normalized or h in parts for h in hints)

    @staticmethod
    def _coerce_numeric_string(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")

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
            "mixed_types": self.mixed_type_summary(),
            "identifier_candidates": self.identifier_candidate_summary(),
            "target_candidates": self.target_candidate_summary(),
            "issue_register": self.issue_register(),
            "recommended_actions": self.recommended_actions(),
        }
        if self.depth in {AnalysisDepth.STANDARD, AnalysisDepth.DEEP}:
            bundle["descriptive_stats"] = self.descriptive_stats()
            bundle["outliers"] = self.outlier_summary()
            bundle["cardinality"] = self.cardinality_summary()
            bundle["constant_columns"] = self.constant_columns()
            bundle["correlation"] = self.correlation_summary()
            bundle["numeric_distribution"] = self.numeric_distribution_summary()
            bundle["missingness_patterns"] = self.missingness_patterns()
            bundle["rare_levels"] = self.rare_levels_summary()
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
        return self._cache(
            "overview",
            DatasetOverview(
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
            ),
        )

    def column_summary(self, sample_values: Optional[int] = None) -> pd.DataFrame:
        sample_values = self.sample_values if sample_values is None else self._validate_positive_int(sample_values, "sample_values", minimum=1)
        rows = []
        total_rows = max(len(self.df), 1)
        for col in self.df.columns:
            series = self.df[col]
            missing = int(series.isna().sum())
            non_null = int(series.notna().sum())
            unique_non_null = int(series.nunique(dropna=True))
            unique_ratio = self._safe_divide(unique_non_null, max(non_null, 1))
            rows.append(
                {
                    "column": col,
                    "category": FeatureCatalog._infer_category(series),
                    "dtype": str(series.dtype),
                    "rows": len(series),
                    "non_null": non_null,
                    "missing": missing,
                    "missing_pct": round((missing / total_rows) * 100, 4),
                    "unique_non_null": unique_non_null,
                    "unique_ratio_non_null": round(unique_ratio, 4),
                    "memory_mb": round(series.memory_usage(deep=True) / (1024**2), 6),
                    "sample_values": series.dropna().astype(str).drop_duplicates().head(sample_values).tolist(),
                }
            )
        result = pd.DataFrame(rows).sort_values(["missing_pct", "column"], ascending=[False, True]).reset_index(drop=True)
        return self._cache("column_summary", result)

    def missingness_summary(self) -> pd.DataFrame:
        total_rows = max(len(self.df), 1)
        rows = []
        for col in self.df.columns:
            missing_count = int(self.df[col].isna().sum())
            rows.append(
                {
                    "column": col,
                    "missing_count": missing_count,
                    "missing_pct": round((missing_count / total_rows) * 100, 4),
                    "all_missing": missing_count == len(self.df) and len(self.df) > 0,
                }
            )
        result = pd.DataFrame(rows).sort_values(["missing_count", "column"], ascending=[False, True]).reset_index(drop=True)
        return self._cache("missingness", result)

    def missingness_patterns(self, *, top_n: int = 10) -> pd.DataFrame:
        self._validate_positive_int(top_n, "top_n", minimum=1)
        cols_with_missing = [col for col in self.df.columns if self.df[col].isna().any()]
        if not cols_with_missing:
            return self._cache("missingness_patterns", pd.DataFrame(columns=["missing_columns", "row_count", "row_pct"]))
        mask = self.df[cols_with_missing].isna()
        patterns = (
            mask.apply(lambda row: tuple(col for col, is_missing in zip(cols_with_missing, row.tolist()) if is_missing), axis=1)
            .value_counts()
            .head(top_n)
            .reset_index()
        )
        patterns.columns = ["missing_columns", "row_count"]
        patterns["row_pct"] = (patterns["row_count"] / max(len(self.df), 1) * 100).round(4)
        return self._cache("missingness_patterns", patterns)

    def duplicate_summary(self, subset: Optional[Iterable[str]] = None) -> pd.DataFrame:
        subset_list = list(subset) if subset is not None else None
        if subset_list is not None:
            self._ensure_columns_exist(subset_list, label="subset columns")
        duplicate_mask = self.df.duplicated(subset=subset_list, keep=False)
        duplicate_count = int(self.df.duplicated(subset=subset_list).sum())
        result = pd.DataFrame(
            [
                {
                    "scope": "all_columns" if subset_list is None else ",".join(subset_list),
                    "duplicate_rows": duplicate_count,
                    "duplicate_row_pct": round((duplicate_count / max(len(self.df), 1)) * 100, 4),
                    "rows_in_duplicate_groups": int(duplicate_mask.sum()),
                }
            ]
        )
        return self._cache("duplicates", result)

    @staticmethod
    def _numeric_parse_rate(series: pd.Series) -> float:
        non_null = series.dropna()
        if non_null.empty:
            return 1.0
        parsed = KStatProfiler._coerce_numeric_string(non_null)
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

    def dtype_audit(
        self,
        *,
        numeric_threshold: float = 0.95,
        datetime_threshold: float = 0.95,
        boolean_threshold: float = 0.98,
    ) -> pd.DataFrame:
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
            rows.append(
                {
                    "column": col,
                    "current_dtype": current_dtype,
                    "recommended_dtype": recommended_dtype,
                    "numeric_parse_rate": round(numeric_rate, 4),
                    "datetime_parse_rate": round(datetime_rate, 4),
                    "boolean_parse_rate": round(boolean_rate, 4),
                    "flag_inconsistency": recommended_dtype != current_dtype and not (
                        recommended_dtype == "numeric" and current_dtype.startswith(("int", "float"))
                    ),
                }
            )
        result = pd.DataFrame(rows).sort_values(["flag_inconsistency", "column"], ascending=[False, True]).reset_index(drop=True)
        return self._cache("dtype_audit", result)

    def mixed_type_summary(self) -> pd.DataFrame:
        rows = []
        for col in self.df.columns:
            series = self.df[col]
            if not (is_object_dtype(series.dtype) or is_string_dtype(series.dtype) or isinstance(series.dtype, CategoricalDtype)):
                continue
            type_names = self._series_python_type_names(series)
            rows.append(
                {
                    "column": col,
                    "dtype": str(series.dtype),
                    "python_type_count": len(type_names),
                    "python_types": type_names[:10],
                    "flag_mixed_types": len(type_names) > 1,
                }
            )
        result = pd.DataFrame(rows)
        if not result.empty:
            result = result.sort_values(["flag_mixed_types", "python_type_count", "column"], ascending=[False, False, True]).reset_index(drop=True)
        return self._cache("mixed_types", result)

    def cardinality_summary(self, high_cardinality_threshold: int = 50) -> pd.DataFrame:
        self._validate_positive_int(high_cardinality_threshold, "high_cardinality_threshold", minimum=1)
        object_like = self.df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        rows = [
            {
                "column": col,
                "unique_non_null": int(self.df[col].nunique(dropna=True)),
                "high_cardinality": int(self.df[col].nunique(dropna=True)) > high_cardinality_threshold,
            }
            for col in object_like
        ]
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
            rows.append(
                {
                    "column": col,
                    "top_value": vc.index[0],
                    "top_ratio": round(top_ratio, 4),
                    "is_constant": vc.shape[0] == 1,
                    "is_near_constant": top_ratio >= near_constant_threshold,
                }
            )
        result = pd.DataFrame(rows).sort_values(["is_constant", "top_ratio", "column"], ascending=[False, False, True]).reset_index(drop=True)
        return self._cache("constant_columns", result)

    def rare_levels_summary(self, *, max_levels: int = 50, rare_level_threshold: float = 0.01) -> pd.DataFrame:
        self._validate_positive_int(max_levels, "max_levels", minimum=1)
        self._validate_ratio(rare_level_threshold, "rare_level_threshold")
        rows = []
        for col in self.df.select_dtypes(include=["object", "string", "category", "boolean"]).columns:
            s = self.df[col].dropna()
            if s.empty:
                continue
            vc = s.value_counts(normalize=True, dropna=False)
            if len(vc) > max_levels:
                continue
            rare_mask = vc < rare_level_threshold
            rows.append(
                {
                    "column": col,
                    "levels": int(len(vc)),
                    "rare_levels": int(rare_mask.sum()),
                    "rare_row_pct": round(float(vc[rare_mask].sum() * 100), 4),
                    "top_levels": vc.head(5).to_dict(),
                }
            )
        result = pd.DataFrame(rows)
        if not result.empty:
            result = result.sort_values(["rare_levels", "rare_row_pct", "column"], ascending=[False, False, True]).reset_index(drop=True)
        return self._cache("rare_levels", result)

    def descriptive_stats(self) -> pd.DataFrame:
        if self.df.empty:
            return self._cache("descriptive_stats", pd.DataFrame())
        try:
            result = self.df.describe(include="all", datetime_is_numeric=True).T
        except TypeError:
            result = self.df.describe(include="all").T
        return self._cache("descriptive_stats", result)

    def numeric_distribution_summary(self) -> pd.DataFrame:
        rows = []
        for col in self.df.select_dtypes(include=[np.number]).columns:
            s = pd.to_numeric(self.df[col], errors="coerce")
            non_null = s.dropna()
            if non_null.empty:
                continue
            rows.append(
                {
                    "column": col,
                    "non_null": int(non_null.shape[0]),
                    "zeros_pct": round(float((non_null == 0).mean() * 100), 4),
                    "negative_pct": round(float((non_null < 0).mean() * 100), 4),
                    "infinite_count": int(np.isinf(non_null).sum()),
                    "skewness": round(float(non_null.skew()), 6) if non_null.shape[0] > 2 else np.nan,
                    "kurtosis": round(float(non_null.kurtosis()), 6) if non_null.shape[0] > 3 else np.nan,
                    "min": float(non_null.min()),
                    "median": float(non_null.median()),
                    "max": float(non_null.max()),
                }
            )
        result = pd.DataFrame(rows)
        if not result.empty:
            result = result.sort_values(["column"]).reset_index(drop=True)
        return self._cache("numeric_distribution", result)

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
            rows.append(
                {
                    "column": col,
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "lower_limit": lower,
                    "upper_limit": upper,
                    "outlier_count": int(mask.sum()),
                    "outlier_pct": round(float(mask.mean() * 100), 4),
                }
            )
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

    def strong_correlations(self, *, threshold: float = 0.8, method: str = "pearson") -> pd.DataFrame:
        self._validate_ratio(abs(float(threshold)), "threshold")
        corr = self.correlation_summary(method=method)
        if corr.empty:
            return pd.DataFrame(columns=["left_column", "right_column", "correlation", "abs_correlation"])
        rows = []
        cols = list(corr.columns)
        for i, left in enumerate(cols):
            for right in cols[i + 1 :]:
                value = corr.loc[left, right]
                if pd.notna(value) and abs(float(value)) >= threshold:
                    rows.append(
                        {
                            "left_column": left,
                            "right_column": right,
                            "correlation": float(value),
                            "abs_correlation": abs(float(value)),
                        }
                    )
        result = pd.DataFrame(rows)
        if not result.empty:
            result = result.sort_values(["abs_correlation", "left_column", "right_column"], ascending=[False, True, True]).reset_index(drop=True)
        self.reports["strong_correlations"] = result
        return result

    def string_quality_summary(self) -> pd.DataFrame:
        object_like = self.df.select_dtypes(include=["object", "string"]).columns.tolist()
        rows = []
        for col in object_like:
            s = self.df[col].astype("string")
            non_null = s.dropna()
            stripped = non_null.str.strip()
            normalized = stripped.str.lower()
            rows.append(
                {
                    "column": col,
                    "leading_or_trailing_spaces": int((non_null != stripped).sum()) if not non_null.empty else 0,
                    "empty_strings": int((stripped == "").sum()) if not non_null.empty else 0,
                    "case_variant_collisions": int(non_null.nunique(dropna=True) - normalized.nunique(dropna=True)) if not non_null.empty else 0,
                }
            )
        result = pd.DataFrame(rows)
        if not result.empty:
            result = result.sort_values(["leading_or_trailing_spaces", "empty_strings", "column"], ascending=[False, False, True]).reset_index(drop=True)
        return self._cache("string_quality", result)

    def detect_possible_value_inconsistencies(
        self,
        column: Optional[str] = None,
        exempt_column: Optional[str] = None,
        cutoff: float = 0.85,
    ) -> pd.DataFrame:
        self._validate_ratio(cutoff, "cutoff")
        object_columns = self.df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        if column is not None and exempt_column is not None:
            raise ValueError("Use either 'column' or 'exempt_column', not both.")
        if column is not None:
            self._ensure_columns_exist([column], label="column")
            columns_to_check = [column]
        elif exempt_column is not None:
            self._ensure_columns_exist([exempt_column], label="exempt_column")
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
            for right in cols[i + 1 :]:
                if self.df[left].equals(self.df[right]):
                    pairs.append({"left_column": left, "right_column": right})
        return self._cache("identical_columns", pd.DataFrame(pairs))

    def datetime_summary(self) -> pd.DataFrame:
        dt_cols = self.df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
        rows = []
        for col in dt_cols:
            s = self.df[col].dropna()
            rows.append(
                {
                    "column": col,
                    "min": s.min() if not s.empty else pd.NaT,
                    "max": s.max() if not s.empty else pd.NaT,
                    "range_days": (s.max() - s.min()).days if len(s) > 1 else np.nan,
                    "missing_count": int(self.df[col].isna().sum()),
                }
            )
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

    def identifier_candidate_summary(
        self,
        *,
        unique_ratio_threshold: float = 0.98,
        non_null_ratio_threshold: float = 0.8,
        name_hints: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        self._validate_ratio(unique_ratio_threshold, "unique_ratio_threshold")
        self._validate_ratio(non_null_ratio_threshold, "non_null_ratio_threshold")
        hints = tuple(name_hints or self.DEFAULT_ID_NAME_HINTS)
        rows = []
        total_rows = max(len(self.df), 1)
        for col in self.df.columns:
            s = self.df[col]
            non_null = int(s.notna().sum())
            unique_non_null = int(s.nunique(dropna=True))
            unique_ratio = self._safe_divide(unique_non_null, max(non_null, 1))
            non_null_ratio = self._safe_divide(non_null, total_rows)
            name_hint = self._name_has_hint(col, hints)
            monotonic = bool(is_numeric_dtype(s.dtype) and s.dropna().is_monotonic_increasing) if non_null > 0 else False
            looks_identifier = (
                non_null_ratio >= non_null_ratio_threshold
                and unique_ratio >= unique_ratio_threshold
                and (name_hint or monotonic or unique_non_null == non_null)
            )
            rows.append(
                {
                    "column": col,
                    "dtype": str(s.dtype),
                    "non_null": non_null,
                    "unique_non_null": unique_non_null,
                    "unique_ratio": round(unique_ratio, 4),
                    "non_null_ratio": round(non_null_ratio, 4),
                    "name_hint": name_hint,
                    "monotonic_increasing": monotonic,
                    "looks_identifier": looks_identifier,
                }
            )
        result = pd.DataFrame(rows).sort_values(["looks_identifier", "unique_ratio", "column"], ascending=[False, False, True]).reset_index(drop=True)
        return self._cache("identifier_candidates", result)

    def target_candidate_summary(
        self,
        *,
        max_class_labels: int = 20,
        include_numeric_regression: bool = True,
    ) -> pd.DataFrame:
        self._validate_positive_int(max_class_labels, "max_class_labels", minimum=2)
        id_candidates = self.identifier_candidate_summary()
        id_like = set(id_candidates.loc[id_candidates["looks_identifier"], "column"].tolist())
        rows = []
        for col in self.df.columns:
            if col in id_like:
                continue
            s = self.df[col]
            non_null = s.dropna()
            unique_non_null = int(non_null.nunique(dropna=True))
            category = FeatureCatalog._infer_category(s)
            if category in {"boolean", "categorical"} and 2 <= unique_non_null <= max_class_labels:
                rows.append(
                    {
                        "column": col,
                        "problem_type": "classification",
                        "category": category,
                        "unique_non_null": unique_non_null,
                        "candidate_score": 1.0 if category == "boolean" else 0.9,
                    }
                )
            elif category == "numerical" and include_numeric_regression and unique_non_null > max_class_labels:
                rows.append(
                    {
                        "column": col,
                        "problem_type": "regression",
                        "category": category,
                        "unique_non_null": unique_non_null,
                        "candidate_score": 0.75,
                    }
                )
            elif category == "numerical" and 2 <= unique_non_null <= max_class_labels:
                rows.append(
                    {
                        "column": col,
                        "problem_type": "classification_or_ordinal",
                        "category": category,
                        "unique_non_null": unique_non_null,
                        "candidate_score": 0.65,
                    }
                )
        result = pd.DataFrame(rows)
        if not result.empty:
            result = result.sort_values(["candidate_score", "unique_non_null", "column"], ascending=[False, True, True]).reset_index(drop=True)
        return self._cache("target_candidates", result)

    def issue_register(
        self,
        *,
        missing_warn_pct: float = 20.0,
        missing_critical_pct: float = 50.0,
        outlier_warn_pct: float = 5.0,
        high_cardinality_threshold: int = 50,
        near_constant_threshold: float = 0.99,
    ) -> pd.DataFrame:
        if missing_warn_pct < 0 or missing_critical_pct < 0 or outlier_warn_pct < 0:
            raise ValueError("percentage thresholds must be non-negative.")
        issues: list[dict[str, Any]] = []

        missingness = self.missingness_summary()
        for _, row in missingness.iterrows():
            if row["all_missing"]:
                issues.append({
                    "severity": "critical",
                    "scope": "column",
                    "column": row["column"],
                    "issue_type": "all_missing",
                    "metric": "missing_pct",
                    "value": row["missing_pct"],
                    "recommendation": "Drop the column or source values before EDA.",
                })
            elif row["missing_pct"] >= missing_critical_pct:
                issues.append({
                    "severity": "high",
                    "scope": "column",
                    "column": row["column"],
                    "issue_type": "high_missingness",
                    "metric": "missing_pct",
                    "value": row["missing_pct"],
                    "recommendation": "Assess missingness mechanism and consider imputation or exclusion.",
                })
            elif row["missing_pct"] >= missing_warn_pct:
                issues.append({
                    "severity": "medium",
                    "scope": "column",
                    "column": row["column"],
                    "issue_type": "moderate_missingness",
                    "metric": "missing_pct",
                    "value": row["missing_pct"],
                    "recommendation": "Review whether the column should be imputed before modeling.",
                })

        dup = self.duplicate_summary().iloc[0]
        if dup["duplicate_rows"] > 0:
            issues.append({
                "severity": "medium",
                "scope": "dataset",
                "column": None,
                "issue_type": "duplicate_rows",
                "metric": "duplicate_rows",
                "value": int(dup["duplicate_rows"]),
                "recommendation": "Inspect duplicates and deduplicate deliberately before EDA or modeling.",
            })

        dtype_audit = self.dtype_audit()
        for _, row in dtype_audit.loc[dtype_audit["flag_inconsistency"]].iterrows():
            issues.append({
                "severity": "medium",
                "scope": "column",
                "column": row["column"],
                "issue_type": "dtype_inconsistency",
                "metric": "recommended_dtype",
                "value": row["recommended_dtype"],
                "recommendation": f"Consider coercing to {row['recommended_dtype']} before downstream analysis.",
            })

        mixed = self.mixed_type_summary()
        if not mixed.empty:
            for _, row in mixed.loc[mixed["flag_mixed_types"]].iterrows():
                issues.append({
                    "severity": "medium",
                    "scope": "column",
                    "column": row["column"],
                    "issue_type": "mixed_python_types",
                    "metric": "python_type_count",
                    "value": int(row["python_type_count"]),
                    "recommendation": "Normalize mixed values or coerce to a single logical dtype.",
                })

        const = self.constant_columns(near_constant_threshold=near_constant_threshold)
        for _, row in const.loc[const["is_constant"]].iterrows():
            issues.append({
                "severity": "high",
                "scope": "column",
                "column": row["column"],
                "issue_type": "constant_column",
                "metric": "top_ratio",
                "value": row["top_ratio"],
                "recommendation": "Drop constant columns; they do not add analytical value.",
            })
        for _, row in const.loc[(~const["is_constant"]) & (const["is_near_constant"])].iterrows():
            issues.append({
                "severity": "low",
                "scope": "column",
                "column": row["column"],
                "issue_type": "near_constant_column",
                "metric": "top_ratio",
                "value": row["top_ratio"],
                "recommendation": "Check whether this feature is informative enough to keep.",
            })

        outliers = self.outlier_summary()
        if not outliers.empty:
            for _, row in outliers.loc[outliers["outlier_pct"] >= outlier_warn_pct].iterrows():
                issues.append({
                    "severity": "low" if row["outlier_pct"] < 15 else "medium",
                    "scope": "column",
                    "column": row["column"],
                    "issue_type": "outliers",
                    "metric": "outlier_pct",
                    "value": row["outlier_pct"],
                    "recommendation": "Inspect distribution shape and decide between winsorization, transformation, or retention.",
                })

        cardinality = self.cardinality_summary(high_cardinality_threshold=high_cardinality_threshold)
        if not cardinality.empty:
            for _, row in cardinality.loc[cardinality["high_cardinality"]].iterrows():
                issues.append({
                    "severity": "low",
                    "scope": "column",
                    "column": row["column"],
                    "issue_type": "high_cardinality",
                    "metric": "unique_non_null",
                    "value": int(row["unique_non_null"]),
                    "recommendation": "Review whether this behaves like free text, an identifier, or needs grouping.",
                })

        string_quality = self.string_quality_summary()
        if not string_quality.empty:
            for _, row in string_quality.iterrows():
                if row["leading_or_trailing_spaces"] > 0:
                    issues.append({
                        "severity": "low",
                        "scope": "column",
                        "column": row["column"],
                        "issue_type": "string_whitespace",
                        "metric": "leading_or_trailing_spaces",
                        "value": int(row["leading_or_trailing_spaces"]),
                        "recommendation": "Strip string values before EDA.",
                    })
                if row["case_variant_collisions"] > 0:
                    issues.append({
                        "severity": "low",
                        "scope": "column",
                        "column": row["column"],
                        "issue_type": "case_variant_collisions",
                        "metric": "case_variant_collisions",
                        "value": int(row["case_variant_collisions"]),
                        "recommendation": "Normalize case if categories should be treated identically.",
                    })

        ids = self.identifier_candidate_summary()
        for _, row in ids.loc[ids["looks_identifier"]].iterrows():
            issues.append({
                "severity": "low",
                "scope": "column",
                "column": row["column"],
                "issue_type": "identifier_candidate",
                "metric": "unique_ratio",
                "value": row["unique_ratio"],
                "recommendation": "Treat as an identifier unless domain knowledge says otherwise.",
            })

        result = pd.DataFrame(issues)
        if result.empty:
            result = pd.DataFrame(columns=["severity", "scope", "column", "issue_type", "metric", "value", "recommendation"])
        else:
            severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            result["_severity_order"] = result["severity"].map(severity_order).fillna(99)
            result = result.sort_values(["_severity_order", "column", "issue_type"], ascending=[True, True, True]).drop(columns="_severity_order").reset_index(drop=True)
        return self._cache("issue_register", result)

    def recommended_actions(self) -> pd.DataFrame:
        issues = self.issue_register()
        if issues.empty:
            result = pd.DataFrame(columns=["priority", "column", "action", "rationale", "related_issues"])
            return self._cache("recommended_actions", result)
        action_rows = []
        grouped = issues.groupby(["column", "recommendation"], dropna=False)
        for (column, recommendation), grp in grouped:
            severities = grp["severity"].tolist()
            priority = "critical" if "critical" in severities else "high" if "high" in severities else "medium" if "medium" in severities else "low"
            action_rows.append({
                "priority": priority,
                "column": column,
                "action": recommendation,
                "rationale": "; ".join(sorted(set(grp["issue_type"].astype(str).tolist()))),
                "related_issues": int(len(grp)),
            })
        result = pd.DataFrame(action_rows)
        if not result.empty:
            severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            result["_priority_order"] = result["priority"].map(severity_order).fillna(99)
            result = result.sort_values(["_priority_order", "column", "action"], ascending=[True, True, True]).drop(columns="_priority_order").reset_index(drop=True)
        return self._cache("recommended_actions", result)

    def _apply_clean_column_names(
        self,
        *,
        lower: bool = False,
        replace_spaces_with: str = "_",
        strip: bool = True,
    ) -> dict[str, str]:
        new_columns = []
        mapping: dict[str, str] = {}
        seen = set()
        for col in self.df.columns:
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
            mapping[str(col)] = candidate
        self.df.columns = new_columns
        return mapping

    def clean_column_names(
        self,
        *,
        inplace: bool = False,
        lower: bool = False,
        replace_spaces_with: str = "_",
        strip: bool = True,
    ) -> pd.DataFrame | "KStatProfiler":
        target = self if inplace else self.copy()
        target._apply_clean_column_names(lower=lower, replace_spaces_with=replace_spaces_with, strip=strip)
        target._refresh_after_mutation()
        return target if inplace else target.df

    def strip_string_values(self, *, columns: Optional[Iterable[str]] = None, inplace: bool = False) -> pd.DataFrame | "KStatProfiler":
        target = self if inplace else self.copy()
        cols = list(columns) if columns is not None else target.df.select_dtypes(include=["object", "string"]).columns.tolist()
        target._ensure_columns_exist(cols, label="columns")
        for col in cols:
            target.df[col] = target.df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
        target._refresh_after_mutation()
        return target if inplace else target.df

    def standardize_missing_tokens(
        self,
        *,
        columns: Optional[Iterable[str]] = None,
        missing_tokens: Optional[Iterable[str]] = None,
        inplace: bool = False,
    ) -> pd.DataFrame | "KStatProfiler":
        tokens = {str(x).strip().lower() for x in (missing_tokens or self.DEFAULT_MISSING_TOKENS)}
        target = self if inplace else self.copy()
        cols = list(columns) if columns is not None else target.df.columns.tolist()
        target._ensure_columns_exist(cols, label="columns")
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
        if subset_list is not None:
            target._ensure_columns_exist(subset_list, label="subset columns")
        target.df = target.df.drop_duplicates(subset=subset_list, keep=keep).reset_index(drop=True)
        target._refresh_after_mutation()
        return target if inplace else target.df

    def _coerce_recommended_dtypes_with_report(
        self,
        *,
        numeric_threshold: float = 0.95,
        datetime_threshold: float = 0.95,
        boolean_threshold: float = 0.98,
        convert_to_category: bool = False,
    ) -> pd.DataFrame:
        audit = self.dtype_audit(
            numeric_threshold=numeric_threshold,
            datetime_threshold=datetime_threshold,
            boolean_threshold=boolean_threshold,
        )
        changes: list[dict[str, Any]] = []
        for _, row in audit.iterrows():
            col = row["column"]
            before = str(self.df[col].dtype)
            recommended = row["recommended_dtype"]
            if recommended == "numeric" and not is_numeric_dtype(self.df[col].dtype):
                self.df[col] = self._coerce_numeric_string(self.df[col])
            elif recommended == "datetime64[ns]" and not is_datetime64_any_dtype(self.df[col].dtype):
                self.df[col] = pd.to_datetime(self.df[col], errors="coerce")
            elif recommended == "boolean" and not is_bool_dtype(self.df[col].dtype):
                mapper = {"true": True, "t": True, "yes": True, "y": True, "1": True, "false": False, "f": False, "no": False, "n": False, "0": False}
                self.df[col] = self.df[col].astype("string").str.strip().str.lower().map(mapper).astype("boolean")
            elif recommended == "object" and convert_to_category:
                nunique = self.df[col].nunique(dropna=True)
                if nunique > 0 and nunique <= max(50, int(len(self.df) * 0.05)):
                    self.df[col] = self.df[col].astype("category")
            after = str(self.df[col].dtype)
            if before != after:
                changes.append({"column": col, "from_dtype": before, "to_dtype": after, "recommended_dtype": recommended})
        return pd.DataFrame(changes)

    def coerce_recommended_dtypes(
        self,
        *,
        inplace: bool = False,
        numeric_threshold: float = 0.95,
        datetime_threshold: float = 0.95,
        boolean_threshold: float = 0.98,
        convert_to_category: bool = False,
    ) -> pd.DataFrame | "KStatProfiler":
        target = self if inplace else self.copy()
        target._coerce_recommended_dtypes_with_report(
            numeric_threshold=numeric_threshold,
            datetime_threshold=datetime_threshold,
            boolean_threshold=boolean_threshold,
            convert_to_category=convert_to_category,
        )
        target._refresh_after_mutation()
        return target if inplace else target.df

    def prepare_for_eda_bundle(
        self,
        *,
        clean_column_names: bool = True,
        lower_column_names: bool = False,
        strip_string_values: bool = True,
        standardize_missing_tokens: bool = True,
        drop_duplicate_rows: bool = False,
        coerce_dtypes: bool = False,
        convert_to_category: bool = False,
        include_profile: bool = True,
        include_prepared_df: bool = True,
    ) -> EDAHandoff:
        working = self.copy()
        original_shape = tuple(working.df.shape)
        step_rows: list[dict[str, Any]] = []

        if clean_column_names:
            before_cols = list(working.df.columns)
            mapping = working._apply_clean_column_names(lower=lower_column_names)
            changed = before_cols != list(working.df.columns)
            step_rows.append({
                "step": "clean_column_names",
                "changed": changed,
                "detail": mapping,
            })

        if strip_string_values:
            before = working.df.copy(deep=True)
            working.strip_string_values(inplace=True)
            changed_cells = int((before.ne(working.df) & ~(before.isna() & working.df.isna())).sum().sum())
            step_rows.append({"step": "strip_string_values", "changed": changed_cells > 0, "detail": {"changed_cells": changed_cells}})

        if standardize_missing_tokens:
            before_missing = int(working.df.isna().sum().sum())
            working.standardize_missing_tokens(inplace=True)
            after_missing = int(working.df.isna().sum().sum())
            step_rows.append({
                "step": "standardize_missing_tokens",
                "changed": after_missing > before_missing,
                "detail": {"missing_cells_before": before_missing, "missing_cells_after": after_missing},
            })

        if drop_duplicate_rows:
            before_rows = len(working.df)
            working.drop_duplicate_rows(inplace=True)
            after_rows = len(working.df)
            step_rows.append({
                "step": "drop_duplicate_rows",
                "changed": after_rows != before_rows,
                "detail": {"rows_before": before_rows, "rows_after": after_rows, "dropped_rows": before_rows - after_rows},
            })

        conversion_report = pd.DataFrame(columns=["column", "from_dtype", "to_dtype", "recommended_dtype"])
        if coerce_dtypes:
            conversion_report = working._coerce_recommended_dtypes_with_report(convert_to_category=convert_to_category)
            working._refresh_after_mutation()
            step_rows.append({
                "step": "coerce_recommended_dtypes",
                "changed": not conversion_report.empty,
                "detail": {"dtype_changes": int(conversion_report.shape[0])},
            })

        if include_profile:
            working.profile(refresh=True)

        return EDAHandoff(
            dataset_name=working.dataset_name,
            original_shape=original_shape,
            prepared_shape=tuple(working.df.shape),
            feature_catalog=working.catalog.as_frame(),
            issue_register=working.issue_register(),
            recommended_actions=working.recommended_actions(),
            target_candidates=working.target_candidate_summary(),
            identifier_candidates=working.identifier_candidate_summary(),
            prep_report=pd.DataFrame(step_rows),
            dtype_conversion_candidates=conversion_report if not conversion_report.empty else working.dtype_audit().loc[lambda d: d["flag_inconsistency"]].reset_index(drop=True),
            prepared_df=working.df.copy(deep=True) if include_prepared_df else None,
        )

    def eda_handoff(self, **kwargs: Any) -> EDAHandoff:
        return self.prepare_for_eda_bundle(**kwargs)

    def prepare_for_eda(
        self,
        *,
        inplace: bool = False,
        clean_column_names: bool = True,
        lower_column_names: bool = False,
        strip_string_values: bool = True,
        standardize_missing_tokens: bool = True,
        drop_duplicate_rows: bool = False,
        coerce_dtypes: bool = False,
    ) -> pd.DataFrame | "KStatProfiler":
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
        return KStatProfiler(
            self.df.copy(deep=True),
            dataset_name=self.dataset_name,
            depth=self.depth,
            runtime=self.runtime,
            copy_df=False,
            sample_values=self.sample_values,
            max_similarity_checks=self.max_similarity_checks,
        )
