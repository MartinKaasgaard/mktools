from __future__ import annotations

import pandas as pd

from mktools.kstat import EDAHandoff, KStatProfiler


def test_issue_register_and_actions() -> None:
    df = pd.DataFrame(
        {
            "row_id": [1, 2, 3, 4],
            "target": [1, 0, 1, 0],
            "city": ["A", "A ", None, "B"],
            "constant": ["x", "x", "x", "x"],
            "num": [1, 2, 2, 100],
        }
    )
    profiler = KStatProfiler(df, depth="deep")
    issues = profiler.issue_register()
    actions = profiler.recommended_actions()

    assert not issues.empty
    assert "constant_column" in set(issues["issue_type"])
    assert not actions.empty
    assert "action" in actions.columns


def test_eda_handoff_bundle() -> None:
    df = pd.DataFrame(
        {
            "Row ID": [1, 2, 2],
            "amount": ["1,000", "25", None],
            "flag": ["yes", "no", None],
            "city": [" A", "B ", "unknown"],
        }
    )
    profiler = KStatProfiler(df, depth="standard")
    handoff = profiler.eda_handoff(
        lower_column_names=True,
        strip_string_values=True,
        standardize_missing_tokens=True,
        drop_duplicate_rows=True,
        coerce_dtypes=True,
        include_profile=True,
        include_prepared_df=True,
    )

    assert isinstance(handoff, EDAHandoff)
    assert handoff.prepared_df is not None
    assert "row_id" in handoff.prepared_df.columns
    assert handoff.prepared_df["amount"].dtype.kind in {"f", "i"}
    assert not handoff.prep_report.empty


def test_cardinality_summary_all_numeric_returns_empty_schema():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    profiler = KStatProfiler(df, dataset_name="numeric_only", depth="deep")
    result = profiler.cardinality_summary()

    assert list(result.columns) == ["column", "unique_non_null", "high_cardinality"]
    assert result.empty


def test_rare_levels_summary_all_numeric_returns_empty_schema():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    profiler = KStatProfiler(df, dataset_name="numeric_only", depth="deep")
    result = profiler.rare_levels_summary()

    assert list(result.columns) == ["column", "level", "frequency_ratio", "frequency_percent"]
    assert result.empty


def test_finalize_report_empty_rows_preserves_columns():
    result = KStatProfiler._finalize_report(
        rows=[],
        columns=["column", "score"],
        sort_by=["score", "column"],
        ascending=[False, True],
    )

    assert list(result.columns) == ["column", "score"]
    assert result.empty
