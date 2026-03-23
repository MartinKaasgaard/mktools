from __future__ import annotations

import pandas as pd

from mktools.kstat import DateSeriesValidator, KStatProfiler


def test_smoke() -> None:
    df = pd.DataFrame(
        {
            "id": [1, 2, 2],
            "value": ["1", "2", None],
            "date": ["2024-01-01", "bad-date", None],
            "flag": ["yes", "no", None],
        }
    )
    profiler = KStatProfiler(df, dataset_name="smoke", depth="deep")
    reports = profiler.profile()
    assert reports["missingness"].shape[0] == df.shape[1]
    assert "recommended_dtype" in reports["dtype_audit"].columns
    cleaned = profiler.prepare_for_eda(coerce_dtypes=True, inplace=False)
    assert "value" in cleaned.columns
    _, report = DateSeriesValidator(date_format="%Y-%m-%d", allow_future=True).validate(df["date"], raise_on_error=False)
    assert report.invalid_rows == 1
