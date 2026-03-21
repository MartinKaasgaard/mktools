from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional, Union

import pandas as pd


@dataclass
class DateValidationReport:
    column_name: str
    total_rows: int
    valid_rows: int
    invalid_rows: int
    future_rows: int
    too_early_rows: int
    too_late_rows: int
    parsed_dtype: str

    @property
    def valid_ratio(self) -> float:
        if self.total_rows == 0:
            return 1.0
        return self.valid_rows / self.total_rows


class DateSeriesValidator:
    """
    Safe datetime parser and validator for a single pandas Series.

    Notes
    -----
    - Missing values are preserved.
    - Parsing issues can either raise immediately or be returned in a report.
    - Validation boundaries are inclusive.
    """

    def __init__(
        self,
        date_format: Optional[str] = None,
        allow_future: bool = False,
        min_date: Optional[Union[str, date]] = None,
        max_date: Optional[Union[str, date]] = None,
        dayfirst: bool = False,
        utc: bool = False,
    ) -> None:
        self.date_format = date_format
        self.allow_future = bool(allow_future)
        self.min_date = pd.to_datetime(min_date).date() if min_date is not None else None
        self.max_date = pd.to_datetime(max_date).date() if max_date is not None else None
        self.dayfirst = bool(dayfirst)
        self.utc = bool(utc)

        if self.min_date and self.max_date and self.min_date > self.max_date:
            raise ValueError("min_date cannot be later than max_date.")

    def parse(self, series: pd.Series) -> pd.Series:
        if not isinstance(series, pd.Series):
            raise TypeError("series must be a pandas Series.")
        return pd.to_datetime(
            series,
            format=self.date_format,
            errors="coerce",
            dayfirst=self.dayfirst,
            utc=self.utc,
        )

    def validate(
        self,
        series: pd.Series,
        *,
        raise_on_error: bool = True,
        column_name: Optional[str] = None,
    ) -> tuple[pd.Series, DateValidationReport]:
        if not isinstance(series, pd.Series):
            raise TypeError("series must be a pandas Series.")

        parsed = self.parse(series)
        non_null_mask = series.notna()
        invalid_mask = non_null_mask & parsed.isna()

        normalized_dates = parsed.dt.date
        today = date.today()

        future_mask = normalized_dates > today if not self.allow_future else pd.Series(False, index=series.index)
        too_early_mask = (
            normalized_dates < self.min_date if self.min_date is not None else pd.Series(False, index=series.index)
        )
        too_late_mask = (
            normalized_dates > self.max_date if self.max_date is not None else pd.Series(False, index=series.index)
        )

        report = DateValidationReport(
            column_name=column_name or getattr(series, "name", "unknown"),
            total_rows=int(len(series)),
            valid_rows=int((~invalid_mask & non_null_mask).sum()),
            invalid_rows=int(invalid_mask.sum()),
            future_rows=int(future_mask.sum()),
            too_early_rows=int(too_early_mask.sum()),
            too_late_rows=int(too_late_mask.sum()),
            parsed_dtype=str(parsed.dtype),
        )

        messages: list[str] = []
        if report.invalid_rows:
            bad = series[invalid_mask].drop_duplicates().astype(str).head(5).tolist()
            messages.append(f"Invalid date strings: {bad}")
        if report.future_rows:
            bad = series[future_mask].drop_duplicates().astype(str).head(5).tolist()
            messages.append(f"Dates in the future: {bad}")
        if report.too_early_rows:
            bad = series[too_early_mask].drop_duplicates().astype(str).head(5).tolist()
            messages.append(f"Dates before {self.min_date}: {bad}")
        if report.too_late_rows:
            bad = series[too_late_mask].drop_duplicates().astype(str).head(5).tolist()
            messages.append(f"Dates after {self.max_date}: {bad}")

        if messages and raise_on_error:
            raise ValueError(" | ".join(messages))
        return parsed, report
