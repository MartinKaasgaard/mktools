from __future__ import annotations

from pathlib import Path

from mktools.kio import DataImporter


def test_kio_smoke(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    importer = DataImporter()
    df = importer.load(csv_path)
    assert df.shape == (2, 2)
    assert importer.last_result is not None
