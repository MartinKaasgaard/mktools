from __future__ import annotations

import zipfile
from pathlib import Path

from mktools.kfile import DirectoryScanner, ZipHandler


def test_kfile_smoke(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "a.txt").write_text("hello", encoding="utf-8")
    (data_dir / "b.csv").write_text("x,y\n1,2\n", encoding="utf-8")

    scanner = DirectoryScanner(data_dir, recurse=False)
    frame = scanner.to_frame()
    assert not frame.empty
    assert frame["is_file"].sum() == 2

    zip_path = tmp_path / "sample.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(data_dir / "a.txt", arcname="a.txt")
    report = ZipHandler(zip_path).ensure_unzipped(tmp_path / "unzipped")
    assert report.total_members == 1
