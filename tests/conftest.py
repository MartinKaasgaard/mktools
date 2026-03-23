from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if not SRC.exists():
    raise RuntimeError(f"Expected src directory not found: {SRC}")

src_str = str(SRC)
if src_str not in sys.path:
    sys.path.insert(0, src_str)
