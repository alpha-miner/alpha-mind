import subprocess
import sys
from pathlib import Path

import pyarrow.compute as pc
import pytest

from alphamind.data.arrow import read_parquet

DEMO = Path(__file__).parents[2] / "examples" / "parquet_demo" / "demo.py"


def test_parquet_demo_runs_end_to_end(tmp_path):
    result = subprocess.run(
        [sys.executable, str(DEMO), "--output", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Created 20 Parquet artifacts" in result.stdout
    manifest = read_parquet(tmp_path / "manifest.parquet")
    assert manifest.num_rows == 20
    assert "features/processed.parquet" in manifest["artifact"].to_pylist()

    optimized = read_parquet(tmp_path / "portfolio/optimized.parquet")
    assert optimized.num_rows == 12
    assert pc.sum(optimized["weight"]).as_py() == pytest.approx(1.0)

    backtest = read_parquet(tmp_path / "backtest/returns.parquet")
    assert backtest.num_rows == 8
    assert pc.sum(backtest["leverage"]).as_py() == pytest.approx(8.0)

    attribution = read_parquet(tmp_path / "analysis/return_attribution.parquet")
    assert attribution.column_names == ["trade_date", "idiosyncratic", "BETA", "SIZE"]
