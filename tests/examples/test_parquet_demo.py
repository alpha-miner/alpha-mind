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

    research = read_parquet(tmp_path / "input/research.parquet")
    processed = read_parquet(tmp_path / "features/processed.parquet")
    assert research.num_rows == 254 * 300
    assert processed.num_rows == 252 * 300

    optimized = read_parquet(tmp_path / "portfolio/optimized.parquet")
    assert optimized.num_rows == 300
    assert pc.sum(optimized["weight"]).as_py() == pytest.approx(1.0)

    backtest = read_parquet(tmp_path / "backtest/returns.parquet")
    assert backtest.num_rows == 252
    assert pc.sum(backtest["leverage"]).as_py() == pytest.approx(252.0)

    positions = read_parquet(tmp_path / "backtest/positions.parquet")
    selected = positions.filter(pc.greater(positions["weight"], 0.0))
    counts = selected.group_by("trade_date").aggregate([("weight", "count")])
    assert counts.num_rows == 252
    assert counts["weight_count"].to_pylist() == [50] * 252

    attribution = read_parquet(tmp_path / "analysis/return_attribution.parquet")
    assert attribution.column_names == ["trade_date", "idiosyncratic", "BETA", "SIZE"]
