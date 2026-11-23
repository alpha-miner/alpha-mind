import json

import pytest

from alphamind.benchmarks.runner import WORKLOADS, get_parser, main, run_case, run_suite


def test_benchmark_case_collects_repeated_samples():
    result = run_case("standardize", size=64, samples=3, warmup=1)
    assert result["name"] == "standardize"
    assert result["size"] == 64
    assert len(result["seconds"]) == 3
    assert result["min_seconds"] <= result["median_seconds"] <= result["max_seconds"]
    assert result["rows_per_second"] > 0


def test_benchmark_suite_writes_machine_and_human_reports(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "test-sha")
    report = run_suite(tmp_path, size=64, samples=2, warmup=1, cases=["standardize", "calc_flow"])

    restored = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert restored == report
    assert restored["git_sha"] == "test-sha"
    assert [case["name"] for case in restored["results"]] == ["standardize", "calc_flow"]
    assert "standardize" in summary
    assert "calc_flow" in summary


def test_benchmark_cli_supports_case_selection(tmp_path):
    args = get_parser().parse_args(
        ["--case", "standardize", "--size", "32", "--samples", "1", "--warmup", "0"]
    )
    assert args.cases == ["standardize"]
    assert (
        main(
            [
                "--case",
                "standardize",
                "--size",
                "32",
                "--samples",
                "1",
                "--warmup",
                "0",
                "--output",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert (tmp_path / "results.json").is_file()


def test_benchmark_rejects_invalid_sampling_parameters():
    for kwargs in ({"size": 0}, {"samples": 0}, {"warmup": -1}):
        with pytest.raises(ValueError):
            run_case("standardize", **{"size": 16, **kwargs})


def test_every_benchmark_workload_can_execute():
    for name in WORKLOADS:
        result = run_case(name, size=16, samples=1, warmup=0)
        assert result["median_seconds"] > 0
