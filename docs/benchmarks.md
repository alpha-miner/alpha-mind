# Benchmarks and Scheduled Jobs

The benchmark command provides reproducible microbenchmarks covering preprocessing, ranked portfolios, full-history turnover, calc-flow, and Parquet I/O. Results are emitted as both JSON and Markdown.

## Local execution

```bash
uv run alphamind-benchmarks --output benchmark-results
```

Quick smoke run:

```bash
uv run alphamind-benchmarks \
  --size 1000 \
  --samples 2 \
  --warmup 1 \
  --output benchmark-results
```

Select specific cases:

```bash
uv run alphamind-benchmarks \
  --case calc_flow \
  --case parquet \
  --samples 10 \
  --output benchmark-results
```

Available `--case` values are `winsorize`, `standardize`, `neutralize`, `rank_build`, `percent_build`, `calc_flow`, `calc_flow_shift`, `turnover`, and `parquet`.

## Parameters

- `--size`: workload row count; must be greater than 0.
- `--samples`: number of measured runs; must be greater than 0.
- `--warmup`: number of warm-up runs; may be 0.
- `--output`: result directory, defaulting to `benchmark-results`.
- `--case`: repeat to select multiple cases; all cases run by default.

## Result files

`results.json` contains:

- Schema version and UTC creation time.
- Git commit and Alpha-Mind version.
- Python version and operating-system platform.
- Raw seconds, minimum, maximum, mean, and median for every case.
- Rows processed per second, calculated from the median.

`summary.md` provides a table suitable for CI or pull requests. Performance comparisons should use identical Python versions, dependencies, hardware, and parameters. Prefer medians and inspect dispersion as well.

## Scheduled workflow

`.github/workflows/benchmarks.yml` runs at 22:17 UTC every day, equivalent to 06:17 Asia/Shanghai on the following day, and also supports manual triggering. The workflow uploads results and retains them for 30 days.

To run manually, select **Run workflow** in the GitHub Actions Benchmarks workflow. Scheduled jobs are intended for trend observation; small fluctuations on shared runners should not be treated as strict performance-regression conclusions.

## Extending benchmarks

Implementation code is located in `src/alphamind/benchmarks/runner.py`:

1. Write a workload factory that returns a callable.
2. Register its name in `WORKLOADS`.
3. Keep input construction outside the timed region.
4. Generate inputs with a fixed random seed.
5. Add result and boundary tests under `tests/benchmarks/`.

File-I/O benchmarks use temporary directories. Every operation writes and reads actual Parquet files; network database latency is not mixed into this microbenchmark.
