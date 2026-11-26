# Development, Testing, and Coverage

## Environment

```bash
uv sync --frozen --group dev
```

The project uses a `src/alphamind` layout, Hatchling builds, and a uv lockfile, with Python 3.9+ support. The code targets Python 3.9, while the public implementation uses approaches such as `from __future__ import annotations` to retain modern type-annotation syntax without breaking Python 3.9 runtime compatibility.

## Ruff

```bash
uv run ruff check .
uv run ruff format --check .
```

Fix safely addressable issues and format automatically with:

```bash
uv run ruff check . --fix
uv run ruff format .
```

Ruff is configured in `pyproject.toml` with a line length of 100, Python 3.9 target syntax, import sorting, and the E4, E7, E9, F, and I rules. Both checks must pass before submission.

## pytest

```bash
uv run pytest -q
```

Run specific functionality with:

```bash
uv run pytest tests/data -q
uv run pytest tests/portfolio -q
uv run pytest tests/examples/test_parquet_demo.py -q
```

The test directory is organized by feature, and integration tests use `@pytest.mark.integration`. Unit tests must not depend on external network data; file examples should use temporary directories.

## Coverage gate

```bash
uv run pytest \
  --cov=alphamind \
  --cov-report=term-missing \
  --cov-report=xml \
  --cov-report=html
```

The configuration enables branch coverage and sets `fail_under = 90`. The command fails below 90%. HTML reports are written to `htmlcov/` and XML to `coverage.xml`; these local artifacts are not committed.

Improve coverage by testing real behavior, error boundaries, and cross-backend differences rather than accumulating lines through assertion-free calls. The Parquet demo's end-to-end pytest actually executes the command and checks optimized weights, backtest row counts, and attribution schemas.

## Database integration tests

```bash
export ALPHAMIND_MYSQL_URI='mysql+pymysql://root:root@127.0.0.1:3306/alpha'
export ALPHAMIND_POSTGRES_URI='postgresql+psycopg2://alpha:alpha@127.0.0.1:5432/alpha'
uv run pytest tests/data/engines -q
```

Real-service tests are skipped when URI variables are unset; SQLite file-database tests still cover principal query and repository behavior. CI runs all integration tests against MySQL 8.4 and PostgreSQL 16 services.

## CI

`.github/workflows/ci.yml` contains:

1. **Quality**: Ruff lint and format checks on Python 3.13.
2. **Test**: a Python 3.9–3.13 matrix with MySQL and PostgreSQL services; Python 3.13 generates the branch-coverage report.
3. **Package**: builds the wheel and source distribution, installs the wheel in an isolated environment, and checks imports and both CLIs.

Pushes to master and pull requests trigger the workflow. Concurrency groups cancel obsolete runs on the same branch or pull request.

## Notebooks

All 27 notebooks use deterministic local data and the current Arrow API. Install execution dependencies and run them one by one with:

```bash
uv sync --frozen --group dev --extra notebooks
uv run python scripts/execute_notebooks.py
```

The CI Notebooks job executes every code cell on Python 3.13 using a real Jupyter kernel. When changing a public API, update the notebooks and run this command.

## Build and installation checks

```bash
uv build
uv venv .package-venv
uv pip install --python .package-venv/bin/python dist/*.whl
.package-venv/bin/python -c "import alphamind; import alphamind.api"
.package-venv/bin/alphamind --help
.package-venv/bin/alphamind-benchmarks --help
```

Clean up local builds and temporary virtual environments afterward; do not commit generated artifacts.

## Documentation contributions

- Write all documentation in English using Markdown, with principal pages under `docs/`.
- Give each feature a separate page and link it from both `docs/README.md` and the root README.
- Examples must match the current public API; avoid copying obsolete pandas or Finance-Python usage from old notebooks.
- When changing workflow behavior, update the corresponding feature page, Parquet demo, and tests together.
- Use relative links so both GitHub rendering and local reading work.
