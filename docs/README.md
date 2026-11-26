# Alpha-Mind Documentation

This documentation describes the current `2026.11.27.1` API. All examples use Arrow as the tabular-data boundary. A complete runnable example is available in [`examples/parquet_demo`](../examples/parquet_demo/README.md).

## Getting started

1. [Introduction](introduction.md): core concepts, data structures, and your first factor.
2. [Setup and the Parquet demo](setup.md): installation, running the demo, input schemas, and outputs.
3. [Arrow and files](arrow-data.md): unified I/O, type conventions, and Parquet pushdown.

The [Notebook index](../notebooks/README.md) contains 27 executable topical examples.

## Data and computation

- [Trading calendars and rebalancing schedules](calendar.md)
- [Factor expressions](factor-expressions.md)
- [Data preprocessing](data-preprocessing.md)
- [MySQL, PostgreSQL, and the ORM](databases.md)
- [Universe definitions and composition](universes.md)

## Research and portfolios

- [Factor analysis and quantile returns](factor-analysis.md)
- [Risk models](risk-models.md)
- [Portfolio construction, constraints, and cvxpy](portfolio-construction.md)
- [Executors and execution pipelines](execution.md)
- [Performance, turnover, and risk attribution](performance-attribution.md)

## Models and production workflows

- [Machine learning and model persistence](machine-learning.md)
- [Strategy backtesting](strategy-backtesting.md)
- [Benchmarks and scheduled jobs](benchmarks.md)

## Project maintenance

- [Development, testing, and coverage](development.md)
- [API Reference](api-reference.md)

## Recommended reading path

For local-file workflows, read **Introduction → Setup → Arrow and Files → Factor Expressions → Portfolio Construction → Performance Attribution**. For database-backed backtests, continue with **Databases → Universes → Machine Learning → Strategy Backtesting**.
