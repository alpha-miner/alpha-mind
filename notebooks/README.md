# Alpha-Mind Notebooks

Every notebook uses the current Arrow, calc-flow, dal-python, and cvxpy APIs with deterministic local data. No database, market-data service, or network download is required.

## Installation and execution

```bash
uv sync --frozen --group dev --extra notebooks
uv run python scripts/execute_notebooks.py
```

The runner executes every code cell in a real Jupyter Python kernel, one notebook at a time. Any failing cell makes the command return a nonzero status. Results are not written back to the notebooks, so the repository examples contain no historical outputs. Direct calc-flow SQL examples use the native `await cf.compute_async(...)` API. Cells containing synchronous `Transformer` calls use top-level `await asyncio.to_thread(...)` to remain compatible with Jupyter's existing event loop.

For interactive reading:

```bash
uv run jupyter lab notebooks
```

## Quick Start

- [Quick Start 1 - Factor Preprocess](<Quick Start 1 - Factor Preprocess.ipynb>)
- [Quick Start 2 - Factor Rank and Quantile](<Quick Start 2 - Factor Rank and Quantile.ipynb>)
- [Quick Start 3 - Portfolio Builder](<Quick Start 3 - Portfolio Builder.ipynb>)
- [Quick Start 5 - Alpha Factor Quantile Analysis](<Quick Start 5 - Alpha Factor Quantile Analysis.ipynb>)
- [Quick Start 6 - Formula Based Stocks Screening](<Quick Start 6 - Formula Based Stocks Screening.ipynb>)
- [Quick Start 7 - Single Factor IC Analysis](<Quick Start 7 - Single Factor IC Analysis.ipynb>)
- [Quick Start 8 - IC Decay Calculation](<Quick Start 8 - IC Decay Calculation.ipynb>)
- [Quick Start 9 - Back Testing Stock Screening](<Quick Start 9 - Back Testing Stock Screening.ipynb>)

## Step By Step

- [Step By Step 01 - Getting Started](<Step By Step 01 - Getting Started.ipynb>)
- [Step By Step 02 - Data Acquisition](<Step By Step 02 - Data Acquisition.ipynb>)
- [Step By Step 03 - Factor Design](<Step By Step 03 - Factor Design.ipynb>)
- [Step By Step 04 - Factor Preprocessing](<Step By Step 04 - Factor Preprocessing.ipynb>)
- [Step By Step 05 - Portfolio Optimization](<Step By Step 05 - Portfolio Optimization.ipynb>)
- [Step By Step 06 - Complete Backtest](<Step By Step 06 - Complete Backtest.ipynb>)
- [Step By Step 07 - Risk Factor Model](<Step By Step 07 - Risk Factor Model.ipynb>)

## Topical examples

- [Example 1 - Factor IC analysis](<Example 1 - Factor IC analysis.ipynb>)
- [Example 2 - Strategy Analysis](<Example 2 - Strategy Analysis.ipynb>)
- [Example 3 - Multi Weight Gap Comparison](<Example 3 - Multi Weight Gap Comparison.ipynb>)
- [Example 4 - Single Factor Analysis](<Example 4 - Single Factor Analysis.ipynb>)
- [Example 5 - Style Factor Analysis](<Example 5 - Style Factor Analysis.ipynb>)
- [Example 6 - Target Volatility Builder](<Example 6 - Target Volatility Builder.ipynb>)
- [Example 7 - Portfolio Optimizer Performance](<Example 7 - Portfolio Optimizer Performance.ipynb>)
- [Example 9 - Linear Optimizer with cvxpy](<Example 9 - Linear Optimizer Comparison with CVXOPT.ipynb>)
- [Example 10 - Quadratic Optimizer with cvxpy](<Example 10 - Quadratic Optimizer Comparison with CVXOPT.ipynb>)
- [Example 11 - Long Short Strategy Model](<Example 11 - Long Short Strategy Model.ipynb>)
- [Example 12 - Machine Learning Model Prediction](<Example 12 - Machine Learning Model Prediction.ipynb>)
- [Example 13 - Evaluation within Industry Groups](<Example 13 - Evaluation within Industry Groups.ipynb>)

Historical numbering is retained so existing links remain easy to locate. The original Markdown explanations, code comments, and complete instructional steps are preserved in English, while the calculations use the current backends.
