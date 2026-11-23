# Alpha-Mind Notebooks

全部 notebook 使用当前 Arrow、calc-flow、dal-python 和 cvxpy API，以本地确定性数据演示功能，不需要数据库、行情服务或网络下载。

## 安装与执行

```bash
uv sync --frozen --group dev --extra notebooks
uv run python scripts/execute_notebooks.py
```

执行器使用真实 Jupyter Python kernel 逐本运行所有代码单元，任一单元失败都会使命令返回非零。执行结果不会写回 notebook，因此仓库中的示例保持无历史输出。含同步 calc-flow 调用的单元使用顶层 `await asyncio.to_thread(...)`，以兼容 Jupyter 已有的事件循环。

交互式阅读：

```bash
uv run jupyter lab notebooks
```

## 快速入门

- [Quick Start 1 - Factor Preprocess](<Quick Start 1 - Factor Preprocess.ipynb>)
- [Quick Start 2 - Factor Rank and Quantile](<Quick Start 2 - Factor Rank and Quantile.ipynb>)
- [Quick Start 3 - Portfolio Builder](<Quick Start 3 - Portfolio Builder.ipynb>)
- [Quick Start 5 - Alpha Factor Quantile Analysis](<Quick Start 5 - Alpha Factor Quantile Analysis.ipynb>)
- [Quick Start 6 - Formula Based Stocks Screening](<Quick Start 6 - Formula Based Stocks Screening.ipynb>)
- [Quick Start 7 - Single Factor IC Analysis](<Quick Start 7 - Single Factor IC Analysis.ipynb>)
- [Quick Start 8 - IC Decay Calculation](<Quick Start 8 - IC Decay Calculation.ipynb>)
- [Quick Start 9 - Back Testing Stock Screening](<Quick Start 9 - Back Testing Stock Screening.ipynb>)

## Step By Step

- [Step By Step 01 - 入门](<Step By Step 01 - 入门.ipynb>)
- [Step By Step 02 - 数据获取](<Step By Step 02 - 数据获取.ipynb>)
- [Step By Step 03 - 因子设计](<Step By Step 03 - 因子设计.ipynb>)
- [Step By Step 04 - 因子预处理](<Step By Step 04 - 因子预处理.ipynb>)
- [Step By Step 05 - 组合优化](<Step By Step 05 - 组合优化.ipynb>)
- [Step By Step 06 - 完整的回测](<Step By Step 06 - 完整的回测.ipynb>)
- [Step By Step 07 - 风险因子模型](<Step By Step 07 - 风险因子模型.ipynb>)

## 专题示例

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

文件名保留历史编号，便于已有链接继续定位；原始 Markdown 说明、代码注释和完整教学步骤均保留，计算实现迁移到当前后端。
