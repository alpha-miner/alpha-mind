# Alpha-Mind 文档

本文档基于当前 `0.4.0` API 编写。所有示例以 Arrow 为表格数据边界，完整可运行示例位于 [`examples/parquet_demo`](../examples/parquet_demo/README.md)。

## 入门

1. [Introduction](introduction.md)：核心概念、数据结构和第一个因子。
2. [Setup 与 Parquet 示例](setup.md)：安装、运行示例、输入 schema 与输出说明。
3. [Arrow 与文件](arrow-data.md)：统一读写入口、类型规范与 Parquet 下推。

## 数据与计算

- [交易日与调仓日程](calendar.md)
- [因子表达式](factor-expressions.md)
- [数据预处理](data-preprocessing.md)
- [MySQL、PostgreSQL 与 ORM](databases.md)
- [股票池定义与组合](universes.md)

## 研究与组合

- [因子分析与分层收益](factor-analysis.md)
- [风险模型](risk-models.md)
- [组合构建、约束与 cvxpy](portfolio-construction.md)
- [执行器与执行管道](execution.md)
- [绩效、换手与风险归因](performance-attribution.md)

## 模型与生产流程

- [机器学习与模型持久化](machine-learning.md)
- [策略回测](strategy-backtesting.md)
- [基准测试与定时任务](benchmarks.md)

## 项目维护

- [开发、测试与覆盖率](development.md)
- [API Reference](api-reference.md)

## 推荐阅读路径

如果只使用本地文件，按 **Introduction → Setup → Arrow 与文件 → 因子表达式 → 组合构建 → 绩效归因** 阅读。需要数据库回测时，再阅读 **数据库 → 股票池 → 机器学习 → 策略回测**。
