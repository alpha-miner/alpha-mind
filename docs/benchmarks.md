# 基准测试与定时任务

基准测试命令提供可复现的微基准，覆盖预处理、排名组合、calc-flow 和 Parquet I/O。结果同时输出 JSON 和 Markdown。

## 本地运行

```bash
uv run alphamind-benchmarks --output benchmark-results
```

快速冒烟：

```bash
uv run alphamind-benchmarks \
  --size 1000 \
  --samples 2 \
  --warmup 1 \
  --output benchmark-results
```

选择部分用例：

```bash
uv run alphamind-benchmarks \
  --case calc_flow \
  --case parquet \
  --samples 10 \
  --output benchmark-results
```

可用 `--case`：`winsorize`、`standardize`、`neutralize`、`rank_build`、`percent_build`、`calc_flow`、`parquet`。

## 参数

- `--size`：工作负载行数，必须大于 0。
- `--samples`：正式计时次数，必须大于 0。
- `--warmup`：预热次数，可为 0。
- `--output`：结果目录，默认 `benchmark-results`。
- `--case`：重复传入以选择多个用例；缺省运行全部。

## 结果文件

`results.json` 包含：

- schema 版本和 UTC 创建时间；
- Git commit、Alpha-Mind 版本；
- Python 版本和操作系统平台；
- 每个用例的原始秒数、最小/最大/均值/中位数；
- 以中位数计算的每秒处理行数。

`summary.md` 提供适合在 CI 或 PR 中阅读的表格。性能比较应使用相同 Python、依赖、硬件和参数，优先看中位数并同时检查离散程度。

## 定时工作流

`.github/workflows/benchmarks.yml` 每天 UTC 22:17 运行，即北京时间次日 06:17，也支持手动触发。工作流上传结果并保留 30 天。

手动执行：在 GitHub Actions 的 Benchmarks 工作流选择 **Run workflow**。定时任务用于趋势观察，不适合把共享 runner 上的微小波动当成严格性能回归结论。

## 扩展基准

实现代码位于 `src/alphamind/benchmarks/runner.py`：

1. 编写返回可调用对象的 workload 工厂。
2. 将名称注册到 `WORKLOADS`。
3. 保持输入构造在计时区域外。
4. 用固定随机种子生成输入。
5. 在 `tests/benchmarks/` 添加结果和边界测试。

文件 I/O 基准使用临时目录，每次操作都会真实写入并读取 Parquet，不会把网络数据库延迟混入该微基准。
