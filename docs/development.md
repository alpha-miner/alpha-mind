# 开发、测试与覆盖率

## 环境

```bash
uv sync --frozen --group dev
```

项目采用 `src/alphamind` 布局、Hatchling 构建和 uv 锁文件，支持 Python 3.9+。代码目标版本为 Python 3.9，公共实现使用 `from __future__ import annotations` 等方式保持较新的类型标注写法，同时不破坏 3.9 运行兼容性。

## Ruff

```bash
uv run ruff check .
uv run ruff format --check .
```

自动修复可安全处理的问题并格式化：

```bash
uv run ruff check . --fix
uv run ruff format .
```

Ruff 配置位于 `pyproject.toml`，行宽 100，目标语法 Python 3.9，启用导入排序及 E4、E7、E9、F、I 规则。提交前两项检查都必须通过。

## pytest

```bash
uv run pytest -q
```

运行指定功能：

```bash
uv run pytest tests/data -q
uv run pytest tests/portfolio -q
uv run pytest tests/examples/test_parquet_demo.py -q
```

测试目录按功能划分，集成测试使用 `@pytest.mark.integration`。不要在单元测试中依赖外部网络数据；文件示例应使用临时目录。

## 覆盖率门禁

```bash
uv run pytest \
  --cov=alphamind \
  --cov-report=term-missing \
  --cov-report=xml \
  --cov-report=html
```

配置启用分支覆盖率，`fail_under = 90`。低于 90% 时命令失败。HTML 报告位于 `htmlcov/`，XML 为 `coverage.xml`，这些本地生成物不会提交。

提升覆盖率时应测试真实行为、错误边界和跨后端差异，而不是用无断言调用凑行数。Parquet demo 的端到端 pytest 会实际执行命令并检查优化权重、回测行数和归因 schema。

## 数据库集成测试

```bash
export ALPHAMIND_MYSQL_URI='mysql+pymysql://root:root@127.0.0.1:3306/alpha'
export ALPHAMIND_POSTGRES_URI='postgresql+psycopg2://alpha:alpha@127.0.0.1:5432/alpha'
uv run pytest tests/data/engines -q
```

未设置 URI 时真实服务测试跳过；SQLite 文件数据库测试仍覆盖主要查询和仓储行为。CI 在 MySQL 8.4、PostgreSQL 16 服务上运行全部集成测试。

## CI

`.github/workflows/ci.yml` 包含：

1. **Quality**：Python 3.13 上运行 Ruff lint 和格式检查。
2. **Test**：Python 3.9–3.13 矩阵，启动 MySQL 和 PostgreSQL；3.13 生成分支覆盖率报告。
3. **Package**：构建 wheel 和源码包，在隔离环境安装 wheel，并检查导入及两个 CLI。

推送 master 和 Pull Request 都会触发。并发组会取消同一分支或 PR 中已过时的运行。

## Notebooks

27 本 notebook 均使用本地确定性数据和当前 Arrow API。安装执行依赖并逐本运行：

```bash
uv sync --frozen --group dev --extra notebooks
uv run python scripts/execute_notebooks.py
```

CI 的 Notebooks 任务会在 Python 3.13 上使用真实 Jupyter kernel 执行全部代码单元。修改公开 API 时，应同步更新 notebook 并运行该命令。

## 构建与安装检查

```bash
uv build
uv venv .package-venv
uv pip install --python .package-venv/bin/python dist/*.whl
.package-venv/bin/python -c "import alphamind; import alphamind.api"
.package-venv/bin/alphamind --help
.package-venv/bin/alphamind-benchmarks --help
```

完成后清理本地构建和临时虚拟环境，不要提交生成物。

## 文档贡献

- 所有文档使用 Markdown，集中放在 `docs/`。
- 每个功能有独立页面，并从 `docs/README.md` 和根 README 链接。
- 示例必须匹配当前公开 API，避免复制旧 notebooks 中已经失效的 pandas 或 Finance-Python 用法。
- 修改流程行为时同步更新对应功能页、Parquet demo 和测试。
- 使用相对链接，确保 GitHub 渲染和本地阅读都可用。
