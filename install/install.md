# Alpha-Mind 安装指南

## 环境要求

- Python 3.9 或更新版本；
- [uv](https://docs.astral.sh/uv/)；
- 使用真实数据库时，需要可访问的 MySQL 或 PostgreSQL 服务。

核心计算和文件读写不需要单独安装 C/C++ 编译器；PyArrow、ConnectorX、calc-flow、DAL 和 cvxpy 均通过项目依赖安装。

## 使用 uv 安装

```bash
git clone https://github.com/alpha-miner/alpha-mind.git
cd alpha-mind
uv sync --group dev
```

安装到现有项目：

```bash
uv add alpha-mind
```

构建 wheel 和源码包：

```bash
uv build
```

## 数据库配置

`DB_VENDOR` 可选择 `mysql`；不设置时默认使用 PostgreSQL 模型。

```bash
export DB_VENDOR=mysql
export DB_URI='mysql+pymysql://user:password@localhost:3306/alpha?charset=utf8'
export FACTOR_TABLES='factor_momentum,factor_valuation_estimation'
```

数据库读取由 ConnectorX 完成并直接返回 Arrow。密码等敏感信息应通过安全的密钥管理方式注入，不要写入源代码或提交到版本库。

## 验证

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

默认测试使用本地文件数据库。需要运行真实数据库集成测试时，再设置 `ALPHAMIND_TEST_DATABASE=1`。
