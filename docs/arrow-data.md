# Arrow 与文件

## 数据边界

`alphamind.data.arrow` 是文件、数据库和内部数值计算之间的边界。公开读写函数返回 `pyarrow.Table`；`pa.RecordBatch` 和 `pa.RecordBatchReader` 也可作为写入输入。

```python
import pyarrow as pa
from alphamind.data import read_table, write_table

source = pa.table({"code": ["000001", "000002"], "value": [1.0, 2.0]})
write_table(source, "factor.parquet")
loaded = read_table("factor.parquet")
assert isinstance(loaded, pa.Table)
```

不要使用 `.to_pandas()` 在模块间传递数据。需要矩阵运算时，在数值函数边界显式转换：

```python
from alphamind.data.arrow import column_values, to_numpy

matrix = to_numpy(loaded, ["value"], dtype=float)  # shape: (n, 1)
vector = column_values(loaded, "value")  # shape: (n,)
```

## 支持的格式

| 格式 | 后缀 | 读取 | 写入 | 备注 |
| --- | --- | --- | --- | --- |
| Parquet | `.parquet`、`.pq` | `read_parquet` / `read_table` | `write_parquet` / `write_table` | 默认 Zstandard 压缩 |
| Arrow IPC | `.arrow`、`.ipc` | `read_table` | `write_table` | 可读取 IPC file 或 stream |
| Feather | `.feather` | `read_table` | `write_table` | Arrow Feather 文件 |
| CSV | `.csv` | `read_table` | `write_table` | 可传 PyArrow CSV 选项 |

后缀不足以判断时显式传 `file_format="parquet"`。无法识别的格式会抛出 `ValueError`。

## Parquet 列裁剪与过滤

```python
from alphamind.data import read_parquet, write_parquet

write_parquet(source, "factor.parquet", compression="zstd", compression_level=5)
selected = read_parquet(
    "factor.parquet",
    columns=["code", "value"],
    filters=[("code", "in", ["000001", "000002"])],
)
```

`columns` 减少读取列，`filters` 使用 PyArrow 的过滤表达式并可在读取时下推。其他关键字参数传给 `pyarrow.parquet.read_table` 或 `write_table`。

IPC 和 Feather 不支持这里的 `columns` / `filters` 参数；CSV 只支持读取后的列选择，不支持 `filters`。

## 类型规范化

`as_arrow` 和文件读取会统一常见外部类型：

- 日期、时间戳以及名称以 `date` 结尾或为 `updatetime` 的时序列转换为 `timestamp[us, UTC]`。
- 无时区时间按 UTC 解释；已有其他时区的时间转换为 UTC。
- binary、large binary 和 large string 转为 string。
- decimal 转为 float64。

因此，读写前后的时间列类型可能不同。证券代码应显式使用字符串，避免前导零丢失；风险方差、收益和权重建议使用 float64。

## 常用 Arrow 辅助函数

```python
from alphamind.data.arrow import (
    align_values,
    as_arrow,
    drop_duplicates,
    drop_nulls,
    from_numpy,
    replace_inf,
    with_column,
)

clean = replace_inf(loaded)
clean = drop_nulls(clean, ["code", "value"])
clean = drop_duplicates(clean, ["code"])
clean = with_column(clean, "value", [2.0, 3.0])
new_table = from_numpy([[1.0, 2.0], [3.0, 4.0]], names=["f1", "f2"])
weights = align_values(["A", "B"], clean["code"], clean["value"], default=0.0)
```

- `drop_nulls` 默认同时剔除浮点 NaN。
- `drop_duplicates` 保留每个键首次出现的行，不自动排序。
- `from_numpy` 接受一维或二维数组，一维数组会成为单列。
- `with_column` 覆盖同名列，不存在时追加。
- `align_values` 使用 Arrow 索引内核对齐代码和数值；源代码重复时取最后值，目标代码缺失时填 `default`，不逐行构造 Python 字典。

## 数据质量建议

在计算窗口因子或连接风险数据前：

1. 固定 `trade_date`、`code` 和风险列 schema。
2. 按 `trade_date`、`code` 排序，并检查键唯一性。
3. 区分 `null`、NaN 和无穷大，不要把缺失窗口默认填成零。
4. 确保同一截面内资产顺序与 NumPy 风险矩阵一致。
5. 将每个研究阶段的输入输出保存为 Parquet，便于审计和复现。

完整文件工作流见 [Parquet demo](setup.md)。
