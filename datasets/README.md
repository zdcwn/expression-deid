# 数据目录

| 路径 | 数据类型 | 谁使用 | 是否由程序生成 |
|---|---|---|---|
| `kmu-fed/` | KMU 近红外源图 | 数据准备 | 否，源图只读 |
| `raf-db/train/`、`raf-db/valid/` | RAF-DB 的训练与验证图 | 表情训练、评价 | 否 |
| `kmu-fed-aligned/` | 256 灰度对齐图、清单和处理信息 | 图片实验、训练和检查 | 是，`python -m expression_deid prepare` |

对齐和归一化代码位于 `expression_deid/preprocessing/`，训练侧的数据集封装位于 `expression_deid/training/data.py`。原图与派生数据分开保存，清单记录源文件与输出文件的对应关系。

匿名缓存由 `scripts/training/train_realtime_expression.py --prepare-only` 生成，保存在 `outputs/`，不混入源图目录。扩展训练所用的本地数据由对应训练参数指定，分享和使用遵守各数据集许可。
