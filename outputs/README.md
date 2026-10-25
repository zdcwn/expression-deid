# 本地生成结果

`outputs/` 只放运行过程中产生的文件，除本说明外默认不提交。

| 内容 | 常见位置 |
|---|---|
| 单图匿名结果和预测 JSON | 命令行 `infer --output` 指定的目录 |
| 处理后的视频与逐帧记录 | `video --output`、`--telemetry` 指定的路径 |
| 匿名训练缓存 | `realtime_training/anonymous_cache.npz` |
| 离线评价 | `evaluation/` 或评价命令指定的目录 |
| 桌面和视频检查 | 检查脚本的 `--output` 目录 |

随项目提供的视频在 `examples/videos/`。训练权重在 `models/training/`，发布权重在 `models/` 的相应功能目录，三者分别管理。
