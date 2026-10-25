# 命令脚本

在项目根目录调用下列路径，脚本也会根据自身位置定位项目根目录。

| 目录 | 文件 | 用途 |
|---|---|---|
| `assets/` | `download_anonymization_assets.py` | 下载匿名化与关键点资产，写入对应模型目录 |
| `data/` | `prepare_gan_expression.py` | 生成匿名训练图及清单 |
| `training/` | `train_realtime_expression.py` | 构建匿名缓存、训练轻量视频候选 |
| `training/` | `train_video_quality.py` | 在匿名缓存上训练质量模型 |
| `export/` | `export_video_quality.py` | 导出 TorchScript、核对输出并生成输入契约 |
| `checks/` | `check_anonymous_cache.py` | 检查匿名缓存来源及结构 |
| `checks/` | `check_detector_reference.py` | 对照 YuNet 解码结果 |
| `checks/` | `check_training_paths.py` | 小样本执行常规训练入口 |
| `checks/` | `check_video_quality.py` | 质量模型数据集评价与校准 |
| `checks/` | `check_video_stability.py` | 连续帧表情稳定性检查 |
| `checks/` | `check_desktop_runtime.py` | 图片、视频、编码及桌面线程检查 |
| `checks/` | `check_desktop_presentation.py` | 三个桌面页面和实际示例检查 |

有参数的脚本可使用 `--help`；资产下载脚本直接执行。详细训练顺序和产物位置见 `docs/architecture/project-layout.md`。
