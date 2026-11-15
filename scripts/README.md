# 命令脚本

在项目根目录调用下列路径，脚本也会根据自身位置定位项目根目录。

| 目录 | 文件 | 用途 |
|---|---|---|
| `assets/` | `download_anonymization_assets.py` | 下载匿名化与关键点资产，写入对应模型目录 |
| `data/` | `download_required.py` | 下载指定 Kaggle 数据源、校验并解压 |
| `data/` | `prepare_required.py` | 准备 CelebA-HQ、FER-2013、Pins 及唯一辅助集 |
| 项目主入口 | `python -m expression_deid train-*` | 统一的身份、匿名化、表情训练流程 |
| `checks/` | `check_anonymous_cache.py` | 检查匿名缓存来源及结构 |
| `checks/` | `check_detector_reference.py` | 对照 YuNet 解码结果 |
| `checks/` | `check_training_paths.py` | 执行三个新训练阶段，检查梯度、检查点与面部外像素 |
| `checks/` | `check_current_expression.py` | 复测当前表情检查点，核对运行匿名缓存像素，输出 FER/KMU/RAF 完整划分指标 |
| `checks/` | `check_video_stability.py` | 连续帧表情稳定性检查 |
| `checks/` | `check_desktop_runtime.py` | 图片、视频、编码及桌面线程检查 |
| `checks/` | `check_desktop_presentation.py` | 三个桌面页面和实际示例检查 |

有参数的脚本可使用 `--help`；资产下载脚本直接执行。详细训练顺序和产物位置见 `docs/architecture/project-layout.md`。

当前表情训练以匿名 FER-2013 为主、KMU 为辅，入口 `training/expression_current.py`；身份与匿名化继续沿用后续版本。`check_desktop_presentation.py --max-images 24` 可在完整图库中选取覆盖各类别的有界联调样本，零或省略该参数处理全部图库。页面截图使用实际输出，不能代替独立模型准确率。

`data/prepare_runtime_expression_cache.py` 生成完整运行流程的匿名输入，支持 FER、KMU 和 RAF；模型训练、固定权重复测与软件打包分别使用自己的入口，发行不包含训练候选。
