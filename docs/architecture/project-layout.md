# 工程目录与操作路径

工程按“数据准备、训练、模型文件、运行推理、桌面界面、验证与交付”组织。下列路径对应实际文件；根目录只保留项目说明和功能目录。完整目录索引见 [file-tree.txt](file-tree.txt)。

## 一、源码各自负责什么

所有自有 Python 代码在 `expression_deid/`。`__init__.py` 用于声明包；主要文件如下。

| 文件 | 职责 |
|---|---|
| `__main__.py`、`cli.py` | `python -m expression_deid` 命令分发，连接准备、训练、推理和评价入口 |
| `desktop_ui.py` | 保留的桌面启动命令，转入 `ui/desktop.py` |
| `config.py` | 项目根目录、数据和资源路径、对齐尺寸、检测阈值 |
| `labels.py` | 七类表情名称、中文显示名、KMU 文件名标签解析 |
| `models.py` | 网络结构、输入张量检查、检查点保存与加载；不是模型权重目录 |
| `preprocessing/detect.py` | YuNet 推理及检测输出解码 |
| `preprocessing/align.py` | 五点筛选、相似变换与灰度脸对齐 |
| `preprocessing/normalize.py` | 灰度读取、标准化和表情输入构造 |
| `preprocessing/norm_stats.json` | 训练使用的灰度统计量 |
| `preprocessing/feature_geometry.py` | 局部几何和身份比例编辑的几何运算 |
| `preprocessing/prepare.py` | KMU 对齐、处理清单与统计量准备 |
| `preprocessing/prepare_ppb.py` | PPB 中央红外中性视频抽帧与对齐 |
| `training/data.py` | 数据集读取、标签映射、人员/序列划分与数据增强 |
| `training/train.py` | 表情、身份、匿名化和联合适配训练；计算损失并选择检查点 |
| `training/evaluate.py` | 对给定模型与数据集做离线评价并输出统计 |
| `runtime/pipeline.py` | 单图的局部匿名化、区域合成与匿名表情推理 |
| `runtime/anonymization.py` | 关键点条件生成组件与局部合成支持 |
| `runtime/expression.py` | 质量表情模型加载、输入契约与哈希核对、匿名图概率输出 |
| `runtime/temporal.py` | 因果表情滤波、连续证据和状态重置 |
| `runtime/video.py` | 视频编解码、检测复用、跟踪、逐帧匿名分析与计时 |
| `ui/desktop.py` | 三页桌面、工作线程、图片缓存、视频及摄像头操作 |
| `ui/widgets.py` | 桌面样式、图标、卡片、图表等公共组件 |
| `ui/opencv_demo.py` | 推理实例装配与 OpenCV 本地演示 |

运行侧的依赖方向为 `ui → runtime → preprocessing/models`。训练侧使用相同的预处理和网络定义，并在匿名域适配时调用匿名化接口；模型类本身不依赖桌面窗口。

## 二、训练在哪儿，产物到哪儿

常规训练由 `python -m expression_deid` 调用 `training/train.py`。在项目根目录运行：

| 命令 | 主要输入 | 默认产物 |
|---|---|---|
| `prepare` | KMU 源图、检测模型 | `datasets/kmu-fed-aligned/`、统计量 |
| `train-expression` | RAF-DB 训练/验证图 | `models/training/expression.pt` |
| `train-identity` | 对齐图与人员编号 | `models/training/identity.pt` |
| `train-expression-kmu` | 基线检查点、KMU 对齐图 | `models/training/expression_kmu.pt` |
| `train-expression-joint` | 基线检查点及配置的数据域 | `models/training/expression_joint.pt` |
| `train-anonymizer` | 自建身份/表情检查点与对齐图 | `models/training/anonymizer.pt` |

训练入口接受 `--output`，可以指定新的候选路径。默认候选与运行权重分开，训练不会悄悄改变桌面端当前加载的版本。

专用训练分为三个入口：

1. `scripts/training/train_realtime_expression.py` 先生成匿名缓存，也可在该缓存上训练轻量候选。`--prepare-only` 只准备数据，`--cache` 指定缓存路径；默认候选为 `models/training/expression_realtime_candidate.pt`。
2. `scripts/training/train_video_quality.py` 接收 `--anonymous-cache`、`--initial` 与 `--output`。初始化检查点应包含分类头、特征均值和标准差；训练目录生成 `best.pt`、`best_valid.npz` 和 `training.json`。
3. `scripts/export/export_video_quality.py` 将选中的质量检查点导出为 TorchScript，验证导出前后的输出，并生成配套 JSON。

例如，使用已经准备好的匿名缓存和质量初始化检查点：

```powershell
python scripts/training/train_video_quality.py `
  --anonymous-cache outputs/realtime_training/anonymous_cache.npz `
  --initial models/training/quality_initial.pt `
  --output models/training/quality_run --device cuda

python scripts/export/export_video_quality.py `
  --checkpoint models/training/quality_run/best.pt `
  --onnx models/training/enet_b2_7.onnx `
  --output models/training/quality_bundle --device cuda
```

导出目录是一组可共同校验的文件：质量 TorchScript、轻量表情权重及 JSON。先以该目录的 JSON 运行质量验证，再决定部署；候选不会被桌面端自动扫描。

```powershell
python scripts/checks/check_video_quality.py `
  --manifest models/training/quality_bundle/expression_video_quality.json `
  --anonymous-cache outputs/realtime_training/anonymous_cache.npz `
  --output outputs/evaluation/quality_candidate --device cuda
```

## 三、运行模型如何接入界面

`models/detection/` 保存 YuNet 和关键点模型；`models/anonymization/` 保存统一匿名化组件；`models/expression/` 保存当前表情权重及契约。文件清单与加载位置见 [models/README.md](../../models/README.md)。权重移动到新目录时只修改资源引用，没有重新训练或修改网络参数。

桌面启动仍使用 `python -m expression_deid.desktop_ui --device cuda`。该入口进入 `ui/desktop.py`，工作线程加载推理实例。图片页调用匿名化和表情分析；视频页调用 `runtime/video.py` 连续处理；表现页读取 `docs/model-metrics.json`。

`runtime/pipeline.py` 负责模型调用和图像结果，不直接创建 Qt 窗口。`runtime/video.py` 管理帧、跟踪和时序状态；`ui/desktop.py` 接收已计算结果并更新控件。原图仅用于预处理、匿名处理和对照显示，表情调用的输入必须来自匿名结果。

## 四、输入、输出和交付资料

| 位置 | 保存什么 | 管理原则 |
|---|---|---|
| `datasets/` | 原始数据和明确命名的对齐数据 | 原图只读，清单关联源图与结果 |
| `examples/videos/` | 固定示例输入 | 随项目发布，由桌面视频页读取 |
| `models/training/` | 新训练候选、阶段检查点 | 本地保存，验证后再明确部署 |
| `outputs/` | 匿名缓存、预测、处理视频、日志和评价 | 本地生成，默认忽略 |
| `requirements/` | 各用途的依赖清单 | 基础、训练、匿名化、界面分开安装 |
| `tests/` | 网络、预处理、匿名边界、时序、路径等测试 | 随代码维护 |
| `third_party/` | 引用的推理代码和许可证 | 保留来源声明 |
| `docs/architecture/` | 目录职责和模型接入说明 | 与实际文件同步 |
| `docs/design/` | 桌面设计说明 | 对应 Python 桌面实现 |
| `docs/handoff.md` | 接手与运行步骤 | 使用当前目录和命令 |
| `reports/tongji_final_report/` | 最终报告源码、图和 PDF | 主文件 `main.tex` |
| `reports/tongji_project_plan/` | 计划书源码、图和 PDF | 主文件 `main.tex` |
| `reports/archive/` | 早期说明稿 | 保留查阅，不作当前运行入口 |

## 五、整理后的检查顺序

先检查导入和参数解析，再核对运行模型与契约、图片和示例视频路径，然后运行训练的小样本检查及桌面检查。相同输入整理前后的匿名图像素、表情结果和置信度应一致。

```powershell
python -m compileall -q expression_deid scripts tests
python -m unittest discover -s tests -p "test_*.py" -q
python scripts/checks/check_training_paths.py
python scripts/checks/check_desktop_presentation.py --device cuda
```

训练检查需要已准备的匿名缓存；它验证训练入口、梯度和检查点读写，不作为模型准确率。详细评价由具体数据集和质量验证脚本完成。结构整理不改变数据集划分、匿名化算法和已经发布的模型指标。

## 六、桌面软件发行

`scripts/distribution/` 负责 Windows 运行库收集、资源完整性清单、冻结可执行程序和单文件启动版。`expression_deid/distribution.py` 管理设备探测、资源检查与当前用户日志目录；`distribution_check.py` 在成品程序内验证图片、视频、界面和退出，不依赖源代码运行环境。中文回退字体与许可放在 `expression_deid/ui/assets/`。

单文件版首次自动展开到当前用户目录，之后复用缓存；便携 ZIP 完整解压后启动。发行配置只纳入当前运行权重、示例图库、示例视频、指标和许可，不收集训练产物与本地实验记录。构建和验证步骤见 `docs/distribution/README.md`。
