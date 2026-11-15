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
| `training/required_data.py` | 三个指定主数据集、唯一辅助集、标签和去重划分检查 |
| `training/train.py`、`training/required_training.py` | 统一命令分派；训练身份与匿名化 |
| `training/expression_current.py` | FER 主训练、KMU 辅助、验证选模与概率校准 |
| `training/required_evaluate.py` | 独立划分的身份、匿名化和匿名表情评价，可复测完整原始照片 |
| `runtime/required_pipeline.py` | 对齐残差组件的诊断流程，不是当前桌面默认流程 |
| `runtime/pipeline.py` | 单图的局部匿名化、区域合成与匿名表情推理 |
| `runtime/anonymization.py` | 关键点条件生成组件与局部合成支持 |
| `runtime/restored_expression.py` | 单一表情检查点的匿名输入校验、归一化和概率输出 |
| `runtime/temporal.py` | 因果表情滤波、连续证据和状态重置 |
| `runtime/video.py` | 视频编解码、检测复用、跟踪、逐帧匿名分析与计时 |
| `ui/desktop.py` | 三页桌面、工作线程、图片缓存、视频及摄像头操作 |
| `ui/widgets.py` | 桌面样式、图标、卡片、图表等公共组件 |
| `ui/opencv_demo.py` | 推理实例装配与 OpenCV 本地演示 |

运行侧的依赖方向为 `ui → runtime → preprocessing/models`。训练侧使用相同的预处理和网络定义，并在匿名域适配时调用匿名化接口；模型类本身不依赖桌面窗口。

## 二、训练入口与产物

| 入口 | 输入 | 候选产物 |
|---|---|---|
| `train-identity` | Pins，面部椭圆归一化 | `models/training/identity_pins.pt` |
| `train-anonymizer` | CelebA-HQ，冻结 Pins 身份器 | `models/training/anonymizer_celebahq.pt` |
| `train-expression` | 当前完整运行流程产生的 FER/KMU 匿名缓存 | `models/training/expression_fer2013/best.pt` |
| `scripts/checks/check_current_expression.py` | 固定表情检查点、来源和匿名缓存 | `outputs/verification/expression-validation.json` |

表情继续适配实现位于 `training/expression_current.py`，FER/KMU 抽样比例 85%/15%，弱增强加输出一致性约束。初始化权重含历史多域学习；固定模型后分别评估 FER、KMU、RAF，最高验证结果为 KMU 六类 91.74%（230 张）。完整参数及指标见 README 和检查点元数据。

```powershell
python scripts/checks/check_current_expression.py --data-root datasets --device cuda
python -m expression_deid train-expression --data-root datasets --device cuda
```

缓存必须是 `anonymous-runtime.npy/json`，附带源图、标签、划分和匿名权重摘要。原图不进入表情分类器。`required_evaluate.py` 仅用于对齐残差组件的指定数据诊断，与当前完整运行缓存的表情复测分别记录。

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

训练检查会建立临时小样本，验证三个阶段的梯度、检查点读写和匿名区域边界，不作为模型准确率。当前运行表情模型以 FER 为主训练；v1.1.0 发行包使用相同模型和固定指标。

## 六、桌面软件发行

`scripts/distribution/` 负责 Windows 运行库收集、资源完整性清单、冻结可执行程序和单文件启动版。`expression_deid/distribution.py` 管理设备探测、资源检查与当前用户日志目录；`distribution_check.py` 在成品程序内验证图片、视频、界面和退出，不依赖源代码运行环境。中文回退字体与许可放在 `expression_deid/ui/assets/`。

单文件版首次自动展开到当前用户目录，之后复用缓存；便携 ZIP 完整解压后启动。发行配置只纳入当前运行权重、示例图库、示例视频、指标和许可，不收集训练产物与本地实验记录。构建和验证步骤见 `docs/distribution/README.md`。
