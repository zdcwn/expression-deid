# Expression Deid

车载座舱人脸匿名化与表情分析的本地 Python 项目。桌面端提供图片实验、视频与摄像头分析、模型表现三个页面。

处理顺序为：**图片或视频 → 灰度化 → 人脸检测与对齐 → 统一人脸匿名化 → 匿名图表情识别 → 桌面显示**。原始人脸不进入表情分类器，匿名处理只修改面部区域。

## 项目文件怎么找

```text
expression-deid/
├─ expression_deid/             Python 源码
│  ├─ preprocessing/            灰度、检测、对齐和数据准备
│  ├─ training/                 数据加载、训练与离线评估
│  ├─ runtime/                  匿名化、表情推理与视频处理
│  ├─ ui/                       PySide6 窗口和界面组件
│  ├─ models.py                 网络结构与检查点读写
│  ├─ config.py                 项目路径与处理参数
│  └─ desktop_ui.py             保留的桌面启动入口
├─ scripts/                     按职责分组的命令脚本
│  ├─ assets/  data/            下载资产、准备匿名训练数据
│  ├─ checks/                   工程、桌面、视频与训练检查
│  └─ distribution/             Windows 软件打包与单文件发行
├─ datasets/                    数据源与对齐结果
├─ configs/                     必用数据集与辅助集上限、混训比例
├─ models/                      模型文件，按功能存放
│  ├─ detection/                检测与关键点权重
│  ├─ anonymization/            匿名化运行权重
│  ├─ expression/               表情运行权重及输入契约
│  └─ training/                 本地训练候选，不自动替换运行权重
├─ examples/videos/             随项目提供的示例视频
├─ outputs/                     本地缓存、日志及推理结果
├─ requirements/                基础、训练、匿名化与 UI 依赖
├─ tests/                       自动化测试
├─ third_party/                 第三方推理组件和许可证
├─ docs/                        工程说明、指标和交接资料
└─ reports/                     报告源码、图片及 PDF
```

[逐文件职责、训练产物和调用关系](docs/architecture/project-layout.md) · [模型清单](models/README.md) · [交接操作](docs/handoff.md) · [课程报告](reports/README.md)

## 启动桌面端

无需安装环境的 Windows x64 软件见 [v1.1.0 Release](https://github.com/zdcwn/expression-deid/releases/tag/v1.1.0)。

- **CPU 版**：下载 `ExpressionLab-CPU.exe`，双击启动，适合没有 NVIDIA 显卡的电脑。
- **GPU 版**：下载 `ExpressionLab-GPU-Start.exe` 和所有 `.part` 文件，放在同一目录，双击启动器。程序自动校验、合并，首次展开后启动；没有可用 NVIDIA 加速时回退 CPU。
- 两版使用相同模型，内置全部 **1,100 张带标签图片、4 段视频**和运行依赖，不需要安装 Python、下载模型或另放数据集。首次启动需要等待资源展开。

完整说明、适配范围和故障处理见 [软件使用说明](docs/distribution/使用说明.txt)；构建方法见 [Windows 发行文档](docs/distribution/README.md)。

## 从源码启动

```powershell
git lfs pull
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements/training.txt
python -m pip install -r requirements/anonymization.txt
python -m pip install -r requirements/ui.txt
python -m expression_deid.desktop_ui --device auto
```

以上训练依赖对应 NVIDIA CUDA 环境；CPU 开发环境可使用 `requirements/distribution-cpu.txt`。软件始终是本地 Python/PySide6 桌面端。图片页显示真实标签、匿名对照和预测；实时页支持摄像头、本地视频与内置示例；模型表现页可切换不同评价集合。

## 当前模型与训练数据

运行资源和 SHA-256 固定在 `configs/runtime_models.json`。候选训练文件不会自动替换运行模型。

| 模块 | 主数据集 | 辅助数据 | 运行权重 |
|---|---|---|---|
| 身份识别 | Pins Face Recognition | 无 | `models/identity/identity_pins.pt` |
| 人脸匿名化 | CelebA-HQ | 无 | `models/anonymization/anonymizer_celebahq.pt`，以及关键点条件生成组件 |
| 表情识别 | **FER-2013 原始七类标签** | KMU-FED | `models/expression/expression_current.pt` |

表情沿用四组卷积 CNN，在实际匿名输入上继续训练，抽样 FER-2013 **85%**、KMU **15%**。共运行 16 轮，按验证得分选中第 11 轮；后续五轮无改善后停止。输入、标签和划分校验记录见 [训练记录](docs/validation/expression-training.json)。匿名化和身份模型沿用已确认权重，本轮没有重新训练。

## 固定同一模型的跨数据集评价

| 数据集 | 划分与类别 | 样本数 | 准确率 |
|---|---|---:|---:|
| **KMU-FED** | **六类验证，不含中性** | **230** | **91.74%** |
| FER-2013 | 七类 PublicTest 验证 | 3,543 | 57.41% |
| RAF-DB | 七类跨数据集验证 | 1,595 | 66.58% |
| KMU-FED | 六类测试，不含中性 | 180 | 87.22% |
| FER-2013 | 七类 PrivateTest 测试 | 3,589 | 56.39% |
| RAF-DB | 七类测试 | 1,469 | 61.40% |

先固定模型，再评价各完整划分；不按数据集切换权重、不筛掉预测错误的图片。报告重点展示最高验证结果 **KMU 六类 91.74%**，它专指这 230 张验证样本。分类器始终输出七类，KMU 仅有六类真实标签。初始化权重含历史多域训练，因此当前留出统计按本轮适配回归记录，不宣称严格从未接触相关人员。完整混淆矩阵和输入一致性检查见 [评价记录](docs/validation/expression-validation.json)。

Pins 身份器在对齐原脸上的准确率为 94.27%，局部残差组件匿名后原身份命中率为 1.39%；该数值只属于残差组件的面部椭圆评价，见 [身份记录](docs/validation/identity-residual.json)。

## 训练与复测

先按 [数据目录说明](datasets/README.md) 准备 FER-2013 和 KMU 的 `prepared/` 数组，固定原始标签及划分。训练使用完整局部编辑、合成并重新对齐后生成的匿名缓存，原图不进入表情分类器。

```powershell
python scripts/data/prepare_runtime_expression_cache.py --dataset fer2013 kmu --data-root datasets --device cuda
python -m expression_deid train-expression --data-root datasets --device cuda
python scripts/checks/check_current_expression.py --data-root datasets --device cuda
```

继续训练的候选写入 `models/training/expression_fer2013/best.pt`，不会覆盖运行权重。评价默认还读取 RAF 的处理数组；若只具备主集和辅助集，可加 `--domains fer2013 kmu`。身份与匿名化的训练入口仍保留：

```powershell
python -m expression_deid train-identity --device cuda
python -m expression_deid train-anonymizer --identity-checkpoint models/training/identity_pins.pt --device cuda
python -m expression_deid infer --input example.png --aligned --device auto
```

`evaluate` 用于对齐残差组件的独立诊断；应用中的表情效果以 `check_current_expression.py` 对完整匿名运行输入的复测为准。

## 检查与输出

```powershell
python -m compileall -q expression_deid scripts tests
python -m unittest discover -s tests -p "test_*.py" -q
python scripts/checks/check_desktop_presentation.py --device cuda
```

推理输出、训练缓存和检查记录写入 `outputs/`，默认不提交。示例视频固定放在 `examples/videos/`，与运行中生成的文件分开。

## 输入约束

- 表情模型只接收匿名化后的灰度脸，失败时不使用原脸补齐。
- 不先识别原图表情再选择相似图片覆盖输入。
- 面部外区域、头发与背景保留灰度输入画面。
- 置信度来自匿名输入的真实模型输出。
