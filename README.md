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
│  ├─ training/  export/        训练候选、导出质量模型
│  └─ checks/                   工程、桌面、视频与训练检查
├─ datasets/                    数据源与对齐结果
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

在项目根目录执行，先用 Git LFS 拉取匿名化权重：

```powershell
git lfs pull
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements/training.txt
python -m pip install -r requirements/anonymization.txt
python -m pip install -r requirements/ui.txt
python -m expression_deid.desktop_ui --device cuda
```

没有 CUDA 时使用 `--device cpu`。质量模式的表情输入契约位于 `models/expression/expression_video_quality.json`，运行权重必须与其中的 SHA-256 一致。

图片页提供带真实标签的原图与匿名图对照。实时页支持摄像头、本地视频和四个示例视频，可选择高速或质量模式；显示的分析 FPS 来自模型处理计时。模型表现页读取 `docs/model-metrics.json`。

## 数据与训练

数据说明见 [datasets/README.md](datasets/README.md)。缺少对齐结果时执行：

```powershell
python -m expression_deid prepare
```

训练通用表情网络的例子：

```powershell
python -m expression_deid train-expression --device cuda
```

默认写入 `models/training/expression.pt`。其余训练命令、匿名缓存、质量模型导出及检查流程见[工程目录与操作路径](docs/architecture/project-layout.md)。候选权重先验证，再明确部署到相应运行目录；桌面端不会自动选取训练目录里的新文件。

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
