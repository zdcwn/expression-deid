# Expression Deid

车载人脸去标识与表情分析的本地 Python 项目。运行顺序固定为：

`视频/图片 → 灰度化 → 局部人脸匿名化 → 仅匿名灰度脸进入表情模型`

原始图像不会进入表情分类器，也不会先识别原图表情再挑选替换图。匿名化只修改面部局部，头发、头部轮廓、背景和座舱区域保持原画面。

## 当前版本

主入口是 PySide6 桌面端，不包含网页服务。桌面端有三页：

- 图片实验：KMU 带标签样本、原始灰度图/匿名图对照、匿名图表情和置信度；
- 实时分析：摄像头、本地视频、示例视频，高速模式和质量模式；
- 模型表现：读取 `docs/model-metrics.json` 的静态汇总，不保存逐样本实验产物。

当前主分支只保留运行所需的模型、源数据、示例素材和代码。训练中间权重、逐帧结果、身份诊断库、旧候选权重和网页代码都不在主分支。

## 快速启动

Windows PowerShell：

```powershell
git clone https://github.com/zdcwn/expression-deid.git
cd expression-deid
git lfs pull
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-ui.txt
python -m expression_deid.desktop_ui --device cuda
```

如果没有 CUDA，把最后一行的 `--device cuda` 改成 `--device cpu`。CPU 适合检查界面和小样本，不适合实时视频。

验证安装：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -m compileall -q expression_deid
python -m unittest discover -s tests -p "test_*.py" -q
```

## 模型资产

主分支直接保留这些当前运行资产：

| 路径 | 用途 |
| --- | --- |
| `models/anonymizer_adversarial_v1.pt` | 机器匿名化局部编辑 |
| `models/expression_realtime_stable.pt` | 高速模式匿名图表情模型 |
| `models/expression_machine_selector_texture.pt` | 图片实验质量模式表情模型 |
| `models/expression_video_quality.ts.pt` + `.json` | 质量模式 FP32 表情模型及输入契约 |
| `models/face_detection_yunet_2023mar.onnx` | 人脸检测 |
| `models/face_landmarker.task` | 视觉匿名化的密集关键点 |
| `models/ganonymization_50.ckpt` | 质量模式的人眼展示匿名化 GAN，使用 Git LFS |

旧的 `.pt`、身份模型、SFace 模型和训练候选权重留在开发机或外部模型包，不参与主分支运行。`external_ganonymization/` 只保留推理所需的 MIT 许可 `pix2pix.py` 和许可证文件。

质量模式的表情分类器只接受已经匿名化的 `100×100` 灰度脸；输入契约和权重哈希记录在 `models/expression_video_quality.json`。

## 数据

主分支保留可公开放置的源数据：

- `datasets/kmu-fed/`：车内近红外表情源图，图片实验和匿名化回归使用；
- `datasets/raf-db/`：七类通用表情训练/验证图；
- `datasets/kmu-fed-aligned/`：本地生成的 256 灰度对齐图，缺失时由准备命令重建。

完整数据集仍受原始数据许可约束。不要把 PPB、LFW、真实用户图片或新生成的中间结果直接提交到仓库。

如果对齐目录缺失：

```powershell
python -m expression_deid.prepare
```

## 桌面端操作

```powershell
python -m expression_deid.desktop_ui --device cuda
```

实时页的“高速模式”只加载轻量匿名识别管线，适合摄像头；“质量模式”额外加载质量分类器和 GAN，启动更慢但优先准确性。视频文件按源视频 FPS 播放，界面中的 FPS 卡片只统计匿名化和表情模型处理速度，不把界面刷新率冒充模型帧率。

图片页分析过的样本会复用已加载的推理管线。示例视频位于 `outputs/example_*.mp4`，当前版本为 24 FPS，时长分别为 15 秒、12 秒、3 秒和 3 秒。

## Python 命令行

单张图片或目录：

```powershell
python -m expression_deid.app --mode image `
  --input datasets/kmu-fed-aligned/01_AN_mr_001.png --device cuda
```

视频：

```powershell
python -m expression_deid.video_demo `
  --input outputs/example_kmu_expressions.mp4 `
  --output outputs/video_demo.mp4 `
  --telemetry outputs/video_demo.jsonl `
  --device cuda --detect-interval 5
```

## 训练与评估

训练依赖是可选的：

```powershell
python -m pip install -r requirements-train.txt
python -m pip install -r requirements-quality-train.txt
```

训练脚本会把匿名缓存标记为 `anonymized_only`，原始脸不能作为表情教师输入。新权重应放在本地模型目录，先通过验证脚本后再替换运行权重。

## 代码检查

每次改动后至少运行：

```powershell
python -m compileall -q expression_deid scripts
python -m unittest discover -s tests -p "test_*.py" -q
python scripts/check_desktop_presentation.py --device cuda
```

这些检查验证代码、Qt 页面、真实示例图片/视频和匿名结果输入来源；它们不等同于实车、摄像头陌生人准确率或道路环境验收。

## 不要破坏的约束

- 原始图像不能进入表情分类器；
- 顺序必须是灰度化、局部匿名化、匿名图表情识别；
- 不得用整张别人的脸替换原图；
- 背景、头发、头部轮廓和面部外区域保持不变；
- 置信度必须来自匿名图的真实模型输出；
- 不提交模型训练中间产物、逐帧结果、用户照片、虚拟环境或网页代码。
