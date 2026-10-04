# Expression De-identification 交接说明

这份交接说明对应 `gj-dev` 分支。项目目标是：在车载摄像头视频中先把画面转为灰度，再只修改人脸局部，最后把匿名结果送入表情分类器。原始图像不能进入表情分类器；头发、背景、座舱和路况区域保持原样。

## 当前状态

- **桌面端**：`python -m expression_deid.desktop_ui --device cuda`，PySide6/Qt 三页界面，包含实验图片、实时视频/摄像头和整体模型表现。
- **网页端**：`python -m expression_deid.web --device cuda`，是辅助检查入口，本次交接不依赖它。
- **机器表情链路**：快速局部匿名化结果 → 表情模型；原图不会进入表情模型。
- **人眼展示链路**：桌面/网页展示使用 `models/ganonymization_50.ckpt` + `external_ganonymization/` 的 landmark-only 生成器。生成器只接收关键点图，不接收原始脸部像素。
- **实时桌面视频**：机器链路每帧推理；视觉 GAN 每 15 帧刷新一次，再合成回当前灰度背景，保持实时性和视觉模型效果。
- **身份模型**：只用于离线验收和实验图片诊断，不进入业务表情推理链路。

当前已有验收报告：

- 报告：`outputs/evaluation_machine_runtime_fast_strong_1681/evaluation.json`
- 样本数：1100
- 匿名图表情准确率：86.55%
- 1681 人身份图库原身份命中率：34.45%
- 匿名后身份闭集准确率：5.73%
- SFace 平均余弦相似度：0.3798
- RTX 3060 Laptop GPU 上，桌面双分支实时链路实测约 24.9 FPS；首次加载和首帧会明显更慢。

这些数字是当前 KMU/本地模型配置的离线结果，不等于真实道路实车验收。近红外、角度、眼镜、遮挡和连续视频仍需用目标车载相机重新验证。

## 接手前先确认的目录

数据、模型和输出没有全部提交到 Git：数据约 2 GB，模型含数百 MB checkpoint，且外部数据有各自许可。路径盘点在：

- `handoff/asset_manifest.json`
- `handoff/asset_manifest.csv`
- `handoff/dataset-paths.zip`

其中记录了当前机器的绝对路径、仓库相对路径、文件数、总大小、用途和许可备注。换机器时以 `relative_path` 为准，把资源放到仓库对应位置即可。重新盘点运行：

```powershell
python scripts/create_handoff_bundle.py
```

必须准备的本地数据：

| 路径 | 用途 |
| --- | --- |
| `datasets/kmu-fed/` | KMU 近红外源图，去标识和验收 |
| `datasets/raf-db/` | 通用表情训练 |
| `datasets/fer2013-hf/` | FER+ 高一致性标签补充训练 |
| `datasets/kmu-fed-aligned/` | KMU 256 灰度对齐结果；缺失时可重建 |
| `datasets/lfw-sklearn/` 或 `datasets/lfw-gallery/` | 1680 人身份诊断库 |

PPB-Emo 是可选的近红外中性域适配数据：`datasets/ppb-emo/`。它不是真实道路数据，只有在需要补充中性类时使用。

必须准备的模型/资产：

| 路径 | 用途 |
| --- | --- |
| `models/anonymizer_adversarial_v1.pt` | 快速机器匿名化 |
| `models/expression_machine_selector_texture.pt` | 匿名图表情识别 |
| `models/ganonymization_50.ckpt` | 人眼展示用视觉匿名化 |
| `models/face_landmarker.task` | 478 点关键点条件 |
| `models/face_detection_yunet_2023mar.onnx` | 人脸检测 |
| `models/identity_faceonly.pt` | 实验图片/离线身份诊断 |
| `models/face_recognition_sface_2021dec.onnx` | SFace 相似度诊断 |

`external_ganonymization/` 是外部 GANonymization 源码目录，当前被 Git 忽略。需要按下面的依赖安装说明获取，不能把它当作训练数据。

## 新机器快速启动

下面以 Windows PowerShell、Python 3.13 和 NVIDIA CUDA 环境为例。

```powershell
git clone https://github.com/zdcwn/expression-deid.git
cd expression-deid
git switch gj-dev

python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-train.txt
python -m pip install -r requirements-anonymization.txt
python -m pip install -r requirements-ui.txt
```

验证 GPU：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

获取外部视觉匿名化代码和模型资产：

```powershell
git clone https://github.com/hcmlab/GANonymization.git external_ganonymization
python scripts/download_anonymization_assets.py
```

准备 KMU 对齐图和 LFW 身份图库：

```powershell
python -m expression_deid.prepare
python scripts/build_lfw_gallery.py --identities 1680
```

准备 PPB-Emo 中性帧（可选）：

```powershell
python -m expression_deid prepare-ppb --frames-per-video 5
```

运行代码检查：

```powershell
python -m compileall -q expression_deid
python -m unittest discover -s tests -q
python -m expression_deid.desktop_ui --device cuda
```

## 主要操作

启动桌面端：

```powershell
python -m expression_deid.desktop_ui --device cuda
```

第一页选择 KMU 样本，可以查看原图、展示匿名图、匿名图表情和身份诊断。第二页可以打开摄像头、选择本地视频或播放 `outputs/example_*.mp4`。第三页读取当前离线报告。

命令行视频验收：

```powershell
python -m expression_deid.video_demo `
  --input outputs/example_kmu_expressions.mp4 `
  --output outputs/video_demo_check.mp4 `
  --telemetry outputs/video_demo_check.jsonl `
  --device cuda --detect-interval 5
```

单图验收：

```powershell
python -m expression_deid.app --mode image `
  --input datasets/kmu-fed-aligned/01_AN_mr_001.png --device cuda
```

## 继续训练和评估

推荐先做混合表情训练，再训练或替换匿名化模型：

```powershell
# 如果交接包里没有 expression_raf.pt，先生成 RAF 基线：
python -m expression_deid train-expression `
  --device cuda --output models/expression_raf.pt

python -m expression_deid train-expression-joint `
  --device cuda --base-checkpoint models/expression_raf.pt `
  --output models/expression_joint.pt --epochs 8 --batch-size 128 `
  --kmu-fraction 0.20 --ferplus-fraction 0.30 `
  --ferplus-min-votes 5 --component-augment

python -m expression_deid train-anonymizer `
  --device cuda --identity-checkpoint models/identity_faceonly.pt `
  --expression-checkpoint models/expression_joint.pt `
  --output models/anonymizer_next.pt
```

评估时建议使用完整 1680 人图库，先做小规模 smoke test 再跑完整报告：

```powershell
python -m expression_deid evaluate --device cuda `
  --anonymizer-checkpoint models/anonymizer_adversarial_v1.pt `
  --expression-checkpoint models/expression_machine_selector_texture.pt `
  --identity-checkpoint models/identity_faceonly.pt `
  --identity-gallery-size 1680 `
  --output-dir outputs/evaluation_next
```

评估重点看：

1. `expression_anonymized_accuracy`：匿名图表情准确率；
2. `identity_anonymized_accuracy`：身份模型在匿名图上的闭集准确率，越低越好；
3. `gallery_original_match_rate`：1680 人图库命中率，越低越好；
4. `sface_mean_cosine`：脸部紧裁区域的相似度，结合可视效果一起看；
5. `expression_input_source`：必须是 `anonymized_only`。

## 不要破坏的约束

- 原始图像不能进入表情分类器。
- 处理顺序固定为：灰度化 → 局部人脸匿名化 → 仅匿名结果表情识别。
- 匿名化不能直接拿另一张完整脸替换；头发、背景、头部轮廓和座舱区域要保留。
- 身份模型和 SFace 只做诊断，不能参与表情标签、donor 选择或业务推理。
- 每次改代码后至少运行 `compileall`、相关单元测试和一次接口/模型路径检查。
- 不要把 `models/`、真实用户图片、完整外部数据集和 `.venv-anon/` 提交到 Git。

## 当前已知限制

- 当前数据主要是 KMU 固定车内近红外图，不能代表行驶中真实道路光照、振动和遮挡。
- PPB-Emo 是固定模拟器中性数据，只能做域适配补充。
- 视觉 GAN 生成器较重，桌面实时页采用每 15 帧刷新一次展示匿名图；表情分析仍按机器匿名链路逐帧执行。
- 如果没有 `external_ganonymization/`、`ganonymization_50.ckpt` 或 `face_landmarker.task`，程序会回退到局部编辑，视觉效果会明显变弱。
- 训练和完整身份评估都需要 CUDA；CPU 仅适合代码检查和小规模 smoke test。
