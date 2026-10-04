# Expression De-identification

座舱人脸去标识：去掉身份，保留表情。

> 交接入口：先看 [`handoff/README.md`](handoff/README.md)。它记录了当前分支、模型/数据路径、许可边界、已验证指标和新机器启动顺序。运行 `python scripts/create_handoff_bundle.py` 可重新生成路径盘点包 `handoff/dataset-paths.zip`。

摄像头里的脸先做去标识，再交给表情模型读疲劳和情绪。身份模型只做验收，不进车上的业务链路。

运行时链路固定为：原始视频帧 → 灰度化 → 面部匿名化 → 匿名图进入表情模型。原始图不会进入表情分类器，也不会先用原图表情去挑选匿名素材。头发、背景、座舱和路况像素保持不动。

## 数据准备

原图只读。KMU 先对齐成 256 灰度，写到 `datasets/kmu-fed-aligned/`，这个目录不进仓库。RAF 保持官方的 100×100，不再做几何处理。

像素归一化在读图时做，不写回文件。均值和方差用 RAF 训练集算一次，存在 `expression_deid/norm_stats.json`。验证集和 KMU 用同一组数。

```
python -m expression_deid.prepare
```

表情模型的输入走 `expression_deid.normalize.expression_input`，单通道 100×100。增强只在调用时显式打开，给 RAF 训练集用。KMU 的清单和失败名单在对齐目录的 `manifest.jsonl`。

补充训练使用 FER+ 的高一致性标注。`datasets/fer2013-hf/fer2013new.csv` 保存逐张图的投票，配对的 FER2013 像素/增强 CSV 也在同一目录；代码按原始行号配对，只保留至少 5 票支持同一类别的样本。不要把其他镜像的 pickle 训练/验证顺序直接和 FER+ CSV 拼接，它们的顺序可能不同。

PPB-Emo 的中央红外中性视频用 `python -m expression_deid prepare-ppb --frames-per-video 5` 准备。需要把 PPB 作为小比例域适配时，再显式加入 `train-expression-joint --ppb-fraction 0.08`；训练代码会按参与者划分，且把 PPB 测试人员留到最后报告，不会把连续帧混入训练集。

## 模型代码

模型代码已经按图片中的三条线拆开：

- `ExpressionNet`：以 RAF-DB 为通用底座，使用 FER+ 高一致性样本和 KMU 座舱图像做混合域训练的表情分类器；
- `IdentityNet`：只用于 KMU 身份验收，不进入业务链路；
- `AnonymizerUNet`：输入对齐人脸，输出同尺寸灰度匿名图。

训练依赖放在 `requirements-train.txt`，其中固定为 Windows/Python 3.13 可用的 CUDA 13.0 PyTorch wheel。先停止正在运行的 CPU 训练，再安装：

```text
python -m pip uninstall -y torch
python -m pip install -r requirements-train.txt

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

确认 `torch.cuda.is_available()` 输出 `True` 后，依次运行：

```text
python -m expression_deid prepare
python -m expression_deid train-expression --device cuda --output models/expression.pt
python -m expression_deid train-identity --device cuda --output models/identity.pt
python -m expression_deid train-anonymizer --device cuda --identity-checkpoint models/identity.pt --expression-checkpoint models/expression.pt --output models/anonymizer.pt
python -m expression_deid evaluate --device cuda --anonymizer-checkpoint models/anonymizer.pt --expression-checkpoint models/expression.pt --identity-checkpoint models/identity.pt --output-dir outputs/evaluation
```

当前推荐的表情训练命令是：

```text
python -m expression_deid train-expression-joint --device cuda --base-checkpoint models/expression_joint.pt --output models/expression.pt --epochs 8 --batch-size 128 --kmu-fraction 0.20 --ferplus-fraction 0.30 --ferplus-min-votes 5 --component-augment
```

离线旧模型可以继续用于对比，但 Python 实时演示使用单个表情模型和 `fast_mode`，避免每帧加载身份模型或多个表情模型。接口中的 `confidence` 是匿名图分类器的实际校准后验；实时链路不会生成“原图与匿名图共同支持度”，避免把原图信息偷偷带进表情判断。

单张图或一整个目录也可以用 GPU 推理：

```text
python -m expression_deid infer --device cuda --anonymizer-checkpoint models/anonymizer.pt --expression-checkpoint models/expression.pt --identity-checkpoint models/identity.pt --input datasets/kmu-fed --pattern "*.jpg" --output outputs/inference
```

输出目录会保存匿名图和 `results.json`；联合评估会保存每张图的记录以及 `evaluation.json`。身份模型和匿名化模型的检查点都带有模型类型、类别数、训练轮次和指标，避免加载错模型。

匿名化训练默认以原图表情模型的软输出蒸馏为主，辅助使用真实标签，避免把匿名图强行改成数据集标签而失去原图表情。若已有旧版匿名化检查点，建议用当前代码重新训练匿名化模型后再评估。

## 评估口径

当前评估报告只统计匿名图表情准确率，并在每条记录写入 `expression_input_source=anonymized_only`。旧结果文件如果包含原图字段，脚本会兼容读取；新链路不会虚构原图/匿名图一致率。身份模型只用于隐私诊断，不进入表情模型输入链路。

身份隐私诊断另外使用 LFW 的全部 1680 个身份作干扰库。每条样本把原始脸作为 1 个查询身份，再和 1680 个外部身份比较；`gallery_original_match_rate` 越低越好，`gallery_source_rank` 不应长期为 1。比较输入先裁掉头发、背景和外部边角，只保留脸部椭圆区域。默认评估命令也使用全部 1680 个身份，可用 `--identity-gallery-size` 做较小的冒烟测试。

当前默认 Python/前端运行配置使用 `models/anonymizer_adversarial_v1.pt`、`models/expression_machine_selector_texture.pt`，视频路径采用固定的局部机器匿名候选以保持速度；完整 1100 张验收报告在 `outputs/evaluation_machine_runtime_fast_strong_1681/evaluation.json`。报告中表情准确率为 86.55%，1680 人图库原身份匹配率为 34.45%。离线 SFace 候选筛选报告 `outputs/evaluation_machine_selector_texture_adapted_1681/evaluation.json` 仍保留作更强隐私对照。

## Python 桌面应用（当前主入口）

当前应用端是三页式 PySide6/Qt 桌面 UI，不依赖网页。它包含实验图片对照、实时摄像头/视频、整体模型表现三部分。视频和摄像头页显示的是匿名化和表情模型完整推理链路的真实分析 FPS，不把界面刷新率冒充模型帧率。机器匿名化每帧运行，人眼展示用的 landmark-only GAN 每 15 帧刷新一次；在 RTX 3060 Laptop GPU 上，当前双分支实测约 24.9 FPS。先安装桌面依赖，再用 CUDA 启动：

```text
# 三页式桌面 UI
python -m pip install -r requirements-ui.txt
python -m expression_deid.desktop_ui --device cuda

# 摄像头（默认 fast，优先实时帧率）
python -m expression_deid.app --mode camera --device cuda --detect-interval 5

# 摄像头高质量模式（landmark-only 生成器，速度会下降）
python -m expression_deid.app --mode camera --device cuda --stream-quality quality

# 车载视频文件
python -m expression_deid.app --mode video --input <video.mp4> --device cuda --detect-interval 5

# 单张图片原图/匿名图对照
python -m expression_deid.app --mode image --input datasets/kmu-fed-aligned/01_AN_mr_001.png --device cuda

# 六/七类 KMU 示例图片，按 1–7 切换，按 n/空格看下一张
python -m expression_deid.app --mode examples --device cuda
```

按 `q` 或 `Esc` 退出。应用端不加载身份诊断模型，身份/SFace/1681 人库只在离线评估和单张图验收中使用，避免阻塞实时视频。

示例视频位于 `outputs/example_kmu_expressions.mp4`，桌面 UI 的实时页可以直接选择播放。

## Python 视频演示

Python 演示是主入口，不依赖网页。它输出原始灰度与匿名化视频的并排 MP4，同时输出逐帧 JSONL，记录匿名图表情、置信度和输入来源。检测器默认每 5 帧更新一次，其他帧复用关键点以提高速度：

```text
python -m expression_deid.video_demo --input outputs/test_visible_face_h264.mp4 --output outputs/video_demo.mp4 --telemetry outputs/video_demo.jsonl --device cuda --detect-interval 5
```

演示会打印实际处理 FPS。车载摄像头如果能被 PyAV 打开，可把同样的输入路径换成对应采集设备；视频文件模式是可复现的验收方式。

直接打开本机摄像头可以用：

```text
python -m expression_deid camera --index 0 --device cuda --detect-interval 5
```

按 `q` 或 `Esc` 退出。摄像头窗口和文件视频共用同一条匿名后识别链路。

如果要复现论文里“只由 478 点关键点驱动生成器”的高质量路线，可以额外准备 GANonymization 源码、MediaPipe 和训练好的 checkpoint：

```text
python -m pip install -r requirements-anonymization.txt
python scripts/download_anonymization_assets.py
python -m expression_deid video --input <video> --output outputs/video_ganonymization.mp4 --device cuda --ganonymization-checkpoint <checkpoint.ckpt> --ganonymization-repo <GANonymization目录>
```

项目检测到 `models/ganonymization_50.ckpt` 和 `external_ganonymization/` 时，`video` 命令会自动启用这个模式；显式传 `--ganonymization-checkpoint` 可以替换权重。生成器只收到 478 点关键点图，输出只贴回检测到的人脸内椭圆，头发和检测框外背景不改。实际处理速度会随摄像头分辨率、GPU 和人脸检测间隔变化；如果未准备权重才回退到局部高频特征模式。

## 本地演示前端

网页保留为辅助检查入口。它复用同一条匿名图单输入链路，支持上传图像、选择 KMU 样例和逐帧视频预览；正式性能验收请使用上面的 Python 演示。

```text
python -m expression_deid.web --device cuda
```

然后打开 `http://127.0.0.1:7860`。上传的图像只在内存中处理，不写入 `outputs/`；服务的静态文件只允许访问 `frontend/`，不会通过网页接口暴露模型检查点或数据集目录。

KMU 座舱图像和 RAF-DB 存在明显域差。若用自己的座舱数据继续训练，先运行 `train-expression-joint` 做 RAF + KMU 混合训练。它会同时报告 RAF 验证准确率和按序列隔离的 KMU 验证准确率，避免只在座舱数据上微调后忘掉通用表情：

```text
python -m expression_deid train-expression-joint --device cuda --base-checkpoint models/expression_raf.pt --output models/expression.pt
python -m expression_deid train-anonymizer --device cuda --identity-checkpoint models/identity.pt --expression-checkpoint models/expression.pt --output models/anonymizer.pt
```

如果只想做 KMU 域诊断，仍可使用 `train-expression-kmu`，但它会牺牲 RAF 通用验证效果，不建议把它作为默认业务模型。

默认 Python 视频演示优先使用 landmark-only GAN：生成器看不到原始像素，匿名脸只在原检测人脸的内椭圆中合成回去，检测框外像素经过回归测试保持完全一致。为了保住表情，眼睛、鼻子和嘴部的源几何只在合成后的匿名脸局部做低频结构约束；表情分类仍然只读最终匿名帧。没有 GAN 权重时才使用局部高频纹理编辑回退路径。
