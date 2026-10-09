# Expression De-identification

座舱人脸去标识：去掉身份，保留表情。

> 交接入口：先看 [`handoff/README.md`](handoff/README.md)。它记录了当前分支、模型/数据路径、许可边界、已验证指标和新机器启动顺序。运行 `python scripts/create_handoff_bundle.py` 可重新生成路径盘点包 `handoff/dataset-paths.zip`。

摄像头里的脸先做去标识，再交给表情模型读疲劳和情绪。身份模型只做验收，不进车上的业务链路。

运行时链路固定为：原始视频帧 → 灰度化 → 面部匿名化 → 匿名图进入表情模型。原始图不会进入表情分类器，也不会先用原图表情去挑选匿名素材。面部掩膜外的灰度像素保持不动；进入掩膜的头发/遮挡物尚未做语义分割，仍需人工检查。

## 数据准备

原图只读。KMU 先对齐成 256 灰度，写到 `datasets/kmu-fed-aligned/`，这个目录不进仓库。RAF 保持官方的 100×100，不再做几何处理。

像素归一化在读图时做，不写回文件。均值和方差用 RAF 训练集算一次，存在 `expression_deid/norm_stats.json`。验证集和 KMU 用同一组数。

```
python -m expression_deid.prepare
```

表情模型的输入走 `expression_deid.normalize.expression_input`，单通道 100×100。增强只在调用时显式打开，给 RAF 训练集用。KMU 的清单和失败名单在对齐目录的 `manifest.jsonl`。

补充训练使用 FER+ 的高一致性标注。`datasets/fer2013-hf/fer2013new.csv` 保存逐张图的投票，配对的 FER2013 像素/增强 CSV 也在同一目录；代码按原始行号配对，只保留获胜类别至少 5 票且严格超过全部投票半数的样本，排除训练与 PublicTest/PrivateTest 的重复像素。不要把其他镜像的 pickle 训练/验证顺序直接和 FER+ CSV 拼接，它们的顺序可能不同。

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

离线旧模型可以继续用于对比，但 Python 实时演示使用单个表情模型和 `fast_mode`。实时输出采用匿名帧概率的时序平滑与持续证据判断；`confidence` 是显示类别的连续帧概率，不确定时为空。遥测中的 `raw_*` 是匿名图的单帧输出，原图不会进入表情分类器。

单张图或一整个目录也可以用 GPU 推理：

```text
python -m expression_deid infer --device cuda --anonymizer-checkpoint models/anonymizer.pt --expression-checkpoint models/expression.pt --identity-checkpoint models/identity.pt --input datasets/kmu-fed --pattern "*.jpg" --output outputs/inference
```

输出目录会保存匿名图和 `results.json`；联合评估会保存每张图的记录以及 `evaluation.json`。身份模型和匿名化模型的检查点都带有模型类型、类别数、训练轮次和指标，避免加载错模型。

匿名化训练默认只用匿名结果和真实标签计算表情损失，不运行原图表情教师；身份对抗损失使用与验收一致的脸部裁剪。训练和验证采用共享的 KMU 序列尾帧划分，验证图不参与更新。旧匿名权重有训练口径差异，不能用新代码检查替代重新训练后的独立隐私验收。

## 评估口径

当前评估报告只统计匿名图表情准确率，并在每条记录写入 `expression_input_source=anonymized_only`。旧结果文件如果包含原图字段，脚本会兼容读取；新链路不会虚构原图/匿名图一致率。身份模型只用于隐私诊断，不进入表情模型输入链路。

身份隐私诊断另外使用 LFW 的全部 1680 个身份作干扰库。每条样本把原始脸作为 1 个查询身份，再和 1680 个外部身份比较；`gallery_original_match_rate` 越低越好，`gallery_source_rank` 不应长期为 1。比较输入先裁掉头发、背景和外部边角，只保留脸部椭圆区域。默认评估命令也使用全部 1680 个身份，可用 `--identity-gallery-size` 做较小的冒烟测试。

当前视频默认使用 `models/anonymizer_adversarial_v1.pt` 和质量优先的 `models/expression_video_quality.ts.pt` + JSON 清单。质量模型只接收匿名灰度人脸，使用 FP32 左右翻转 TTA；实测约 5–6 FPS，准确率优先于 24 FPS。缺少质量资产时桌面端才回退 `expression_realtime_stable.pt`。2026-10-09 详细复核见 [`docs/full-audit-20261009.md`](docs/full-audit-20261009.md)：质量模型在匿名 RAF 验证集达到 78.39%，中性召回 81.91%，中性误判成悲伤/愤怒 58/680；但独立 SFace 的原身份命中率仍为 96.36%，**默认匿名链路未通过隐私验收**。历史 GAN 结果不能套用到当前质量视频。

## Python 桌面应用（当前主入口）

当前应用端是三页式 PySide6/Qt 桌面 UI，不依赖网页。它包含实验图片对照、实时摄像头/视频、整体模型表现三部分。视频和摄像头页显示实际分析处理均速，不把界面刷新率冒充模型帧率。机器匿名化与表情识别每帧运行，展示 GAN 每 15 帧刷新；RTX 3060 Laptop GPU 上，640×480 视频预热后复测约 32–34 FPS，首次推理和生成刷新仍有耗时峰值。先安装桌面依赖，再用 CUDA 启动：

2026-10-09 桌面界面改为浅色工作区、深色导航和双画面对照。图片页提供缩略图与结果缓存；视频页在匿名画面叠加实际表情/置信度，并显示模型分析 FPS 曲线；模型表现页按 RAF、KMU、PPB 数据集切换，读取本次审计的保存记录，明确标注验证范围。支持 1100×760 及以上窗口；布局、字体与真实模型输出检查见 [`docs/desktop-design-20261009.md`](docs/desktop-design-20261009.md)。

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

OpenCV 演示窗口按 `q` 或 `Esc` 退出；Qt 桌面应用通过窗口关闭按钮退出。实时视频不加载身份诊断模型；桌面实验图片页与离线评估加载身份/SFace/1681 人库。身份诊断不能进入表情标签或原图表情教师链路。

示例视频位于 `outputs/example_kmu_expressions.mp4`，桌面 UI 的实时页可以直接选择播放。

2026-10-09 的视频表情优化加入匿名域全量训练、FP32 质量表征、左右翻转 TTA、温度校准和持续证据判断。尚未确认时显示“暂不确定”，不会硬猜中性。质量模型与快速模型都只使用匿名图；单图/GAN 质量模式保持原路径。训练、逐帧检查、权重哈希和未通过项见 [`docs/full-audit-20261009.md`](docs/full-audit-20261009.md)；[`docs/realtime-expression-20261008.md`](docs/realtime-expression-20261008.md) 保留为旧权重的历史记录。通用表情和用户本人摄像头仍需进一步验证；模型表现页呈现有日期的保存记录，不是当前摄像头的实时准确率。

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

默认 `video` 命令与桌面实时页一致：局部机器匿名化结果每帧识别，视觉 GAN 每 15 帧刷新。显式传 `--ganonymization-checkpoint` 才切换为 GAN 机器输入质量模式；这是不同输入域，需要单独验证对应分类器。生成器只接收 478 点关键点线图。内椭圆外、检测框外背景像素不变；未做人像语义分割，进入面部掩膜的头发和遮挡物仍需人工核对。

## 本地演示前端

网页保留为辅助检查入口。它复用同一条匿名图单输入链路，支持上传图像、选择 KMU 样例和逐帧视频预览；正式性能验收请使用上面的 Python 演示。

```text
python -m expression_deid.web --device cuda
```

然后打开 `http://127.0.0.1:7860`。上传的图像只在内存中处理，不写入 `outputs/`；服务的静态文件只允许访问 `frontend/`，不会通过网页接口暴露模型检查点或数据集目录。

KMU 座舱图像和 RAF-DB 存在明显域差。若用自己的座舱数据继续训练，先运行 `train-expression-joint` 做 RAF + KMU 混合训练。它会同时报告 RAF 验证准确率和按序列尾帧划分的 KMU 验证准确率，避免只在座舱数据上微调后忘掉通用表情：

```text
python -m expression_deid train-expression-joint --device cuda --base-checkpoint models/expression_raf.pt --output models/expression.pt
python -m expression_deid train-anonymizer --device cuda --identity-checkpoint models/identity.pt --expression-checkpoint models/expression.pt --output models/anonymizer.pt
```

如果只想做 KMU 域诊断，仍可使用 `train-expression-kmu`，但它会牺牲 RAF 通用验证效果，不建议把它作为默认业务模型。

默认 Python 视频演示使用每帧局部匿名化后识别的快速链路。密集关键点丢失时，展示只短时复用最近匿名结果，超过期限遮蔽面部；匿名化资源失败时不会把原图送进表情分类器。展示画面的视觉效果不能证明机器身份隐私达标。
