# 本地模型资产

模型权重不提交到 Git。当前机器的完整盘点在 [`../handoff/asset_manifest.csv`](../handoff/asset_manifest.csv)，新机器按相对路径放置权重即可。

默认模型资产：

- `anonymizer_adversarial_v1.pt`：机器匿名化分支；
- `expression_realtime_stable.pt`：2026-10-09 审计后微调的桌面快速视频表情分类器，缺少时回退到下一项；
- `expression_video_quality.ts.pt` + `expression_video_quality.json`：当前桌面视频准确优先模式。它用固定的 EmotiEffLib EfficientNet-B2 表征、匿名域微调、左右翻转 TTA 和验证集校准；桌面端发现这两个文件后自动使用它，缺少时才回退到快速旧模型。
- `expression_machine_selector_texture.pt`：单图/GAN 质量模式分类器及旧实时基线；
- `ganonymization_50.ckpt`：人眼展示用 landmark-only GAN；
- `face_landmarker.task`：GAN 的密集关键点模型；
- `face_detection_yunet_2023mar.onnx`：人脸检测；
- `identity_faceonly.pt` 和 `face_recognition_sface_2021dec.onnx`：离线身份/SFace 诊断。

缺少视觉 GAN 的三个文件时，程序会回退到局部编辑，表情链路仍可运行，但展示效果会明显变弱。运行 `python scripts/download_anonymization_assets.py` 可下载公开的 GAN checkpoint、MediaPipe landmark 和 SFace 资产；项目训练得到的 `.pt` 权重需要从交接机器或训练产物中复制。

当前质量优先 TorchScript 权重大小约 31.8 MB；清单记录了模型和匿名缓存的 SHA-256。它在同一 RAF 匿名验证集上达到 78.39% 准确率，中性召回 81.91%，中性误判成悲伤/愤怒 58/680；KMU 尾帧 99.55%，PPB 未见人员中性测试 100%（这些数据集范围不能代表你的摄像头准确率）。质量模式使用 FP32 左右翻转 TTA，实测单分类器速度低于旧模型，因此优先准确率，不承诺 24 FPS。当前默认匿名链路尚未通过独立身份隐私验收。训练/导出/复测命令见 `scripts/train_video_quality.py`、`scripts/export_video_quality.py` 和 `scripts/check_video_quality.py`。
