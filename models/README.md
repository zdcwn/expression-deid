# 本地模型资产

模型权重不提交到 Git。当前机器的完整盘点在 [`../handoff/asset_manifest.csv`](../handoff/asset_manifest.csv)，新机器按相对路径放置权重即可。

默认模型资产：

- `anonymizer_adversarial_v1.pt`：机器匿名化分支；
- `expression_realtime_stable.pt`：2026-10-09 审计后微调的桌面快速视频表情分类器，缺少时回退到下一项；
- `expression_machine_selector_texture.pt`：单图/GAN 质量模式分类器及旧实时基线；
- `ganonymization_50.ckpt`：人眼展示用 landmark-only GAN；
- `face_landmarker.task`：GAN 的密集关键点模型；
- `face_detection_yunet_2023mar.onnx`：人脸检测；
- `identity_faceonly.pt` 和 `face_recognition_sface_2021dec.onnx`：离线身份/SFace 诊断。

缺少视觉 GAN 的三个文件时，程序会回退到局部编辑，表情链路仍可运行，但展示效果会明显变弱。运行 `python scripts/download_anonymization_assets.py` 可下载公开的 GAN checkpoint、MediaPipe landmark 和 SFace 资产；项目训练得到的 `.pt` 权重需要从交接机器或训练产物中复制。

最新权重大小 3,495,171 字节，SHA-256 为 `5a98c02d10439dd215cc4b6f59abf7bbbf174fd2df0396841b0551fd9c1a26ab`。当前默认匿名链路尚未通过独立身份隐私验收。训练、逐模型检查、复测和限制见 [`../docs/full-audit-20261009.md`](../docs/full-audit-20261009.md)。旧权重记录保留在 [`../docs/realtime-expression-20261008.md`](../docs/realtime-expression-20261008.md)。运行 `python scripts/train_realtime_expression.py --device cuda` 生成候选，验证后再部署，不要直接覆盖原权重。
