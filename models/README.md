# 本地模型资产

模型权重不提交到 Git。当前机器的完整盘点在 [`../handoff/asset_manifest.csv`](../handoff/asset_manifest.csv)，新机器按相对路径放置权重即可。

桌面端和网页端的默认模型：

- `anonymizer_adversarial_v1.pt`：机器匿名化分支；
- `expression_machine_selector_texture.pt`：匿名图表情分类器；
- `ganonymization_50.ckpt`：人眼展示用 landmark-only GAN；
- `face_landmarker.task`：GAN 的密集关键点模型；
- `face_detection_yunet_2023mar.onnx`：人脸检测；
- `identity_faceonly.pt` 和 `face_recognition_sface_2021dec.onnx`：离线身份/SFace 诊断。

缺少视觉 GAN 的三个文件时，程序会回退到局部编辑，表情链路仍可运行，但展示效果会明显变弱。运行 `python scripts/download_anonymization_assets.py` 可下载公开的 GAN checkpoint、MediaPipe landmark 和 SFace 资产；项目训练得到的 `.pt` 权重需要从交接机器或训练产物中复制。
