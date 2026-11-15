# 运行模型

网络代码位于 `expression_deid/models.py`；固定权重和摘要见 `configs/runtime_models.json`。

| 文件 | 作用 |
|---|---|
| `detection/face_detection_yunet_2023mar.onnx` | 人脸框和五点检测 |
| `detection/face_landmarker.task` | 密集关键点 |
| `anonymization/anonymizer_celebahq.pt` | 已确认的 CelebA-HQ 局部残差权重 |
| `anonymization/ganonymization_50.ckpt` | 关键点条件生成组件，Git LFS 管理 |
| `identity/identity_pins.pt` | 已确认的 Pins 105 类身份模型 |
| `expression/expression_current.pt` | 唯一运行表情 CNN：FER-2013 主训练，KMU 辅助 |

表情输入为实际匿名合成结果重新对齐得到的 100×100 灰度脸，均值 0.481619、标准差 0.242606，验证校准温度 1.1。当前检查点 SHA-256 为 `f747deacc851b6b8d48beee05e58b095f75196d487263f5c9cd61eb6d61727fb`。图片、视频和软件发行使用同一检查点，分类器不读取原脸或显示分支的生成图。

匿名化与身份权重本轮未变。旧实时/质量表情资产与停用入口已移出运行目录；训练候选写入 `models/training/`，验证后再明确部署。
