# 当前运行模型

主分支只保留桌面端当前会加载的模型。旧训练候选、身份/SFace 诊断权重和实验中间权重不提交。

| 文件 | 运行位置 | 说明 |
| --- | --- | --- |
| `anonymizer_adversarial_v1.pt` | 高速/质量匿名链路 | 局部面部匿名化残差 |
| `expression_realtime_stable.pt` | 高速视频、质量视频的 compact 分支 | 匿名灰度图表情识别 |
| `expression_machine_selector_texture.pt` | 图片实验 | 匿名图质量模式表情识别 |
| `expression_video_quality.ts.pt` | 质量视频 | FP32 质量分类器 |
| `expression_video_quality.json` | 质量视频 | 类别、预处理、温度和 SHA-256 契约 |
| `face_detection_yunet_2023mar.onnx` | 原始视频检测 | YuNet 人脸检测 |
| `face_landmarker.task` | 质量模式展示 | MediaPipe 密集关键点 |
| `ganonymization_50.ckpt` | 质量模式展示 | landmark-only GAN，Git LFS 文件 |

质量表情模型的输入必须是匿名化后的 `100×100` 灰度 `uint8` 图像。原始图像、原图表情标签和展示分支都不能进入该分类器。

GAN 只接收关键点条件图，原始脸部像素不进入生成器。删除 GAN/LFS 资产后，高速模式仍可运行；质量模式会自动回退到局部编辑显示。

训练过程产生的 `.pt`、`.pth`、`.onnx`、`.ckpt` 默认不应直接加入 Git。需要替换运行模型时，先更新输入契约、SHA-256 和本地检查，再单独提交经过验证的当前资产。
