# 当前模型资产

主分支只保留桌面端当前会加载的运行模型和质量模式组件。旧训练候选、身份诊断权重、身份图库和实验中间权重不提交。

| 文件 | 用途 |
| --- | --- |
| `anonymizer_adversarial_v1.pt` | 局部人脸匿名化残差 |
| `expression_realtime_stable.pt` | 高速视频的匿名图表情识别 |
| `expression_machine_selector_texture.pt` | 图片实验和质量预览的表情识别 |
| `expression_video_quality.ts.pt` | FP32 质量表情识别 |
| `expression_video_quality.json` | 质量模型输入、类别和 SHA-256 契约 |
| `face_detection_yunet_2023mar.onnx` | YuNet 人脸检测 |
| `face_landmarker.task` | 质量模式面部关键点 |
| `ganonymization_50.ckpt` | 关键点条件匿名化组件，使用 Git LFS |

质量表情模型的输入必须是匿名化后的 `100×100` 灰度图，运行结果会带有 `expression_input_source=anonymized_only`。原始图像不能直接进入表情分类器。

质量模式需要 GAN 和关键点资产来生成展示预览；缺少这些可选组件时，高速局部编辑路径仍可运行。替换运行模型前，先更新输入契约和 SHA-256，然后运行编译、单元测试和桌面检查。

训练过程产生的 `.pt`、`.pth`、`.onnx`、`.ckpt` 默认留在本地模型目录。需要替换当前资产时，应先完成数据集验证，再单独提交经过核对的文件。
