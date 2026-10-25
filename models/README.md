# 模型文件

这里存放权重和模型输入契约。网络结构的 Python 代码在 `expression_deid/models.py`，两者职责不同。

## 运行权重

| 相对本目录的路径 | 作用 | 加载代码 |
|---|---|---|
| `detection/face_detection_yunet_2023mar.onnx` | 人脸框与五点检测 | `preprocessing/detect.py` |
| `detection/face_landmarker.task` | 密集关键点 | `runtime/anonymization.py` |
| `anonymization/anonymizer_adversarial_v1.pt` | 统一匿名化的局部残差组件 | `ui/opencv_demo.py`、`runtime/pipeline.py` |
| `anonymization/ganonymization_50.ckpt` | 关键点条件生成组件，Git LFS 管理 | `runtime/anonymization.py` |
| `expression/expression_realtime_stable.pt` | 高速视频表情权重 | `ui/opencv_demo.py` |
| `expression/expression_machine_selector_texture.pt` | 图片实验表情权重 | `ui/opencv_demo.py` |
| `expression/expression_video_quality.ts.pt` | 质量表情导出模型 | `runtime/expression.py` |
| `expression/expression_video_quality.json` | 类别、预处理、温度与权重哈希 | `runtime/expression.py` |

“加载代码”列省略了共同前缀 `expression_deid/`。质量模型只接受匿名化后的 `100×100 uint8` 灰度脸；JSON 与其引用的两个表情权重保持同一目录。

## 训练产物

`training/` 保存本地候选与训练阶段检查点，默认不提交。常规训练的 `--output` 默认都指向这里；质量训练可通过 `--output models/training/quality_run` 生成一组候选与验证记录。

训练不会直接覆盖上述运行权重。经过匿名输入验证、数据集评价和桌面检查后，才将选中的权重部署到对应运行目录。涉及质量模型时，须同时更新 JSON 及 SHA-256，不只复制一个 `.pt` 文件。
