# 主分支模型清单

这份清单按主分支的 Python 入口盘点。主分支只保留桌面端当前会加载的运行模型、检测模型和质量模式所需的组件；训练候选、逐帧缓存、身份图库和网页代码不在主分支。

运行顺序固定为：视频或图片 → 灰度化 → 局部人脸匿名化 → 仅匿名灰度脸进入表情模型。匿名化只处理检测到的人脸区域，发型、头部轮廓、背景和座舱区域保持原画面。

## 当前运行模型

| 文件 | 用途 | 入口 |
| --- | --- | --- |
| `models/face_detection_yunet_2023mar.onnx` | 大图和视频的人脸检测 | `expression_deid/detect.py` |
| `models/anonymizer_adversarial_v1.pt` | 统一人脸匿名化模块的局部残差组件 | `app.py`、`desktop_ui_qt.py`、`video_demo.py` |
| `models/expression_realtime_stable.pt` | 高速视频的匿名图表情识别 | `app.py` 的快速路径 |
| `models/expression_machine_selector_texture.pt` | 图片实验和质量预览的匿名图表情识别 | `app.py`、桌面实验页 |
| `models/expression_video_quality.ts.pt` + `.json` | 质量模式的 FP32 表情分类器及输入契约 | `expression_deid/expression_quality.py` |
| `models/face_landmarker.task` | 质量模式的面部关键点 | `expression_deid/ganonymization_backend.py` |
| `models/ganonymization_50.ckpt` | 关键点条件匿名化组件，文件由 Git LFS 管理 | 桌面和视频质量预览 |

`external_ganonymization/` 只保留运行时需要的 `pix2pix.py` 和许可证文件。质量模式缺少关键点条件组件时，会回退到局部编辑路径；高速模式不依赖它。

## 不随运行提交的内容

旧匿名化候选、旧表情训练权重、训练中间产物、身份诊断权重、外部身份图库、实验输出和网页前端都留在开发机或独立资料包中。它们不会被主分支入口自动发现，也不会影响桌面端启动。

身份评估如果需要，应在主分支之外使用单独的离线实验脚本和独立权重完成；当前桌面运行链路没有身份识别器，表情模型只接收匿名化后的灰度脸。

## 权重和输入检查

`models/expression_video_quality.json` 记录质量表情模型的类别顺序、预处理方式、校准温度和 SHA-256。替换运行模型时，先更新这份契约并运行：

```powershell
python -m compileall -q expression_deid scripts
python -m unittest discover -s tests -p "test_*.py" -q
python scripts/check_desktop_presentation.py --device cuda
```

验证通过后再提交新的运行资产；不要把用户照片、逐帧结果或未核对的训练权重加入主分支。
