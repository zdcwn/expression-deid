# 主分支交接说明

这个版本只保留本地 Python 桌面应用、当前运行模型、必要源数据和示例视频。网页端、逐样本审计 JSON、历史候选权重、身份诊断图库、训练缓存和用户测试图片没有提交。

## 启动

```powershell
git lfs pull
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-train.txt
python -m pip install -r requirements-anonymization.txt
python -m pip install -r requirements-ui.txt
python -m expression_deid.desktop_ui --device cuda
```

桌面端的实时页有高速和质量两种模式。高速模式只加载轻量匿名识别管线；质量模式另外加载质量分类器和视觉 GAN。两条路径都遵守匿名图先于表情识别的输入边界。

## 目录约定

| 路径 | 内容 |
| --- | --- |
| `expression_deid/` | 对齐、匿名化、表情识别、视频和 Qt UI 代码 |
| `datasets/kmu-fed/` | KMU 车内近红外源图 |
| `datasets/raf-db/` | RAF-DB 表情训练/验证图 |
| `datasets/kmu-fed-aligned/` | 256 灰度对齐图，缺失时运行 `python -m expression_deid.prepare` |
| `models/` | 当前运行模型和质量模式输入契约 |
| `outputs/example_*.mp4` | 桌面实时页示例视频 |
| `docs/model-metrics.json` | 桌面表现页使用的精简静态指标 |

## 模型清单

- `anonymizer_adversarial_v1.pt`：机器匿名化局部编辑；
- `expression_realtime_stable.pt`：高速模式表情模型；
- `expression_machine_selector_texture.pt`：图片实验表情模型；
- `expression_video_quality.ts.pt` 与 JSON：质量模式表情模型和哈希契约；
- `face_detection_yunet_2023mar.onnx`：人脸检测；
- `face_landmarker.task`、`ganonymization_50.ckpt`：质量模式展示匿名化。

GAN 权重由 Git LFS 管理。若只需要高速模式，可以不下载 GAN 和 MediaPipe 资产；程序会回退到局部编辑显示，表情识别仍使用匿名结果。

## 验证

```powershell
python -m compileall -q expression_deid scripts
python -m unittest discover -s tests -p "test_*.py" -q
python scripts/check_desktop_presentation.py --device cuda
```

验证只代表源码、Qt 页面、示例素材和本地模型链路。真实道路、陌生驾驶员和目标车载相机仍需单独验收。

## 必须保持

1. 灰度化后才做面部局部匿名化；
2. 原始图像不得进入表情分类器；
3. 匿名化不能替换整张脸；
4. 背景、发型、头部轮廓和面部外区域保持不变；
5. 不把训练缓存、逐帧结果、用户照片和临时报告提交到主分支。
