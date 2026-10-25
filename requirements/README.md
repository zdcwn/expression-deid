# 依赖分组

| 文件 | 安装用途 |
|---|---|
| `base.txt` | 数组、图像和基础预处理 |
| `training.txt` | 基础依赖、PyTorch、视频处理 |
| `anonymization.txt` | 匿名化所需的关键点与图像依赖 |
| `ui.txt` | PySide6 原生桌面界面 |
| `quality-training.txt` | 质量模型训练及导出所需的 ONNX 转换组件 |

按根目录 README 安装前三类运行依赖即可启动桌面端。质量训练额外先安装 `training.txt`，再安装 `quality-training.txt`。各文件中的 `-r` 引用按本目录相对路径解析。
