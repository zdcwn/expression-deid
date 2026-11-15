# 依赖分组

| 文件 | 安装用途 |
|---|---|
| `base.txt` | 数组、图像和基础预处理 |
| `training.txt` | 基础依赖、PyTorch、视频处理 |
| `anonymization.txt` | 匿名化所需的关键点与图像依赖 |
| `ui.txt` | PySide6 原生桌面界面 |
| `distribution.txt` | Windows 桌面软件的冻结打包工具；接收软件的用户不用安装 |
| `distribution-cpu.txt` | CPU 轻量发行版的完整构建依赖，不携带 CUDA 运行库 |

新训练流程统一使用 `training.txt`；按根目录 README 的顺序准备三个指定数据集和训练模型。各文件中的 `-r` 引用按本目录相对路径解析。
