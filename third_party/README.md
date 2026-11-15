# 第三方组件

- `ganonymization/lib/models/pix2pix.py`：匿名化生成组件的公开网络定义，MIT 许可保存在对应 `LICENSE`。
- PyTorch、Torchvision、OpenCV、ONNX Runtime、PyAV、MediaPipe 与 PySide6 用于实际运行；软件发行携带依赖的元数据和许可。
- 桌面字体回退使用 Noto Sans SC，字体及 OFL 许可位于 `expression_deid/ui/assets/`，随软件携带。

项目自己的网络、训练和界面代码位于 `expression_deid/`；报告模板来源与许可单列于 `reports/template-license/`。旧质量表情资源已从当前运行配置中移除。
