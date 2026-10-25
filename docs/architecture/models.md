# 模型接入与文件职责

网络结构和检查点读写由 `expression_deid/models.py` 实现；权重按功能保存在 `models/`。逐个运行文件见[模型清单](../../models/README.md)，训练入口与产物位置见[工程目录](project-layout.md)。

## 运行调用

`ui/opencv_demo.py` 中的 `build_pipeline` 装配当前发布权重，`runtime/pipeline.py` 执行局部匿名化和匿名表情分析。检测使用 `preprocessing/detect.py`，关键点条件组件由 `runtime/anonymization.py` 加载。文件路径均从项目根目录解析。

质量视频表情由 `runtime/expression.py` 加载 `models/expression/expression_video_quality.json`，核对来源标记、类别顺序、预处理和两个表情权重的 SHA-256。匿名输入限定为 `100×100 uint8` 灰度脸。视频模块负责连续帧状态，UI 不直接拼造模型概率。

## 训练与部署

`training/train.py` 保留表情、匿名化和项目自建身份模型的训练入口。身份模型用于离线训练和评价，不在桌面实时识别链路中加载。

训练默认产物位于 `models/training/`，目录中的候选不参与运行时自动选择。质量候选经过 `scripts/export/export_video_quality.py` 导出后，以配套 JSON 进行验证，再明确部署到 `models/expression/`。关键点条件权重由 Git LFS 管理，第三方实现和许可放在 `third_party/`。
