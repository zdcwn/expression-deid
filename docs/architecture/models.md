# 当前模型与调用关系

## 处理流程

图片、视频、摄像头 → 灰度化和定位对齐 → 统一人脸匿名化模块 → 匿名灰度脸 → 表情 CNN → 类别、置信度、因果时序状态。

`ui/opencv_demo.py::build_pipeline` 装配运行资源；`runtime/pipeline.py` 执行局部编辑、合成和表情推理，`runtime/video.py` 管理逐帧跟踪。`ui/desktop.py` 接收结果，不自行构造类别或概率。命令行 `infer` 也使用该完整运行流程。

匿名化和身份模型保留已确认版本；表情使用 FER-2013 为主、KMU 为辅重新适配的四组卷积 CNN，图片与视频复用同一权重。实际模型文件和摘要见 `configs/runtime_models.json`。

## 输入与评价

分类器只接收实际局部匿名化、贴回和重新对齐后的 100×100 灰度脸。显示处理不改变分类输入域。缓存记录源数组、标签、划分及匿名权重摘要，生成缓存时不调用表情预测。

`check_current_expression.py` 固定检查点后，分别评价 FER、KMU 和 RAF 的完整验证/测试划分，并抽查运行输入与缓存逐像素一致。最高验证结果为 KMU 六类 91.74%（230 张），其他分域结果及训练来源分别见 `docs/validation/expression-validation.json`、`expression-training.json`。初始化含历史多域学习，当前留出结果按本轮回归范围解释。

`required_evaluate.py` 与 `required_pipeline.py` 保留对齐残差组件的诊断用途；Pins 原脸 94.27%、残差匿名后 1.39% 专属于该组件，不替代完整运行流程的表情复测。

## 训练与发行

`training/required_training.py` 训练 Pins 身份与 CelebA-HQ 匿名化，`training/expression_current.py` 负责 FER 主训练及 KMU 辅助。表情候选保存在 `models/training/expression_fer2013/best.pt`。发行构建读取同一运行清单，不扫描候选目录；旧质量模型与重复表情权重已经清理。
