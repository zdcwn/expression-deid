# 本地训练候选

- `train-identity` 输出 `identity_pins.pt`。
- `train-anonymizer` 输出 `anonymizer_celebahq.pt`。
- `train-expression` 输出 `expression_fer2013/best.pt` 与 `training.json`，FER-2013 为主、KMU 为唯一辅助。

此目录仅跟踪本说明，候选与日志保持本地。验证选中的权重不会自动替换 `models/expression/expression_current.pt`。身份、匿名化的训练记录为同名 `.history.json`；表情记录保存抽样比例、最佳轮、验证分数和校准。小样本工程检查不作为模型准确率。
