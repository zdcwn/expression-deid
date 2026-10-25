# 本地训练候选

此目录保存训练生成的检查点、候选模型及相应训练记录，不参与桌面端的自动选择。

- `train-expression`：默认 `expression.pt`。
- `train-identity`：默认 `identity.pt`，用于离线身份训练和评价。
- `train-expression-kmu`：默认 `expression_kmu.pt`。
- `train-expression-joint`：默认 `expression_joint.pt`。
- `train-anonymizer`：默认 `anonymizer.pt`。
- `scripts/training/train_realtime_expression.py`：默认 `expression_realtime_candidate.pt`。
- 质量训练使用 `--output` 指定独立子目录，其中保存 `best.pt` 与训练记录。

本目录仅跟踪此说明文件。部署时明确选择经过验证的候选，不用修改桌面代码来自动扫描候选目录。
