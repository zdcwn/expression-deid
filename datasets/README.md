# 数据目录

当前运行模型的实际训练配置见 `configs/runtime_models.json`：身份使用 Pins，匿名化残差使用 CelebA-HQ，表情以 FER-2013 为主、KMU 为辅进行匿名域适配，RAF 只用于本轮跨数据集评价。FER 使用原始七类 emotion 标签，训练抽样占 85%，KMU 占 15%。

| 路径 | 数据类型 | 谁使用 | 是否由程序生成 |
|---|---|---|---|
| `celebahq/raw/`、`celebahq/source.json` | 指定的 CelebA-HQ 256 图像和下载校验记录 | 匿名化训练 | 下载与校验生成 |
| `pins/raw/`、`pins/source.json` | 指定的 Pins 105 类图像和下载校验记录 | 自建身份模型训练与评价 | 下载与校验生成 |
| `fer2013/fer2013.csv`、`fer2013/source.json` | FER-2013 原始标签与官方 Usage 划分 | 主训练、验证及测试 | 原始 CSV 只读 |
| 各主数据集的 `prepared/` | 100×100 灰度数组、标签、划分和来源清单 | 三个统一训练命令 | `python -m expression_deid prepare` 生成 |
| `kmu-auxiliary/prepared/` | 按人员划分的 KMU 六类表情 | 仅在配置选中 KMU 时辅助表情训练 | 数据准备生成 |
| `ppb-emo/prepared/` | 按人员划分的中央红外中性片段抽帧 | 仅在配置选中 PPB-Emo 时辅助表情训练 | 数据准备生成 |
| `affectnet-kaggle/raw/`、`affectnet-kaggle/source.json` | 用户提供的 AffectNet 八类压缩包及校验记录 | 仅在配置选中 AffectNet 时辅助表情训练 | 用户提供、程序解压 |
| `affectnet-auxiliary/prepared/` | 排除 contempt 后的七类对齐灰度图、划分和匿名缓存 | 仅在配置选中 AffectNet 时辅助表情训练 | 数据准备生成 |
| `kmu-fed-aligned/` | 桌面内置的 1,100 张带标签示例与清单 | 已发布应用的图片演示 | 随项目保存 |

`configs/required_datasets.json` 约束指定数据的准备和诊断，当前已恢复表情配方另见 `configs/runtime_models.json`。`prepare all` 同时准备三个主集和当前选定的辅助集；准备可用辅助集不等于同时混入训练。训练与评价读取 `training/required_data.py` 和 `training/required_training.py`。

Pins 的同一身份可出现在不同划分，但近重复图片组不会跨划分。FER-2013 使用原版 emotion 标签；跨官方划分的相同像素样本从较低优先级划分中移除。辅助集按人员划分，避免相邻视频帧跨训练与验证。

AffectNet 压缩包的 `class001` 至 `class007` 映射到项目七类表情，`class008`（contempt）排除。包内训练集保留为辅助训练部分，包内验证集经过重复组检查后再划分为辅助 valid/test；与FER-2013的近重复图优先保留FER侧。

当前表情继续训练读取 `fer2013/prepared/anonymous-runtime.npy` 与 `kmu-auxiliary/prepared/anonymous-runtime.npy` 及同名 JSON。它们由完整局部匿名运行流程生成，包含贴回和重新对齐；残差组件单独生成的 `anonymous-<摘要>.npy` 仅用于组件诊断，不能替代这两个训练缓存。训练核对源图、标签、划分及匿名权重摘要。原始下载、匿名缓存和候选权重不自动提交，分享和使用遵守数据许可。
