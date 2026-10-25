# 车载场景下隐私保护人脸表情识别项目说明

**用途：**课程或项目作业提交
**版本日期：**2026 年 10 月 11 日
**代码基线：**`main` 分支的 Python/PySide6 桌面运行版

## 一、项目目标

项目面向车载座舱摄像头视频，在不把原始人脸交给表情分类器的前提下，完成表情识别。系统需要同时做到：

1. 原始人脸不进入表情分类器；
2. 匿名化只修改检测到的人脸局部，尽量保留发型、头部轮廓、背景和座舱环境；
3. 表情结构尽量保持，输出表情、置信度和视频中的稳定结果。

项目不是用另一张完整人脸覆盖原图，而是对灰度人脸的局部区域做匿名编辑，并把编辑结果合成回原画面。

## 二、系统流程

```text
摄像头或视频
    ↓
灰度化
    ↓
YuNet 人脸检测与五点对齐
    ↓
统一人脸匿名化模块
    （局部残差、几何约束和关键点条件）
    ↓
重新对齐为 100×100 灰度脸
    ↓
表情识别模块
    （只接收匿名图，输出类别、置信度和不确定状态）
    ↓
视频显示与遥测记录
```

代码在 `ExpressionDeidentifier.process()` 和 `FastVideoRunner.process_frame()` 中固定了输入边界：表情模型调用使用匿名结果，遥测字段为 `expression_input_source=anonymized_only`。原图只用于生成桌面端左侧对照画面，不参与表情分类。

对外文档按一个统一的人脸匿名化模块和一个表情识别模块描述功能。模块内部的实现和权重选择由桌面端的高速、质量两种运行设置管理，报告中的指标按数据集和验证条件分别记录，不能拼接成一次端到端准确率。

## 三、使用的数据集

| 数据集 | 内容与规模 | 用途 | 限制 |
|---|---|---|---|
| RAF-DB | 15,339 张对齐人脸；训练 12,271 张、验证 3,068 张；七类表情 | 通用表情训练和验证 | 主要是普通彩色人脸，与车内近红外有域差异，类别不均衡 |
| KMU-FED | 真实车辆座舱近红外图像约 1,100 张，12 名受试者，六类表情 | 车内匿名化训练、回归和图片示例 | 人数少，同一人相邻帧相关，不能代表陌生驾驶员泛化 |
| PPB-Emo | 固定驾驶模拟器中央红外视频；项目使用 185 帧，按参与者划分为 114/34/35 | 补充车内中性样本，检查中性误判为悲伤或愤怒 | 是模拟器，不是真实道路；主要验证中性 |
| FER2013/FER+ | 原始像素和投票标签 | 补充高一致性表情样本和中性样本 | 仍是普通图像域，不能替代座舱数据 |

iCarB-Face、On-road DFE 等数据集没有写成当前已完成训练数据，当前指标不包含它们。原始数据的许可和分发限制仍以数据集提供方为准。

## 四、预处理和划分

- 所有输入先转为单通道灰度图。
- YuNet 检测人脸后，用眼睛、鼻尖和嘴角五点做相似变换对齐。
- KMU 和 PPB 保存为 256×256 对齐图，分类器输入再缩放到 100×100。
- 表情模型使用训练集统计得到的均值和标准差；验证时不做随机增强。
- 匿名结果合成回原灰度画面，再从合成结果重新裁取分类器输入；检测框外的像素保持不变。
- RAF 使用官方训练/验证划分；KMU 按人员和序列尾帧划分；PPB 按参与者划分，三部分人员不重叠。

匿名缓存记录源文件哈希、代码哈希、参数和 `anonymized_only` 标记。缓存契约不匹配时必须重新生成，避免把原图或旧匿名结果误当作当前输入。

## 五、模型和训练方法

### 1. 统一人脸匿名化模块

模块使用单通道残差式 U-Net，结合面部几何和关键点条件约束。输入和输出都是灰度人脸，运行时只在检测到的人脸区域合成，外部背景、头发和头部轮廓不参与替换。

训练目标包括像素重建、表情保持、身份间隔/余弦约束和总变分平滑。匿名化结果由实际运行链路重新对齐后送入表情识别模块，不能先识别原图表情再挑选替换图。

质量预览使用关键点条件的 GAN 组件，生成器只接收关键点条件图，原始脸部像素不直接进入生成器；缺少该组件时回退到局部编辑路径。主分支用 `models/anonymization/ganonymization_50.ckpt` 和 `models/detection/face_landmarker.task` 管理这一组件。

### 2. 表情识别模块

表情模块使用 `ExpressionNet` 轻量 CNN 完成实时推理，并提供一个固定输入契约的 FP32 质量权重用于离线或低帧率分析。两种运行设置都只读取匿名灰度脸，输出七类表情：惊讶、恐惧、厌恶、高兴、悲伤、愤怒和中性。

实时训练从匿名缓存开始，使用 RAF、KMU、FER+ 和 PPB 的混合样本，通过加权采样减小类别和域差异。训练时加入轻微亮度/平移增强及一致性损失；只有在 RAF、KMU、PPB 验证结果没有明显退步时才保留候选。

质量权重使用匿名缓存微调分类头和后段特征，并在同一匿名验证集上进行温度校准和左右翻转测试时增强。质量设置速度较低，适合准确率优先的离线视频。

典型训练命令：

```powershell
python -m pip install -r requirements/training.txt
python -m pip install -r requirements/anonymization.txt
python scripts/training/train_realtime_expression.py `
  --device cuda --epochs 8 --per-class 650 `
  --base models/expression/expression_machine_selector_texture.pt `
  --output models/training/expression_realtime_candidate.pt
```

训练权重先写入本地模型目录，完成输入边界、单元测试和数据集验证后再替换运行权重。

## 六、验证结果

所有表情指标均在匿名灰度图上得到，原始图没有进入表情分类器。主分支当前静态汇总保存在 `docs/model-metrics.json`。

| 运行设置和验证域 | 样本数 | 准确率 | 说明 |
|---|---:|---:|---|
| 实时设置 · RAF valid | 3,068 | 60.50% | 通用自然场景匿名图 |
| 实时设置 · KMU valid | 220 | 99.09% | 同训练人员尾帧，工程回归 |
| 质量设置 · RAF valid | 3,068 | 78.39% | 中性召回 81.91% |
| 质量设置 · KMU valid | 220 | 99.55% | 1 个错误 |
| 质量设置 · PPB valid | 34 | 100.00% | 仅中性 |
| 质量设置 · PPB test | 35 | 100.00% | 另一组参与者，仅中性 |

RAF 是相对严格的通用验证集；KMU 与 PPB 样本存在同人或相邻帧相关性，不能把高分直接外推到所有驾驶员和道路环境。RAF 中性样本的悲伤/愤怒误判为 58/680，仍需要真实座舱中性数据继续校准。

匿名化回归记录中，强几何匿名路径在 220 个 KMU 尾帧上的身份命中率为 16.82%，这里的身份指标只用于离线匿名化检查，不是桌面运行时的业务识别器。实际部署仍需用未参与训练的驾驶员重新评估。

RTX 3060 Laptop GPU 上，当前视频遥测记录的稳态分析帧率约为 29.13–33.99 FPS。质量设置的生成和 FP32 分类器会带来更高单帧耗时；界面刷新率不等于模型完成一帧的计算速度。

## 七、桌面端和验证入口

主入口是 Python/PySide6 桌面端，不包含网页服务，包含图片实验、实时摄像头/视频和模型表现三页：

```powershell
python -m expression_deid.desktop_ui --device cuda
```

安装和检查：

```powershell
python -m pip install -r requirements/training.txt
python -m pip install -r requirements/anonymization.txt
python -m pip install -r requirements/ui.txt
python -m compileall -q expression_deid scripts tests
python -m unittest discover -s tests -p "test_*.py" -q
python scripts/checks/check_desktop_presentation.py --device cuda
```

代码位置：

- `expression_deid/runtime/pipeline.py`：检测、对齐、局部匿名化和匿名输入边界；
- `expression_deid/runtime/video.py`：视频解码、跟踪、时序滤波、输出和遥测；
- `expression_deid/runtime/expression.py`：质量设置的输入契约和校准；
- `expression_deid/runtime/temporal.py`：概率平滑和“不确定”状态；
- `expression_deid/training/train.py`：匿名化和表情训练器；
- `scripts/training/train_realtime_expression.py`、`scripts/training/train_video_quality.py`：两种训练入口；
- `docs/architecture/models.md`：主分支模型和资产边界。

## 八、已知限制

1. 当前数据仍以普通人脸、固定座舱和模拟器为主，缺少真实道路白天、夜间、隧道、遮挡和振动条件下的完整七类表情视频。
2. KMU 回归集包含同人相邻尾帧，不能当作严格未见驾驶员测试。
3. 近红外低对比度画面可能出现局部纹理模糊或关键点失效，需要继续补充车载近红外数据。
4. 摄像头接口已提供采集和推理入口，陌生驾驶员的实车准确率仍需现场验收。

报告中的数字是带数据集范围和输入条件的工程验证结果，不代表所有摄像头、驾驶员和道路环境都能达到同样水平。
