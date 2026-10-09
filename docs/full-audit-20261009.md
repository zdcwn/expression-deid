# 全链路检查、修复和模型实验（2026-10-08～09）

## 结论与使用边界

检查对象是 `D:\git-home\expression-deid` 的 `dev/gj` 分支。逐个检查了项目模块、命令入口、数据读取/划分、25 个 `.pt` 权重、4 个外部模型资产、匿名像素进入分类器的边界、Qt 线程、视频编码、摄像头采集与交接压缩包。详细原始证据保存在 `outputs/full_audit_20261008/`，精简结果随本报告一起提交。该目录名称按检查开始日期保留。

已经修复实际复现的问题，并部署通过表情回归的 epoch 7 快速视频分类器。52 项回归测试通过；5 条训练路径均实际执行了前向、反向、保存和重新加载。训练小样本执行检查不是模型准确率验收。

**默认匿名化的独立身份隐私仍未通过。** 当前快速匿名输入在 220 个 KMU 尾帧查询、1,680 人外部干扰库上的 SFace 原身份命中率为 **96.36%**，高于要求的 30%～40%。小型闭集身份分类器的 7.27% 不能替代这个结果。加强匿名化可把 SFace 命中率降到 15.91%～16.82%，但连续表情测试丢失悲伤，因此这些方案保留为实验，没有作为默认配置。

**视觉自然程度也未验收通过。** 人眼展示沿用 GANonymization 路线；实际输出抽帧仍能看到眉眼变平、纹理模糊和局部过渡痕迹，不能声称达到了用户参考图的自然程度。PPB-P24 的 900 帧中，130 帧因密集关键点失效而遮蔽显示面部。不能把这些帧算作正常生成成功。

**表情改进不等于个人照片问题已经解决。** 匿名 RAF 全验证集从 60.50% 提高到 68.42%，KMU 尾帧从 99.09% 提高到 100%，PPB 中性验证帧从 91.18% 提高到 100%。但用户提供的第一个中性截图裁片仍被新模型判为厌恶，第二个裁片只有低置信度的中性预测。裁片不是原始照片，不能据此宣布本人摄像头识别已修好。

网页源码和 `frontend/` 未修改。桌面只处理错误状态、资源释放、数据交付和指标口径，不重做视觉布局。

## 固定链路与验收口径

快速视频：输入帧 → 灰度 → 可信人脸检测/对齐 → 局部素材编辑和匿名网络残差 → 合成到原灰度画面 → 再对齐实际匿名结果 → 单个表情分类器 → 因果时序决策。

表情分类器不得先识别源图，也不得用源图表情类别挑选另一张图片。匿名素材选择使用局部结构距离，不使用表情模型或预测标签。身份诊断可以读取源脸用于隐私比较，不能混入业务表情输入。

实时展示 GAN 每 15 帧刷新，分类器仍然每帧分析实际机器匿名结果。工程遥测分别记录用途，产品画面只标注“匿名视频”，不把展示帧错误标成分类器的实际输入。检测、轨迹或匿名化资源失败时撤销表情状态；无法可靠匿名化时不把源脸送入分类器。显示密集关键点临时失败，最多短时复用 0.5 秒内的匿名结果，之后遮蔽面部。

时序置信度是匿名帧概率的因果平均，不是投票比例，不会人为加分。证据不足显示“暂不确定”，置信度为空。`raw_*` 是匿名图的单帧输出，**不是原图结果**。默认推理和评估只加载指定分类器；旧 RAF/joint 辅助模型需要显式 `--expression-ensemble`。

`analysis_fps` 的分子是完成匿名化和分类的成功帧，分母包含所有帧的模型处理耗时，失败帧也计入耗时。另有尝试处理速度和端到端编码速度，不混为同一指标。模型加载、首帧、刷新峰值和稳态分别报告。

## 已复现并修复的问题

| 编号 | 文件/路径 | 原问题 | 修改和验证 |
|---|---|---|---|
| F01 | `models.py: tensor_from_input` | 缺少 numpy 导入，数组接口抛 NameError | 修正导入；实际构造 1×1×100×100 张量，错误形状拒绝 |
| F02 | `models.py: load_checkpoint` | 不受限 pickle 加载、无有限性校验 | `weights_only=True`，仅兼容必要的旧 TorchVersion；类型、state_dict、有限权重检查；损坏/NaN 检查点拒绝 |
| F03 | `normalize.py` | 非有限归一化参数可传播；增强随机数不受训练种子控制 | 均值/方差范围校验；使用可控 numpy 随机状态；重复种子回归 |
| F04 | `identity_gallery.py`、`build_lfw_gallery.py` | 图库清单保存绝对路径，换机器可能读旧盘 | 新清单相对路径；旧绝对路径按当前图库根目录迁移，禁止越界；重复身份、零向量和非有限向量拒绝 |
| F05 | `evaluate.py` | 缺失真实标签时统计异常；结果缺少当前模型来源 | 无标签只做可做的诊断；记录路径、SHA、设备、参数、辅助模型数量和诊断错误；不虚构原图表情一致率 |
| F06 | `cli.py`、`evaluate.py` | 指定一个检查点仍可能暗中混入旧 RAF/joint 模型 | 默认指定单模型，ensemble 显式开启；实际 CPU infer/evaluate 和输出字段检查 |
| F07 | `inference.py: _prepare_gray` | `aligned=False` 的 100/256 方图仍可能跳过检测 | 显式未对齐输入强制检测；真实模型加可控检测器回归验证调用次数 |
| F08 | `inference.py: anonymize/process` | 对齐图输出尺寸、合成和分类器输入口径不一致 | 保留 100/256 原尺寸，先局部合成再对齐；捕获分类器实参，逐像素核对 |
| F09 | `inference.py` | 素材缺失可能退回源图；诊断操作可能干扰素材状态 | 缺素材明确失败；诊断开/关匿名像素一致；每帧分类只调用一次 |
| F10 | `inference.py: donor bank` | 素材库包含 220 张 KMU 验证图 | 使用共享尾帧划分，只纳入 880 张训练 KMU 和 300 张 RAF 训练中性；集合交集断言为零 |
| F11 | `inference.py: donor search` | numpy 搜索造成首帧/轨迹重建迟滞 | CUDA FP32 向量距离，缓存素材向量；64 个输入 CPU 2.190 秒、GPU 0.099 秒，选中素材完全一致 |
| F12 | `inference.py` | uint8/0～1 素材路径转换尺度不统一 | 统一舍入与范围；实际对比两种输入表示，素材一致 |
| F13 | `video_demo.py: FastVideoRunner` | 倒退时间、来源重启、失脸后仍可沿用旧状态 | 时间、分辨率、帧号和大幅位置变化重置；失脸/匿名失败不分类，隐私失败遮蔽；回归核对调用和像素 |
| F14 | `video_demo.py: display cache` | 刷新失败的旧脸可能无限保留 | 成功刷新计数，源时间过期后遮蔽；缓存失败/恢复/过期逐个测试 |
| F15 | `video_demo.py`、`app.py`、Qt | 处理尝试或界面刷新可被算作分析成功 | 成功帧/所有耗时统计；10 帧无脸视频实际 analysis_fps=0 |
| F16 | `video_demo.py`、`app.py`、`evaluate.py`、CLI | 异常或退出路径遗漏释放模型/容器 | try/finally，编码 flush、容器关闭、MP 关闭；损坏视频、错误输出路径、预热失败、无视频流验证 |
| F17 | Qt `VideoPage`/`VideoWorker` | 停止和快速切来源可能丢掉运行线程或消费旧帧 | 非阻塞停止，保留线程至 finished，合并待切来源，丢弃旧信号；实际加载时关闭窗口通过 |
| F18 | Qt `VideoWorker` | UI 帧事件可能无限堆积 | 最多一帧待显示，UI ACK；模型仍分析每帧。真实 worker 连续交付 81 帧、停止并释放通过 |
| F19 | Qt `ExperimentPage`/`PerformancePage` | 实验失败按钮/诊断缺失状态不可靠；旧报告冒充当前 | 失败重新启用操作，缺失指标为“未验证/—”；历史结果明确标注；实际六类图片与三页构建验证 |
| F20 | `feature_geometry.py` | 强度 2 时局部缩放可能变负，翻转几何 | 最小尺度限制 0.20，有限关键点校验；此修正改变旧强匿名输入域，旧 34.45% 报告失去当前配置适用性 |
| F21 | `align.py` | 退化或非有限点仍求相似变换 | 双侧有限、等数量、非退化检查，坏点立即拒绝 |
| F22 | `ganonymization_backend.py` | 暗近红外丢密集点后可继续使用旧点/空条件 | 清空旧状态，几何检测可尝试低对比增强；五点锚拟合阈值；无法定位明确失败，G 不接受原图像素 |
| F23 | `prepare.py` | 失败 partial 清单不能重试；改源图后可能复用旧结果 | 失败可重试，成功项记录源字节数/mtime；真实坏图、两张 KMU 与 resume 路径执行 |
| F24 | `prepare_ppb.py` | 小数帧率/缺失 duration 的抽帧位置错误；无视频流不释放 | 正确 duration/time_base/fps，解码计数回退、0 起始目标帧；真实 PPB 抽帧和无流关闭回归 |
| F25 | `train.py: identity loss` | 身份对抗裁剪与验收不一致，margin 在批平均后截断 | 可微脸部椭圆裁剪匹配 PIL，面外梯度为零；逐样本 hinge 再平均，不能由容易样本抵消困难样本 |
| F26 | `train.py: anonymizer` | 默认调用源图表情教师；用全部 1100 图更新、没有验证选择 | 默认仅匿名输出+GT 标签，关闭教师时拒绝无效的旧蒸馏权重；880 训练/220 尾帧验证，按验证目标保存；保存验证范围说明。小样本 18/6 实际执行，不宣称旧权重已自动修正 |
| F27 | `train.py: joint` | PPB 验证不参与选择/温度；旧 GAN 图跨源划分 | PPB 同时参与域验证、选模、温度校准；按源图排除 GAN 跨划分；五条训练路径完整执行 |
| F28 | `train.py` | 报告匿名损失最后一批值，难以解释 | 按全轮样本数加权统计各损失；不把训练日志里的 surrogate 当独立隐私结果 |
| F29 | `data.py: FER+` | 未知/NF 多票或低一致性仍可能被当成七类标签 | 全部投票含未知项，要求严格多数、至少 5 票，平票拒绝；真实 CSV 行顺序/Usage/像素格式检查 |
| F30 | `data.py: FER+` | train/PublicTest/PrivateTest 存在同像素图；增强 CSV 类别顺序错 | 按像素排除训练侧重复，发现 train/valid 204 个重复键；增强标签 FER→RAF 映射纠正，修复后 train/valid 交集为零 |
| F31 | `prepare_gan_expression.py` | 缓存未经实际合成再对齐；privacy-selected 标志无有效诊断 | 生成真实 runtime 输入，训练/验证按源尾帧；筛选必须有诊断且实际执行；两张真实 GAN 样本成功 |
| F32 | `train_realtime_expression.py` | 旧缓存可以混用不同代码/素材/划分；候选可能覆盖稳定权重 | 缓存 schema、数据/代码/参数/SHA 校验；分域不退步门槛、pending 原子保存、匿名验证温度校准；追加六轮未改善候选被拒绝 |
| F33 | `create_handoff_bundle.py`、文档 | 换机无法确认权重版本；分支指令仍写 gj-dev；旧指标当当前 | 文件 SHA 写入清单，真实分支 dev/gj，报告分开历史/当前；小路径 ZIP 加入本审计 |
| F34 | `requirements.txt` | 仅靠 MediaPipe 间接引入 cv2，单装基础/桌面依赖不完整 | 明确声明本机已验证 opencv-contrib-python 5.0.0.93；pip check 无冲突；未声称完整空白机器安装已实测 |
| F35 | 视频叠加标题 | 展示帧被错误标作 EXPRESSION INPUT | 改为 ANONYMIZED VIDEO；不在产品里暴露实现分支，也不作错误输入声明 |

## 逐模块与入口检查矩阵

以下是覆盖范围，不代表穷举所有输入组合。未执行的环境路径明确标为静态检查或待验收，不用 compileall 冒充效果测试。

| 模块 | 逐点检查/实际执行 | 结果与界限 |
|---|---|---|
| `__init__.py` | 包导入、版本导出 | 通过；无计算分支 |
| `__main__.py` | 11 个子命令 help 与分发 | 通过；训练执行另见训练矩阵 |
| `config.py` | ROOT、各数据根、模型根、100/256 模板与阈值 | 当前资源均存在；YuNet 是 CPU，不能称全链路 GPU |
| `labels.py` | 7 类 RAF 次序、6 类 KMU 映射、坏名字/非法编号 | 回归通过；KMU 没有中性，因此 GUI 数据示例只有六类 |
| `detect.py` | RGB/BGR、640 letterbox、三个 stride 解码、NMS、空脸 | 6 张真实 KMU/RAF 与 OpenCV 原生 YuNet 同像素比较，最大坐标差约 0.000107 像素；真实 6300 帧检测另测 |
| `align.py` | 面部选择、五点相似变换、正/逆合成、退化点、fit_error | 实际 KMU/PPB 对齐和像素边界回归通过；检测大角度可靠性仍需目标相机 |
| `normalize.py` | L 灰度、BILINEAR 尺寸、RAF 统计、增强/种子、异常参数 | 全 RAF 训练统计复算一致；正常推理无随机增强 |
| `data.py` | RAF/KMU/PPB/FER+/匿名缓存适配、标签、人员/尾帧划分 | 全清单图解码、FER 35887 行配对；泄漏修复；KMU 同人邻帧仍重叠 |
| `prepare.py` | 对齐、质量标签、失败清单、partial/resume、源文件变化 | 两张真图、坏源图、重试实测；未擅自重写全数据集 |
| `prepare_ppb.py` | PyAV 开流/抽帧/对齐/清单、帧率和缺时长回退 | 真视频抽帧成功；无视频流正确关闭；无 metadata 分支为可控容器回归 |
| `models.py` | 三架构、7 类、身份类别、restricted load/save、dtype/finite | 全 25 权重重加载、CUDA FP32/FP16 实际前向，逐权重表在后文 |
| `train.py` | 五条训练入口、混合采样、匿名/身份损失、校准、保存/选择 | 五条各一轮匿名夹具前后向和 reload，通过；模型效果靠独立缓存/视频报告，不靠这次夹具准确率 |
| `feature_geometry.py` | 478 点局部变形、强度边界、源几何迁移 | 修正负缩放；强匿名视频与 GAN 实际路径复查；几何相似不等于身份去除或表情准确 |
| `ganonymization_backend.py` | AST 提取生成器、权重、MP 检测/重试、线图、失点、关闭 | 六类实际生成和视频通过运行检查；暗 NIR 仍有失点/遮蔽，图像自然程度未通过 |
| `identity_gallery.py` | 脸部裁剪、112 输入、embedding、1680 库、softmax、迁移 | 实际 220 对查询；相对图库 softmax 不是经过校准的“身份概率” |
| `inference.py` | 自动/强制对齐、素材/残差/掩膜、诊断、单模型/ensemble、资源失败 | 8532 张输入与真实视频分类器实参全量像素一致，最大差 0；身份命中率仍不合格 |
| `temporal.py` | 初始化、持续证据、孤立抖动、真实切换、撤销、时间断点、类别数异常 | 回归通过；6300 中性帧最终标签 0 次跳变，不能用此替代其他表情准确率 |
| `video_demo.py` | 每帧分类、周期检测/展示、轨迹、故障、PyAV 导出、摄像头 | 24 帧编码解码/遥测一一对应；无脸 10 帧成功分析 FPS=0；坏路径拒绝 |
| `app.py` | image/examples/video/camera 的共享 builder、fast/quality、设备、清理 | builder、视频和错误路径执行，OpenCV 交互按键未由无人值守检查模拟全部操作 |
| `desktop_ui.py` | 原生 Qt 主入口 | help 与实际 MainWindow 构建通过，不启动网页 |
| `desktop_ui_qt.py` | 三页、实验结果、真实 VideoWorker、UI ACK、切来源/停止/关闭、指标缺失 | offscreen 三页构建、六类实验、81 帧实际信号、加载中关闭通过；不等于人工审美验收 |
| `cli.py` | help、infer/evaluate/video/camera 分发，显式丢失权重、资源错误 | 18 个 CLI/module help；CPU infer/evaluate 真模型；视频实际执行，摄像头采集仅无脸 30 帧 |
| `evaluate.py` | 标签 None、分类矩阵、ID/SFace/gallery、报告 provenance、finally | CPU 2 张实际验收入口+220 独立隐私回归；旧报告只读兼容，不虚构源图表情 |
| `web.py` | 服务构建、GET 状态/样例/报告、路径保护、畸形请求/CORS | 只读 HTTP 与无效请求验证；网页源码未改，本次不重新训练/优化网页界面 |

| 脚本/其他路径 | 执行与检查内容 | 范围 |
|---|---|---|
| `scripts/build_lfw_gallery.py` | 1680 人清单实图检查；路径输出改相对；参数正数检查 | 现有图库有效，未为同一批图重复下载重建 |
| `scripts/download_anonymization_assets.py` | URL、下载临时文件/大小校验、现有资产 SHA 与 ledger 核对 | 已有 3 项 ledger 完全相符；本次未重新下载 686 MB 同权重，不宣称所有断网/下载错误已实测 |
| `scripts/create_handoff_bundle.py` | 当前目录盘点、文件 SHA、JSON/CSV/ZIP 输出、ZIP 自检 | 生成便携交接清单；大权重仍需要单独复制 |
| `scripts/prepare_gan_expression.py` | 原真实 KMU 的 2 张 GAN 条件/筛选/合成/保存/清单 | classification source=anonymized_only，composited_and_realigned=true |
| `scripts/train_realtime_expression.py` | 缓存准备、3 域采样、8 轮一致性微调、原子保存、温度、失败门槛 | 多候选实际训练；不通过者不覆盖 runtime，资源 finally 关闭 |
| `scripts/tune_machine_privacy.py` | 训练侧几何/纹理参数扫描与冻结分类器/SFace | 只作候选搜索，不能拿训练侧分数作为验收 |
| `scripts/audit_project.py` | AST 逐文件清单、全图读取、25 权重质量/FP16、资产 SHA | 当前全权重报告已生成；修正 relative cache 参数写报告错误和失败 exit code |
| `scripts/audit_entrypoints.py` | help、FER 配对、真对齐、CPU inference/eval、错输入、只读 HTTP、Qt、pip | 所列 8 组检查全部通过；其中实际业务处理与 help 检查分别记录 |
| `scripts/audit_runtime_privacy.py` | 当前实际合成再对齐的 KMU 验证输入，闭集 ID 和外部 SFace | 默认/ratio/motion 三组各 220 查询，全 1681 库；保留 scope |
| `scripts/check_anonymous_cache.py` | 可控 canonical detector + 真 FastVideoRunner 捕获分类器像素 | 全 8532 张，差 0；它验证像素一致，不验证真实 detector 泛化 |
| `scripts/check_training_paths.py` | 五种训练器实际一轮、有限梯度/权重、输出/reload | 只用匿名夹具；匿名训练 18 train/6 valid，原权重不被覆盖 |
| `scripts/check_video_stability.py` | 真视频逐帧原始匿名概率、决策、轨迹、成功 FPS、p95、显示失败 | 7×900 PPB 帧 + 4 例视频；保存全部逐帧结果，finally 关闭模型 |
| `scripts/check_desktop_runtime.py` | 真 Qt 模型 worker、线程停止/窗口关闭、PyAV、坏视频、本机 camera | 全 7 组通过；摄像头无脸，未宣称其准确率通过；修正检查器误访问已销毁 Qt wrapper |
| `scripts/check_detector_reference.py` | 手写 YuNet 解码 vs 原生 OpenCV | 同模型、同 padded 像素、6 张实图，两者吻合 |
| `scripts/audit_gan_runtime.py` | 当前 Qt 图片/GAN 路径，六类均衡 24 张，再扩到全部 220 张尾帧 | 当前路径 24/220 张回归，不能套用到 fast 视频或历史 1100 图 |
| `tests/test_models.py` | 架构、保存加载、匿名损失等原有模型检查 | 纳入 52 项总回归 |
| `tests/test_preprocess.py` | 预处理、对齐和业务边界原有回归 | 纳入 52 项总回归；部分为源码约束检查，非整车验收 |
| `tests/test_temporal.py` | 因果滤波、轨迹/时间、每帧像素、失点缓存/遮蔽 | 控制夹具验证所有列出的故障路径；实际视频另测 |
| `tests/test_audit_regressions.py` | F01～F34 中可确定复现的回归，含真实 CPU 模型边界 | 当前数据/模型依赖与临时资源夹具分别记录，不把所有 tests 称作端到端 |
| `requirements*.txt` | 本机版本、pip check、GPU 识别、GUI/视频依赖 | 本机通过；干净新机器安装、不同 GPU/驱动未实测 |
| `frontend/` | 只读静态文件和接口行为 | 未修改；共享后端修复会影响共同调用的预处理 |
| `handoff/`、README、models README | 分支、当前/历史指标、资源路径、哈希、操作说明 | 修正旧分支和口径；压缩包逐文件检查见后文 |

## 数据逐路径检查

| 数据路径 | 数量与检查 | 划分/风险 |
|---|---|---|
| `datasets/kmu-fed/` | 1100 源图，原图未覆盖；实际对齐和 source stat 检查 | 12 个身份、六种表情。不能补中性 |
| `datasets/kmu-fed-aligned/` | 清单 1100，1100 图全部能解码、标签合法、无缺失/越界输出 | 880 train/220 valid 为同人同序列尾帧，邻帧会跨边界，绝不是 person-held-out |
| `datasets/raf-db/` | 12271 train/3068 valid，全部读图；跨 train/valid 文件 SHA 重复 0 | 验证类数 `[329,74,160,1185,478,162,680]`；少数类明显不平衡 |
| `datasets/fer2013-hf/` | 35887 原像素/官方 FER+ 投票行 Usage 对应；2304 像素长度/范围检查 | 严格多数和重复过滤后，train/valid 像素交集 0；增强 valid 405 图类别顺序修正 |
| `datasets/ppb-emo/raw-cir-neutral/` | 已有中央 IR 中性视频；真抽帧、七名 test 人员整段视频执行 | 模拟器中性场景，不是七类自然表情道路数据 |
| `datasets/ppb-emo/neutral-aligned/` | 清单 185，全部 decode/标签检查；只取 status=ok | 当前采用 114 train/34 valid/35 test 帧；另 2 张 fit 项拒绝；同一人的几帧高度相关，不能当独立统计样本 |
| `datasets/lfw-gallery/` | 1680 清单/实图/身份/embedding 合法，旧路径迁移核对 | 每次原始脸为第 1681 个查询身份；与 closed ID 类数不是一回事 |
| `datasets/kmu-gan-aligned/` | 1100 清单/实图无缺失、坏图、路径越界 | 历史生成条件；新几何/掩膜改变后不得视为当前视频域 |
| `datasets/kmu-gan-strength100/` | 1100 清单/实图全部检查 | 历史强度实验，保留复现，不作当前默认效果 |
| `datasets/kmu-machine-selected/` | 1100 清单/实图全部检查 | 历史筛选版本；源尾帧跨训练的旧记录在新 joint loader 中过滤 |
| `outputs/full_audit_20261008/anonymous_realtime_v3_cache.npz` | 8532 个匿名 100×100 uint8 输入；来源/标签/域/划分/hash/signature | 当前素材库隔离后重建；全量逐像素与真实 runtime 输入匹配 |

RAF 均值 0.481619、标准差 0.242606 与全训练图复算一致。归一化统计只来自 RAF train，不用验证图重算。

FER+ 过滤后训练七类计数（RAF 次序）：`[2507,323,42,7013,2330,1582,7432]`；验证：`[378,40,13,847,279,234,1063]`。厌恶和恐惧样本仍非常少，不能简单靠多轮训练解决。

PPB 人员：train `08,09,10,11,12,14,16,17,18,19,20,21,23,25,27,29,30,31,32,33,35,36,37`；valid `02,03,15,26,28,34,40`；test `04,05,07,22,24,38,39`，彼此无重叠。继承的早期权重存在旧素材库和旧匿名训练使用全部 KMU 的历史，修复 loader 不会消除旧权重的记忆，因此本报告 KMU 指标只能称为回归，不能称为严格未见身份泛化。

## 模型优化实验与默认部署

当前部署：`models/expression_realtime_stable.pt`，epoch 7，温度 0.953874886，3,495,171 字节，SHA-256：

```text
5a98c02d10439dd215cc4b6f59abf7bbbf174fd2df0396841b0551fd9c1a26ab
```

另外全量核对了 3357 个 valid/test 输入的运行精度：FP32 与实际 FP16 的类别一致率 99.732%，概率最大差 0.01072；GPU FP16 的匿名 RAF 准确率 **68.25%**（FP32 为 68.42%），KMU/PPB 均保持 100%。下文静态模型表采用 FP32，视频/Qt 帧率和决策采用实际 FP16。不能将两种精度结果无区别地混用。

旧 epoch 3 权重保留在 `outputs/full_audit_20261008/expression_realtime_stable_before_audit.pt`，SHA `4db0902e9f39234c9b6646b66357dc27f694b20e74c38bbd9f7d3bce163d1539`。新权重不进 Git，需要单独交接。

使用每类最多 650 个 RAF train 样本、完整 RAF valid、KMU 尾帧、按人员划分的 PPB。缓存 8532 张，训练 5175、验证 3322、test 35。匿名图上做轻微亮度/平移的两个视图，标签监督 + 0.3 一致性损失，8 轮中选 epoch 7。验证选择和温度校准使用同一 valid 集，test PPB 仅中性，必须保留这个限制。

选模规则要求三个域分别不比同输入基线退步超过 1 个百分点，且平均分提升。另在素材隔离后的当前输入域重新验证。追加 6 轮候选未超过冻结模型门槛，拒绝保存/部署。更大的 MobileNetV3 尝试 25 轮，匿名 RAF 最好 59.42%，低于对应紧凑模型 62.22%，因此删除公开架构/训练依赖入口，失败实验只留在 ignored 证据目录，没有堆叠更复杂网络。

| 当前相同匿名输入 | 旧 epoch 3 | 新 epoch 7 | 样本/范围 |
|---|---:|---:|---|
| RAF valid 准确率 | 60.50% | 68.42% | 3068；原始 RAF 图片经过当前匿名链路 |
| KMU valid 准确率 | 99.09% | 100% | 220 尾帧；非未见身份 |
| PPB valid 中性准确率 | 91.18% | 100% | 34 帧，7 人 |
| PPB test 中性准确率 | 100% | 100% | 35 帧，另 7 人；不能代表七类 |
| RAF NLL | 1.1188 | 0.9100 | 降低，较好 |
| RAF ECE，10 bins | 0.01302 | 0.01830 | 小幅变差，不能只报告准确率 |
| RAF 平均最高概率 | 0.6026 | 0.6950 | 必须结合真实准确率和分布解释 |

身份实验：

| 匿名输入/候选 | 独立 SFace 原身份命中 | 闭集 ID 正确 | 表情/采用决定 |
|---|---:|---:|---|
| 当前 fast 默认 | 96.36% | 7.27% | 表情最佳，但隐私不合格；保持为可运行基线，明确标记未过隐私 |
| ratio_v2 强几何 | 16.82% | 9.09% | 视频丢失悲伤，拒绝成为默认 |
| motion 候选，保留嘴部尺度 | 15.91% | 15.91% | 静态 KMU/PPB 正确，视频仍未确认悲伤，拒绝成为默认 |

另外，当前桌面实验图片的 **GAN 机器输入路径**也全量检查了同一批 220 张尾帧：表情准确率 **86.36%**，机器输入 SFace 命中率 **35.00%**，展示输出 SFace 命中率 **96.82%**。机器输入的单项隐私指标达到 30%～40% 区间，但这是更慢的 GAN 输入，不能把它的 35% 写成快速视频结果，也不能把展示输出当成该分类器的输入。先测 24 张得到 16.67%，扩到 220 张变为 35%，说明小样本挑选会明显影响结论。最终以 220 张为准。

三行均为当前素材库下 220 个 KMU 尾帧、每个查询加入 1680 外部身份。图库命中不是标准 FAR/TAR 阈值评估，不是对所有攻击模型的安全保证。没有用分类器先识别原始表情来筛选结果。`configs/` 中 stronger profile 仅供实验，默认权重的 `machine_privacy` 仍为空配置；不要复制失败候选覆盖 stable。

## 视频、桌面与速度

七段 PPB 中性共 6300 帧，6290 帧可信检测，6261 帧形成最终中性决策；其余为暂不确定或无脸。已决策帧都是中性，**最终类别跳变 0 次**。仍保留单帧 raw 的错误和覆盖率，不能把不确定帧直接算正确。

六类切换示例 120 帧：118 帧形成决策、5 次类别切换，六种类别都出现；每段末尾检查与标签相符。mixed 96 帧：95 检测、91 决策，3 次最终切换；单独怒/惊讶例各 24 帧、22 帧决策。它们由已有样例生成，是工程回归，不能用于宣称真实自然动作准确率。

RTX 3060 Laptop 上实际双分支模型分析稳态：PPB 各段约 29.54～33.99 FPS，示例约 29.13～33.88 FPS；真实 Qt worker 最后一组 24 帧模型处理均速约 29.67 FPS。均包含匿名化和表情运算，图像展示 GAN 按 15 帧间隔刷新。不是单个分类网络的 FPS，也不是 UI 刷新 FPS。

但**每帧稳定 24 FPS 没通过**：GAN 刷新导致 p95 单帧约 115～154 ms；冷启动六类短片总分析均速仅 21.81 FPS。24 帧完整 PyAV 导出例：模型分析 24.91 FPS，含解码/画字/编码的端到端 13.62 FPS。模型加载和 GUI 冷启动另有多秒耗时。公开 GAN 代码快照为 `7dbe45e3b172670ff59be982c662749165ab5ced`，交接包保留 LICENSE/CITATION。不能声称所有画面每 41.7 ms 处理完成。

最终使用当前代码另导出 24 帧（`desktop_runtime/encoded_current.mp4`），分析 11.64 FPS、含编码端到端 7.30 FPS，24 帧均能重新解码。该短片结果包含 GAN 初次推理/刷新开销，与长视频稳态不同；说明短片与启动阶段速度仍不稳定，不能只引用较快的一次导出。当前可视化标题已改为 `ANONYMIZED VIDEO`，避免把展示帧误标为分类器输入。

真实 Qt 检查：三页构建/切换/关闭；VideoWorker 用真实模型交付 81 帧 QImage；ExperimentWorker 六类样例各自预测正确并提供 1681 库诊断；模型加载时请求关闭保留线程，finished 后退出。检查器采用 Qt offscreen，因此不是重新评审界面美观。检查器首次曾误访问已经 deleteLater 的 C++ 对象，修正为完成信号记录，再跑全部检查通过；这是检查器错误，不能算作应用未释放线程。

本机 camera 0 实际打开并采集 30 帧，成功检测人脸 0 帧。没有保存这些原始相机帧，未进行个人表情标注。因此摄像头捕获接口通过，但本人脸部识别准确率、动态光照和真实道路应用**未验证**。

人眼展示抽查 `outputs/full_audit_20261008/visual_output_check.png` 中三帧：外部背景/轮廓保留，脸部改变可见，但纹理和眉眼过渡仍不自然。失败样本 `dense_fail_samples.png` 与 P24 遮蔽帧证明近红外密集关键点还有缺口。没有把重新采样截图当作真实照片，也没有用图像生成工具美化验收结果。

## 尚需优先解决的实际问题

1. **独立身份隐私与自然表情不能同时达标。** 需要重新训练经过当前 face-only 对抗损失、真实合成/对齐输入口径的匿名模型，再做独立验证器和表情变化联合验收。仅把 LFW 库加大或只追求 closed ID 小分数不够。
2. **用户本人中性图仍有错误。** 当前截图裁片测得旧模型厌恶 43.6%/悲伤 37.2%；新模型厌恶 53.3%/中性 42.8%。第二张改变不等于两个都修好。需要原始 RGB/近红外视频、稳定中性与真实过渡的人工标签，尤其眼镜、弱光和低幅度表情；截图缩放影响预测，不能作训练标注来源。
3. **PPB 当前只补中性。** 七个人中性全对不支持悲伤/愤怒分辨已解决。需要自然近红外七类/连续动作数据，按人员和视频隔离，包含眼镜、偏转、遮挡和行驶光照。iCarB-Face 若只补中性，也不能单独解决这两个类别或整车测试。本次未虚构新增数据下载成果。
4. **生成器显示质量与失点率。** 公开 GAN 权重的 RGB 训练域与座舱灰度不匹配；P24 仍有 150 帧关键点失败、130 帧遮蔽。不能通过偷偷展示旧帧、源脸或预选表情图片掩盖失败。
5. **逐帧 24 FPS 最坏时延。** 稳态平均够用，生成刷新、初始化峰值仍不够。下一步应单独测量并异步处理展示生成、对齐和 GPU 同步，而不是先把界面刷新数字调高。
6. **统计口径限制。** KMU 尾帧重叠人物和邻帧，继承权重有旧数据记忆；PPB test 仅 7 人/35 帧；图库没有标准 verification ROC。需要从新划分重新训练后报告人员级置信区间、各类召回/精确率、误判和不确定覆盖率。
7. **本地处理不等于库完全离线。** 实测 MediaPipe 日志出现 `portable_clearcut_uploader` 请求失败。官方维护者在 [MediaPipe issue 6291](https://github.com/google-ai-edge/mediapipe/issues/6291) 说明发布 SDK 带使用遥测，称不发送输入数据。本次没有抓包核实载荷，也未修改共享 Python 防火墙；正式离线车端需单独审查网络访问或源码构建。不能声称本次已验证零联网。

## 复现与回退

```powershell
cd D:\git-home\expression-deid
git switch dev/gj
python -m pip install -r requirements-train.txt -r requirements-anonymization.txt -r requirements-ui.txt
python -m pip check
python -m compileall -q expression_deid scripts tests
python -m unittest discover -s tests -q
python scripts/audit_entrypoints.py --output outputs/audit_entrypoints.json
python scripts/train_realtime_expression.py --prepare-only --device cuda --base models/expression_realtime_stable.pt --cache outputs/realtime_training/anonymous_cache.npz --per-class 650 --validation-per-class 0
python scripts/check_training_paths.py --cache outputs/realtime_training/anonymous_cache.npz
python scripts/check_detector_reference.py
python -m expression_deid.desktop_ui --device cuda
```

训练候选与缓存检查，不覆盖默认：

```powershell
python scripts/train_realtime_expression.py --device cuda --base models/expression_realtime_stable.pt --output outputs/retrain/candidate.pt --cache outputs/retrain/anonymous_cache.npz --per-class 650 --validation-per-class 0 --epochs 8
python scripts/check_anonymous_cache.py --cache outputs/retrain/anonymous_cache.npz --checkpoint outputs/retrain/candidate.pt --output outputs/retrain/input_check.json
python scripts/audit_project.py --section models --device cuda --cache outputs/full_audit_20261008/anonymous_realtime_v3_cache.npz --output outputs/model_audit
python scripts/audit_runtime_privacy.py --cache outputs/full_audit_20261008/anonymous_realtime_v3_cache.npz --output outputs/privacy_check.json
python scripts/check_video_stability.py outputs/example_kmu_expressions.mp4 outputs/example_mixed.mp4 --human-display --frames 900 --device cuda --output outputs/video_check
python scripts/check_desktop_runtime.py --device cuda
python scripts/audit_gan_runtime.py
```

原始输入不得为表情教师；准备缓存脚本只做匿名化和保存真实标签。更改素材/源图/参数/代码后，使用 `--rebuild-cache`，不要只改 manifest 声称缓存还是同一输入域。选模若失败退出，runtime 不变，不应直接复制最后一轮权重。

回退本次表情权重：停止桌面程序，将 `outputs/full_audit_20261008/expression_realtime_stable_before_audit.pt` 复制到 `models/expression_realtime_stable.pt` 后重新启动；核对旧 SHA。回退不能解决默认机器隐私问题。

## 交接、包与证据位置

数据分包目录 `D:\temp\expression-deid-datasets-20261004`：`01_kmu.7z`、`02_raf-db.7z`、`03_fer2013-ferplus.7z`、`04_ppb-emo.7z.001/.002`、`05_lfw.7z`。已逐包 `7z t`，5 组全部成功，6 个文件均小于十进制 1 GB；PPB 两卷必须放在同目录一起解压。完整大小/SHA 在 `dataset_archives.json` 与分包目录的 `SHA256SUMS.txt`。

完整模型/外部代码包另存 `D:\temp\expression-deid-models-20261009.zip`，833,190,046 字节，小于 1 GB，57 个文件，ZIP CRC 完整检查通过；SHA-256 为 `10639eb306450b1b5de0df5eb3fd9038bcfca4426ca97a45b8bad1511fbc7c58`。解压到仓库根目录，核对模型 SHA 再启动。这个包不含数据集或用户照片。

Git 保留代码、精简结果、报告和小路径 ZIP，不提交匿名训练缓存、真实用户相机帧、大模型或原始诊断逐帧大文件。`handoff/asset_manifest.json/csv` 记录单文件大小和 SHA，换机以 `relative_path` 为准；不要仅依据文件名判断权重版本。仍需外部 `external_ganonymization/` 代码和模型文件。

主要原始证据：

| 文件（均在 `outputs/full_audit_20261008/` 下） | 内容 |
|---|---|
| `pre_audit.patch` | 检查开始前已有改动备份，未丢弃原工作 |
| `datasets.json` | 所有 manifest/RAF 文件与标签/重复检查 |
| `post_cli_defaults_entrypoints.json` | 18 帮助入口及 8 组实际入口检查 |
| `training_paths.json` | 五条匿名夹具训练、损失、保存信息 |
| `final_input_equivalence.json` | 8532 个实际输入逐张像素差及 cache signature |
| `final_expression_comparison.json` | 同一当前匿名输入的新旧指标、混淆矩阵、各类准确率 |
| `final_fast_privacy.json`、`final_ratio_privacy.json`、`final_motion_privacy.json` | 各 220 查询，逐图独立身份诊断 |
| `final_v2_neutral/`、`final_v2_examples/` | 最终逐帧、未知覆盖、切换、FPS、显示失败 |
| `motion_video/`、`ratio_video/` | 被拒匿名候选的逐帧失败证据 |
| `desktop_runtime/summary.json` | 真 Qt worker/退出、编码、无脸、错误文件、摄像头检查 |
| `detector_reference.json` | 手写/原生 YuNet 数值比较 |
| `final_gan_runtime.json`、`final_gan_runtime_full.json` | 当前 desktop/GAN 的 24 张和全部 220 张尾帧回归 |
| `final_models/models.json` | 25 权重和外部资产的完整 SHA、finite/形状/FP16、域指标 |
| `final_sources/sources.json` | 每个项目 Python 文件 SHA、定义列表、语法状态 |
| `final_runtime_precision.json` | 3357 个输入 FP32/实际 FP16 全量差异 |
| `deployment.json` | 已部署权重/备份哈希、隐私门槛 false |
| `dataset_archives.json` | 数据六个分包大小、SHA、解压完整性 |
| `rejected_mobile_experiment/` | 被拒架构与 25 轮结果，未接入应用 |

以下附表由上述当前 JSON 生成，完整机器可读精简结果为 `audit-results-20261009.json`。


## 附表：逐权重实际检查

全部 25 个 .pt 权重有限、输出形状正确、CUDA FP32/FP16 输出有限。下表表情准确率采用**共同的当前 fast 匿名缓存**：RAF valid 3068，KMU valid 220，PPB valid 34/test 35。它可比较当前 fast 输入适配性，不能单凭这张表宣布 GAN 分类器在其本来生成域失效。六个匿名网络逐个检查权重和前向有效，身份隐私合格与否需要完整合成后独立验收。两个 ID 模型为 12 类，图库为 1680 外部身份，不能混淆。

| 权重 | 类型 | 输出形状 | FP16 最大绝对差 | RAF/KMU/PPB valid/test (%) | SHA-256 前 16 位 | 状态 |
|---|---|---|---:|---|---|---|
| anonymizer.pt | anonymizer | [8, 1, 100, 100] | 0.000832 | 不适用 | `a9021f765492aea2` | pass |
| anonymizer_adversarial_v1.pt | anonymizer | [8, 1, 100, 100] | 0.001198 | 不适用 | `3870f572051e813e` | pass |
| anonymizer_before_identity_cosine.pt | anonymizer | [8, 1, 100, 100] | 0.000844 | 不适用 | `23044d18f046e3f2` | pass |
| anonymizer_identity_cosine_v1.pt | anonymizer | [8, 1, 100, 100] | 0.000937 | 不适用 | `3d6340b14f9c5cb1` | pass |
| anonymizer_quality.pt | anonymizer | [8, 1, 100, 100] | 0.000832 | 不适用 | `a9021f765492aea2` | pass |
| anonymizer_residual_v2.pt | anonymizer | [8, 1, 100, 100] | 0.001482 | 不适用 | `9e0c19acdb29ee13` | pass |
| expression.pt | expression | [8, 7] | 0.011349 | 62.84 / 87.73 / 32.35 / 48.57 | `5d57fd11632ab98c` | pass |
| expression_anonymized.pt | expression | [8, 7] | 0.020302 | 63.82 / 92.27 / 82.35 / 80.00 | `2b31b65049d69f8f` | pass |
| expression_gan_adapted.pt | expression | [8, 7] | 0.013822 | 63.14 / 95.91 / 79.41 / 82.86 | `83a751a44936a835` | pass |
| expression_gan_final.pt | expression | [8, 7] | 0.008693 | 62.35 / 92.27 / 82.35 / 74.29 | `b0e9daa9b36b4bca` | pass |
| expression_gan_geometry.pt | expression | [8, 7] | 0.024413 | 65.12 / 94.09 / 73.53 / 71.43 | `635afb277d22ac9d` | pass |
| expression_gan_innerface.pt | expression | [8, 7] | 0.019398 | 62.06 / 91.36 / 70.59 / 71.43 | `dcfc753808c0aa0d` | pass |
| expression_gan_runtime_ppb.pt | expression | [8, 7] | 0.019547 | 63.46 / 92.73 / 79.41 / 82.86 | `f2e71c7f395e4bbb` | pass |
| expression_gan_split.pt | expression | [8, 7] | 0.012439 | 63.20 / 92.73 / 76.47 / 74.29 | `a9b303e5251b63bf` | pass |
| expression_gan_split_ppb.pt | expression | [8, 7] | 0.008911 | 62.91 / 94.09 / 82.35 / 88.57 | `c78e3427518b1ec0` | pass |
| expression_gan_strength100.pt | expression | [8, 7] | 0.018516 | 63.59 / 93.64 / 82.35 / 77.14 | `a1304fac673ceb8d` | pass |
| expression_joint.pt | expression | [8, 7] | 0.012064 | 62.03 / 82.73 / 14.71 / 17.14 | `6652e2dcbccf3f0f` | pass |
| expression_joint_ferplus.pt | expression | [8, 7] | 0.011349 | 62.84 / 87.73 / 32.35 / 48.57 | `5d57fd11632ab98c` | pass |
| expression_joint_ppb.pt | expression | [8, 7] | 0.017607 | 61.38 / 88.18 / 52.94 / 80.00 | `dc2f360785ccea4c` | pass |
| expression_kmu.pt | expression | [8, 7] | 0.011432 | 36.21 / 81.82 / 5.88 / 0.00 | `bda732d487040dfc` | pass |
| expression_machine_selector_texture.pt | expression | [8, 7] | 0.029073 | 62.71 / 92.73 / 67.65 / 74.29 | `feb78392e6431694` | pass |
| expression_raf.pt | expression | [8, 7] | 0.006994 | 60.14 / 27.73 / 44.12 / 68.57 | `d97829754cfd841b` | pass |
| expression_realtime_stable.pt | expression | [8, 7] | 0.015744 | 68.42 / 100.00 / 100.00 / 100.00 | `5a98c02d10439dd2` | pass |
| identity.pt | identity | [8, 12] | 0.022895 | 不适用 | `bc2e21b4d2af07a3` | pass |
| identity_faceonly.pt | identity | [8, 12] | 0.032070 | 不适用 | `13a8bbebb34f85ca` | pass |

### 外部模型资产

| 文件 | 字节数 | SHA-256 | 已执行路径 |
|---|---:|---|---|
| face_detection_yunet_2023mar.onnx | 232589 | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` | 6 张原生对照 + 视频检测 |
| face_landmarker.task | 3758596 | `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff` | Qt/GAN 和 6300 视频帧关键点 |
| face_recognition_sface_2021dec.onnx | 38696353 | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` | 220×3 隐私查询 + Qt/GAN 诊断 |
| ganonymization_50.ckpt | 686311019 | `34e6698712b50f9325ac36ec3daf8febb5cca6813a1605bc2d7ecb78a896fe62` | 真实生成、视频显示和匿名缓存准备 |

### 当前匿名 RAF 各类准确率

| 类别 | 样本 | 旧模型 | 新模型 |
|---|---:|---:|---:|
| 惊讶 | 329 | 57.45% | 71.73% |
| 恐惧 | 74 | 58.11% | 54.05% |
| 厌恶 | 160 | 18.75% | 28.12% |
| 高兴 | 1185 | 70.63% | 76.96% |
| 悲伤 | 478 | 52.51% | 57.11% |
| 愤怒 | 162 | 56.79% | 65.43% |
| 中性 | 680 | 60.88% | 71.62% |

### 最终逐段视频分析

| 视频 | 帧/检测/决策 | 最终跳变 | 稳态成功分析 FPS | 单帧 p95 ms | 关键点失败/遮蔽 |
|---|---|---:|---:|---:|---|
| PPB_Emo_dataset@video-face-CIR-30s-P04-ND.mp4 | 900/900/897 | 0 | 33.46 | 153.81 | 0/0 |
| PPB_Emo_dataset@video-face-CIR-30s-P05-ND.mp4 | 900/900/897 | 0 | 33.56 | 153.54 | 0/0 |
| PPB_Emo_dataset@video-face-CIR-30s-P07-ND.mp4 | 900/900/897 | 0 | 33.29 | 134.88 | 5/3 |
| PPB_Emo_dataset@video-face-CIR-30s-P22-ND.mp4 | 900/900/897 | 0 | 33.99 | 151.53 | 0/0 |
| PPB_Emo_dataset@video-face-CIR-30s-P24-ND.mp4 | 900/900/897 | 0 | 29.54 | 141.49 | 150/130 |
| PPB_Emo_dataset@video-face-CIR-30s-P38-ND.mp4 | 900/890/879 | 0 | 33.38 | 114.93 | 5/5 |
| PPB_Emo_dataset@video-face-CIR-30s-P39-ND.mp4 | 900/900/897 | 0 | 33.56 | 149.73 | 0/0 |
| example_kmu_expressions.mp4 | 120/120/118 | 5 | 31.10 | 134.96 | 0/0 |
| example_mixed.mp4 | 96/95/91 | 3 | 29.13 | 135.07 | 0/0 |
| example_anger.mp4 | 24/24/22 | 0 | 31.62 | 152.44 | 0/0 |
| example_surprise.mp4 | 24/24/22 | 0 | 33.88 | 126.38 | 0/0 |

### 逐文件源码指纹

项目 Python 文件 43 个均通过 AST 和 compileall；完整定义列表在机器可读结果中。这里的语法状态不等于所有业务路径均已实测，执行范围以检查矩阵为准。

| 文件 | 行数 | SHA-256 前 16 位 | 语法 |
|---|---:|---|---|
| `expression_deid\__init__.py` | 5 | `24413d6d2ccdcc51` | pass |
| `expression_deid\__main__.py` | 5 | `f879bba9ddeb0856` | pass |
| `expression_deid\align.py` | 144 | `8159f8fc68cecf1f` | pass |
| `expression_deid\app.py` | 266 | `685b9aa3c44b643e` | pass |
| `expression_deid\cli.py` | 116 | `f4baf9a1b2ab3c61` | pass |
| `expression_deid\config.py` | 58 | `15b589e9dd604377` | pass |
| `expression_deid\data.py` | 396 | `468170c2039b9c9e` | pass |
| `expression_deid\desktop_ui.py` | 11 | `5f5c1ee4926e8fc3` | pass |
| `expression_deid\desktop_ui_qt.py` | 473 | `d9820878f1a08448` | pass |
| `expression_deid\detect.py` | 121 | `aa735e5233d70ca4` | pass |
| `expression_deid\evaluate.py` | 268 | `9dac9491dedf0a97` | pass |
| `expression_deid\feature_geometry.py` | 111 | `cd740c93ddbe8bf4` | pass |
| `expression_deid\ganonymization_backend.py` | 149 | `cfb8bf9c36f7c8b8` | pass |
| `expression_deid\identity_gallery.py` | 109 | `a75fc5aa3085101e` | pass |
| `expression_deid\inference.py` | 1060 | `69dfd90a82d33b6c` | pass |
| `expression_deid\labels.py` | 50 | `46f8a3b740ad5218` | pass |
| `expression_deid\models.py` | 237 | `f7028c717cdf0e26` | pass |
| `expression_deid\normalize.py` | 201 | `337caafc82425801` | pass |
| `expression_deid\prepare.py` | 269 | `0cf93d5dfb091231` | pass |
| `expression_deid\prepare_ppb.py` | 123 | `5a66ba4352adc00d` | pass |
| `expression_deid\temporal.py` | 103 | `0de85b62b496f0ec` | pass |
| `expression_deid\train.py` | 701 | `ce02468505ae91d8` | pass |
| `expression_deid\video_demo.py` | 443 | `c9ba57969646275b` | pass |
| `expression_deid\web.py` | 377 | `85d14c678e7a9f75` | pass |
| `scripts\audit_entrypoints.py` | 199 | `eab80ccc0546da30` | pass |
| `scripts\audit_gan_runtime.py` | 49 | `0409a3318935739c` | pass |
| `scripts\audit_project.py` | 188 | `3c99c95c46b7cdaf` | pass |
| `scripts\audit_runtime_privacy.py` | 59 | `bd64840772a1cfc4` | pass |
| `scripts\build_lfw_gallery.py` | 46 | `1366b1697cac023f` | pass |
| `scripts\check_anonymous_cache.py` | 61 | `3583a6958bf29657` | pass |
| `scripts\check_desktop_runtime.py` | 162 | `51ec9bbfef9869d9` | pass |
| `scripts\check_detector_reference.py` | 39 | `5744566d44be7335` | pass |
| `scripts\check_training_paths.py` | 71 | `896c1392ed6049fa` | pass |
| `scripts\check_video_stability.py` | 105 | `47d8fdfc6fef89d7` | pass |
| `scripts\create_handoff_bundle.py` | 145 | `0c1f5effc1a1f429` | pass |
| `scripts\download_anonymization_assets.py` | 43 | `c51b574dabc3a946` | pass |
| `scripts\prepare_gan_expression.py` | 95 | `09160ba0cfa72c43` | pass |
| `scripts\train_realtime_expression.py` | 266 | `384cbe138c00b85a` | pass |
| `scripts\tune_machine_privacy.py` | 55 | `62d69c3c1fe320f4` | pass |
| `tests\test_audit_regressions.py` | 310 | `9034d4c15a5b396c` | pass |
| `tests\test_models.py` | 143 | `da301239a23b1683` | pass |
| `tests\test_preprocess.py` | 119 | `3c184459e95583f5` | pass |
| `tests\test_temporal.py` | 224 | `1b52b02ca8bbc5ec` | pass |
