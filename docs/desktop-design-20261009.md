# Python 桌面界面更新（2026-10-09）

这次修改针对 PySide6 桌面应用的呈现和交互。沿用已审计的匿名化、表情识别、模型权重和视频 worker；网页代码未修改。全链路审计中的源码指纹对应当时的审计快照，本次桌面呈现复测单独保存。

## 界面与操作

- 左侧三个入口：图片实验、实时分析、模型表现。深色导航与浅色工作区分离；统一字号、留白、图标与指标卡。初始窗口根据屏幕可用区域调整，最低 1100×760。
- 图片实验：点击带标签的缩略图开始分析，或点击“分析全部样本”。原始灰度图与匿名结果等比显示；真实标签、预测表情、置信度、SFace 原身份命中分别显示。分析过的图片再次点击直接查看缓存结果；重新运行全部样本仍执行模型计算。
- 实时分析：接入摄像头、本地视频或现有示例。匿名画面上叠加真实表情和置信度，底部呈现模型分析 FPS、处理帧号与 FPS 曲线。曲线虚线是 24 FPS 目标，不代表每帧保证。叠加文字只在 Qt 绘制层绘制，不修改分类器输入像素。
- 模型表现：选择 RAF 七类、KMU 六类尾帧、PPB 中性验证/测试集。准确率、样本数、各类条形图和验证范围一起变化；身份诊断独立显示快速链路的审计记录。不会将单个示例置信度当成数据集准确率，也不会将 PPB 中性结果当成七类结果。
- 图像控件保留完整 pixmap，调整窗口时重新等比缩放；不会因反复放大缩小丢失原有分辨率。补充 Windows 中文字体加载，使无屏渲染检查也能验证文字，而不是输出方框。

## 启动

在仓库根目录运行：

```powershell
python -m expression_deid.desktop_ui --device cuda
```

右上角窗口关闭按钮会等待模型线程释放资源。实时视频仍使用原有单帧 UI 队列确认机制；界面落后时不会不断堆积图片。

## 本次验证

```powershell
python -m py_compile expression_deid/desktop_ui_qt.py expression_deid/desktop_design.py scripts/check_desktop_presentation.py
python -m unittest discover -s tests -q
python scripts/check_desktop_presentation.py --device cuda
```

52 项原有测试通过。新增呈现检查实际完成：

1. 三页分别在 1440×920、1280×800、1100×760 渲染；窗口尺寸保持正确，模型表现页小窗口可以滚动。
2. 切换四个验证数据集，核对样本总数和各类统计与保存报告一致。
3. 六张示例均使用真实模型完成匿名化、表情分析和身份诊断；核对 `expression_input_source=anonymized_only`，查看已分析样本不会重新启动 worker。
4. 实际示例视频收到 81 帧 Qt 图像；确认匿名画面叠加、FPS 曲线、帧号、队列确认和停止回收正常。
5. worker 完成后窗口可以正常关闭。

本地截图与结果在 `outputs/desktop_design_20261009/`，包括 `images_actual.png`、`video_actual.png`、`performance_actual.png`、`presentation-check.json`。实际查看了三页以及小窗口的渲染结果。这是桌面布局和交互验证，不是新增模型准确率、匿名隐私或实车验证；模型限制继续以全链路审计报告为准。
