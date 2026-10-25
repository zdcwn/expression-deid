# 接手操作

## 先运行

从项目根目录按 [README](../README.md) 安装依赖并启动 `python -m expression_deid.desktop_ui --device cuda`。Git LFS 用于获取匿名化 GAN 权重。图片页先运行一个带标签示例，再在视频页播放示例，最后检查模型表现页。

## 再找文件

- 修改窗口、交互或界面组件：`expression_deid/ui/`。
- 修改检测、对齐或灰度处理：`expression_deid/preprocessing/`。
- 修改匿名化、表情推理或时序逻辑：`expression_deid/runtime/`。
- 修改网络结构、保存和读取格式：`expression_deid/models.py`。
- 修改训练数据、损失或离线评价：`expression_deid/training/`。
- 启动专门训练、导出或检查：`scripts/` 下的相应分组。

逐个文件的职责和输入输出见[目录与操作路径](architecture/project-layout.md)。不要把 UI 的显示刷新率当作模型分析帧率。

## 替换模型

训练输出保存在 `models/training/`。先使用同一组验证输入比较候选，再将选中权重部署到 `models/anonymization/` 或 `models/expression/`。质量模型必须连同 JSON 和其引用的权重一起核对。桌面端不会自动加载训练目录中的候选。

## 提交前检查

```powershell
python -m compileall -q expression_deid scripts tests
python -m unittest discover -s tests -p "test_*.py" -q
python scripts/checks/check_desktop_presentation.py --device cuda
```

确认原图没有进入表情分类器，示例路径能解析，窗口关闭后线程与视频资源正常释放。用户照片、匿名缓存、逐帧记录和训练候选不提交。
