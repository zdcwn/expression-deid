# Windows 桌面软件发行

发行包支持 Windows 10/11 x64，携带 Python、Qt、推理库、当前运行权重、1,100 张 KMU 对齐示例图、4 段示例视频、指标记录及第三方许可。CPU 可用；启动时实际运行 CUDA 卷积探测，通过后才启用 NVIDIA GPU。没有兼容驱动或显卡时自动使用 CPU，不要求用户安装 CUDA 工具包。

单文件版双击启动，首次自动展开内置资源到当前用户的本地应用数据目录，后续直接启动缓存中的应用。ZIP 版完整解压后运行主程序。两者都不需要安装 Python、执行命令或下载模型。建议 8 GB 及以上内存，为运行库与展开空间保留至少 8 GB 磁盘空间；CPU 的质量模式速度取决于硬件。

发行时可提供 CPU 轻量版与 GPU 版。CPU 轻量版保留相同模型、全部示例和三个页面，不携带 CUDA 库，因此即使电脑有 NVIDIA 显卡也使用 CPU；GPU 版保留自动选择 NVIDIA 加速和 CPU 回退。构建器会从匿名化训练检查点无损导出生成器，剔除优化器和判别器状态，逐张量核对数值与类型，原始训练检查点不变。

图片页直接列出全部 1,100 张示例，可按实际包含的六类表情筛选；缩略图随滚动加载，避免打开页面时一次性读取全部图片。选中图片后可分析单张，也可批量分析当前筛选下的样本。批量分析可停止，已完成的结果保留在本次会话中，再次选择时直接显示。更新版本后应关闭旧软件并打开新版程序，旧版本的缓存不会自动更新。

## 构建

使用项目已验证的 64 位 Windows 推理环境，另安装 `requirements/distribution.txt`。在项目根目录执行：

```powershell
.\.venv-anon\Scripts\python.exe scripts/distribution/build_windows.py --skip-zip
```

运行冻结后的程序检查，确认通过后生成 ZIP 和单文件版：

```powershell
outputs/distribution/dist/ExpressionLab/ExpressionLab.exe --check-package outputs/distribution/check_cpu --device cpu
outputs/distribution/dist/ExpressionLab/ExpressionLab.exe --check-package outputs/distribution/check_gpu --device auto
.\.venv-anon\Scripts\python.exe scripts/distribution/build_windows.py --skip-build
.\.venv-anon\Scripts\python.exe scripts/distribution/build_single_file.py
```

CPU 轻量版在独立环境中安装 `requirements/distribution-cpu.txt` 后，用该环境的 Python 执行相同构建脚本，并通过 `--output outputs/distribution_cpu_compact` 指定独立输出目录。CPU/GPU 两版分别验证，不能用其中一版的结果代替另一版。只改运行资源时可以对现有冻结目录执行 `--skip-build`，更新资源与哈希清单；修改 Python 运行代码仍须重新冻结。

构建清单明确指定运行资产，不带训练候选、训练集、仓库历史或本地实验输出。`package_manifest.json` 保存资源大小及 SHA-256，拒绝 Git LFS 指针。PyInstaller 配置会收集 Qt 插件、视频解码与模型依赖。单文件启动器由 Windows 自带的 .NET Framework 编译器生成，首次展开结束后才标记缓存就绪；并发启动使用互斥锁，解压检查目录边界。

## 验证与使用状态

GitHub Releases 的单个附件不能达到 2 GiB。GPU 单文件超过该限制时，使用下面的脚本生成每卷 900 MiB 的附件与自动启动器；原始 EXE 不变。CPU 版直接上传原文件。

```powershell
.\.venv-anon\Scripts\python.exe scripts/distribution/build_release_parts.py --source outputs/releases/ExpressionLab-1.0.1/GPU/ExpressionLab-GPU.exe --output outputs/releases/ExpressionLab-1.0.1/github_assets
```

接收者下载 `ExpressionLab-GPU-Start.exe` 和三个 `ExpressionLab-GPU.exe.part1` 至 `.part3`，放在同一文件夹后双击启动器。启动器用 Windows 自带组件合并到当前用户缓存，并核对完整文件的 SHA-256；无需额外解压软件、管理员权限或运行命令。`parts-check.json` 记录原文件与分卷哈希，发布时同时提供 `SHA256SUMS.txt`。上传后核对附件的大小和 GitHub 返回的摘要，再发布 Release。

自检会阻止网络连接，核对资源哈希、解码全部图库与示例视频、检查小屏幕和三个页面，并验证全部图库条目、分类筛选和末尾图片的缩略图。图片推理采用每类抽样及末尾样本，同时运行高速/质量视频分析，保存截图并检查正常退出。源码桌面检查另行对完整图库逐张分析。发布前还需在移除开发环境路径的进程中运行，验证中文路径以及单文件首次展开和缓存复用。检查的是冻结后的程序，不能用源码检查代替。

日志写入当前用户的 `AppData/Local/ExpressionLab/logs`；运行资源只读。关键点模型直接读取字节，避免中文路径和共享临时目录权限问题。字体使用随包携带的 Noto Sans SC 回退，界面保留 Qt 的 DPI 缩放，小屏幕可滚动到所有控件。

此包面向 Windows x64；macOS、Linux 和 Windows ARM 需在对应系统上另行构建与验证。当前机器的验证记录不能代表其他物理电脑、摄像头或驱动都已验收。
