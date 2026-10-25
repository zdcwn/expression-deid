# 课程报告材料

- `tongji_final_report/main.pdf`：项目研究与实践报告，除封面 15 页。
- `tongji_project_plan/main.pdf`：项目计划书，除封面 10 页。
- 两份目录各 1 页，均无摘要。
- 两份可编译源码、图片、模板和参考文献保存在对应目录。
- `交付检查.md`：本轮内容与版式逐项检查。

在项目根目录执行 `./reports/build_reports.ps1` 可重新生成两份 PDF；也可用 `-Target tongji_final_report` 或 `-Target tongji_project_plan` 单独编译。源文件使用 XeLaTeX 和 BibTeX，当前机器已配置可用的本地编译工具。

两份封面已补齐六位成员姓名与学号、学院、专业、课程和授课教师。研究与实践报告的副标题为“项目研究与实践报告”，另一份保留“项目计划书”。

## 文件范围

两份工程均保留正文、封面、图片、参考文献、模板样式及编译入口。`main.pdf` 为已编译版本，可直接查看；继续编辑时修改相应 `sections/` 文件即可。

编译需要 Windows 中文字体，以及包含 XeLaTeX、BibTeX 和 GB/T 7714 参考文献样式的 TeX 发行版。模板来源及许可见 `template-license/`。本地草稿备份、模板解压目录、临时修订脚本和编译日志不属于交付文件。
