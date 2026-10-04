"""High-fidelity Qt desktop application for the expression privacy project.

This module is intentionally separate from the browser frontend.  It uses
PySide6 for a product-like desktop shell while reusing the same Python model
and video workers.  The expression classifier receives only the anonymized
face in every realtime frame.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, QPoint, QRect, QSize, Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPushButton,
    QFileDialog, QScrollArea, QSizePolicy, QStackedWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from expression_deid.app import _image_frame, _pick_checkpoint, _resize_rgb, build_pipeline
from expression_deid.config import KMU_ALIGNED_DIR, ROOT
from expression_deid.inference import ExpressionDeidentifier
from expression_deid.video_demo import FastVideoRunner


BG = "#0b1417"
SIDEBAR = "#101d21"
SURFACE = "#132328"
SURFACE_2 = "#182b30"
SURFACE_3 = "#20363b"
BORDER = "#29454a"
TEXT = "#e5f1ef"
MUTED = "#8da6a5"
FAINT = "#5e7778"
MINT = "#b9edce"
AQUA = "#62e3ce"
YELLOW = "#f1cb73"
RED = "#f08376"


def qimage_from_array(array: np.ndarray) -> QImage:
    array = np.asarray(array)
    if array.ndim == 2:
        image = QImage(array.data, array.shape[1], array.shape[0], array.strides[0], QImage.Format_Grayscale8)
    else:
        rgb = np.ascontiguousarray(array[:, :, :3])
        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
    return image.copy()


def set_image(label: QLabel, array: np.ndarray, *, max_size: QSize | None = None):
    image = qimage_from_array(array)
    pixmap = QPixmap.fromImage(image)
    if max_size is None:
        max_size = label.size()
    if max_size.width() > 10 and max_size.height() > 10:
        pixmap = pixmap.scaled(max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    label.setPixmap(pixmap)
    label.setText("")


def samples() -> list[dict]:
    manifest = KMU_ALIGNED_DIR / "manifest.jsonl"
    result = {}
    if not manifest.is_file():
        return []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        label = int(record.get("label", -1))
        path = (KMU_ALIGNED_DIR / str(record.get("output", ""))).resolve()
        if label not in result and path.is_file() and path.is_relative_to(KMU_ALIGNED_DIR.resolve()):
            result[label] = {"label": label, "expression": record.get("expression", str(label)), "path": path, "source": record.get("source", path.name), "person": record.get("person", "")}
    return [result[key] for key in sorted(result)]


def diagnostic_pipeline(device: str) -> ExpressionDeidentifier:
    anonymizer = _pick_checkpoint((ROOT / "models/anonymizer_adversarial_v1.pt", ROOT / "models/anonymizer.pt"))
    expression = _pick_checkpoint((ROOT / "models/expression_machine_selector_texture.pt", ROOT / "models/expression.pt"))
    identity = _pick_checkpoint((ROOT / "models/identity_faceonly.pt", ROOT / "models/identity.pt"))
    # Keep the desktop display branch aligned with the web display branch:
    # the human-facing result uses the landmark-only GAN when its checkpoint
    # and audited repository are available.  The expression classifier still
    # receives the separate machine-targeted anonymized face.
    gan_checkpoint = ROOT / "models/ganonymization_50.ckpt"
    gan_repo = ROOT / "external_ganonymization"
    import torch
    resolved = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    return ExpressionDeidentifier(
        anonymizer,
        expression,
        identity_checkpoint=identity,
        device=resolved,
        fast_mode=True,
        privacy_selection=False,
        identity_gallery_size=1680,
        ganonymization_checkpoint=gan_checkpoint if gan_checkpoint.is_file() else None,
        ganonymization_repo=gan_repo if gan_repo.is_dir() else None,
    )


class MetricCard(QFrame):
    def __init__(self, title: str, value: str, note: str = "", accent: bool = True, parent=None):
        super().__init__(parent)
        self.setProperty("card", True)
        layout = QVBoxLayout(self); layout.setContentsMargins(18, 15, 18, 15); layout.setSpacing(3)
        title_label = QLabel(title.upper()); title_label.setObjectName("cardTitle")
        value_label = QLabel(value); value_label.setObjectName("cardValue" if accent else "cardValueMuted")
        note_label = QLabel(note); note_label.setObjectName("cardNote")
        layout.addWidget(title_label); layout.addWidget(value_label); layout.addWidget(note_label)
        self.value_label = value_label

    def set_value(self, value: str):
        self.value_label.setText(value)


class ExperimentWorker(QThread):
    result_ready = Signal(object, int, int, float)
    error = Signal(str)
    finished_all = Signal(int)

    def __init__(self, items: list[dict], device: str):
        super().__init__(); self.items = items; self.device = device

    def run(self):
        try:
            pipeline = diagnostic_pipeline(self.device)
            for index, item in enumerate(self.items, start=1):
                started = time.perf_counter()
                result = pipeline.process(item["path"], aligned=True, diagnostics=True)
                self.result_ready.emit({"item": item, "result": result}, index, len(self.items), time.perf_counter() - started)
            self.finished_all.emit(len(self.items))
        except Exception as exc:  # pragma: no cover - dependent on local model files
            self.error.emit(str(exc))


class ExperimentPage(QWidget):
    def __init__(self, device: str):
        super().__init__(); self.device = device; self.items = samples(); self.worker = None; self._build()

    def _build(self):
        root = QVBoxLayout(self); root.setContentsMargins(28, 24, 28, 24); root.setSpacing(18)
        title = QLabel("实验图片对照"); title.setObjectName("pageTitle")
        sub = QLabel("带标签样本 → 局部匿名化 → 匿名图表情分析 → 身份诊断"); sub.setObjectName("pageSubtitle")
        root.addWidget(title); root.addWidget(sub)
        content = QHBoxLayout(); content.setSpacing(18); root.addLayout(content, 1)

        sidebar = QFrame(); sidebar.setProperty("panel", True); sidebar.setFixedWidth(248)
        left = QVBoxLayout(sidebar); left.setContentsMargins(16, 16, 16, 16); left.setSpacing(8)
        left.addWidget(QLabel("KMU 示例数据", objectName="panelTitle"))
        left.addWidget(QLabel("选择一张样本，查看完整验收链路。", objectName="panelNote"))
        self.sample_list = QListWidget(); self.sample_list.setObjectName("sampleList")
        for index, item in enumerate(self.items, start=1):
            row = QListWidgetItem(f"{index:02d}   {item['expression']}   /   {item['person']}")
            row.setData(Qt.UserRole, item); self.sample_list.addItem(row)
        left.addWidget(self.sample_list, 1)
        self.run_button = QPushButton("运行全部样本   →"); self.run_button.setObjectName("primaryButton"); self.run_button.clicked.connect(self.run_all); left.addWidget(self.run_button)
        self.exp_status = QLabel("等待运行身份诊断"); self.exp_status.setObjectName("panelNote"); self.exp_status.setWordWrap(True); left.addWidget(self.exp_status)
        content.addWidget(sidebar)

        center = QFrame(); center.setProperty("panel", True); center_layout = QVBoxLayout(center); center_layout.setContentsMargins(18, 16, 18, 16); center_layout.setSpacing(10)
        media_header = QHBoxLayout(); media_header.addWidget(QLabel("对照结果", objectName="panelTitle")); media_header.addStretch(); media_header.addWidget(QLabel("FACE-ONLY EDIT", objectName="statusChip")); center_layout.addLayout(media_header)
        media = QHBoxLayout(); media.setSpacing(12)
        self.original_view = self._media_card("ORIGINAL INPUT", "选择样本开始", False); self.anonymized_view = self._media_card("ANONYMIZED OUTPUT", "等待处理", True)
        media.addWidget(self.original_view[0], 1); media.addWidget(self.anonymized_view[0], 1); center_layout.addLayout(media, 1)
        self.metric_line = QLabel("表情：—      匿名图置信度：—      ID：—"); self.metric_line.setObjectName("metricLine"); center_layout.addWidget(self.metric_line)
        self.detail_line = QLabel("SFace：—      1681 人库命中：—"); self.detail_line.setObjectName("detailLine"); center_layout.addWidget(self.detail_line)
        content.addWidget(center, 1)

        self.sample_list.itemClicked.connect(self.run_selected)

    def _media_card(self, heading: str, empty: str, accent: bool):
        frame = QFrame(); frame.setProperty("mediaCard", True); layout = QVBoxLayout(frame); layout.setContentsMargins(10, 10, 10, 10); layout.setSpacing(8)
        top = QHBoxLayout(); top.addWidget(QLabel(heading, objectName="mediaHeading")); top.addStretch(); top.addWidget(QLabel("ANONYMIZED" if accent else "SOURCE", objectName="mediaTag")); layout.addLayout(top)
        image = QLabel(empty); image.setObjectName("mediaImage"); image.setAlignment(Qt.AlignCenter); image.setMinimumSize(360, 300); image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding); layout.addWidget(image, 1)
        return frame, image

    @Slot()
    def run_selected(self):
        row = self.sample_list.currentItem()
        if row: self._start([row.data(Qt.UserRole)])

    @Slot()
    def run_all(self): self._start(self.items)

    def _start(self, items):
        if self.worker and self.worker.isRunning(): return
        self.run_button.setEnabled(False); self.exp_status.setText("正在加载身份诊断模型…")
        self.worker = ExperimentWorker(items, self.device); self.worker.result_ready.connect(self._show_result); self.worker.error.connect(self._error); self.worker.finished_all.connect(self._finished); self.worker.start()

    @Slot(object, int, int, float)
    def _show_result(self, payload, index, total, elapsed):
        item, result = payload["item"], payload["result"]
        set_image(self.original_view[1], np.asarray(_image_frame(item["path"])), max_size=self.original_view[1].size())
        set_image(self.anonymized_view[1], result["anonymized"], max_size=self.anonymized_view[1].size())
        exp = result.get("expression_anonymized", {}); changed = result.get("identity_prediction_changed", False)
        self.metric_line.setText(f"标签 {item['expression']}  →  {exp.get('name_zh', exp.get('name', '—'))}     ·     匿名图置信度 {float(exp.get('confidence', 0)) * 100:.1f}%     ·     ID {'已改变' if changed else '未改变'}")
        self.detail_line.setText(f"SFace {float(result.get('sface_cosine', 0)):.3f}     ·     1681 人库原身份命中：{'是' if result.get('gallery_match_original') else '否'}     ·     本张 {elapsed * 1000:.0f} ms     ·     {index}/{total}")
        self.exp_status.setText(f"已完成 {index}/{total} 张样本")

    def _error(self, message): self.exp_status.setText(f"实验失败：{message}")
    def _finished(self, count): self.run_button.setEnabled(True); self.exp_status.setText(f"完成 {count} 张样本的身份诊断")


class VideoWorker(QThread):
    frame_ready = Signal(object, object, str, float, float, int)
    worker_error = Signal(str)
    ended = Signal()

    def __init__(self, source, device: str):
        super().__init__(); self.source = source; self.device = device; self.running = True

    def stop(self): self.running = False

    def run(self):
        capture = None
        try:
            # Match the web visual output while keeping the realtime analysis
            # path responsive: the classifier pipeline stays on the fast local
            # edit, and the human-facing GAN is refreshed every fifteen frames for
            # display only.  FastVideoRunner never sends that display branch to
            # the expression classifier.
            pipeline = build_pipeline(self.device, use_gan=False)
            display_pipeline = build_pipeline(self.device, use_gan=True)
            runner = FastVideoRunner(
                pipeline,
                detect_interval=5,
                display_pipeline=display_pipeline,
                display_interval=15,
            )
            kind, value = self.source; capture = cv2.VideoCapture(int(value), cv2.CAP_DSHOW) if kind == "camera" else cv2.VideoCapture(str(value))
            if not capture.isOpened(): raise RuntimeError("无法打开摄像头或视频文件")
            ticks = deque(maxlen=24); index = 0
            while self.running:
                ok, bgr = capture.read()
                if not ok: break
                rgb = _resize_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), 960)
                anonymized, info = runner.process_frame(rgb, index)
                ticks.append(time.perf_counter())
                fps = (len(ticks) - 1) / max(1e-6, ticks[-1] - ticks[0]) if len(ticks) > 1 else 0.0
                label = info.get("expression_zh", info.get("expression", "未检测到人脸")); confidence = float(info.get("confidence") or 0.0)
                self.frame_ready.emit(qimage_from_array(pipeline._image_gray(rgb)), qimage_from_array(anonymized), label, confidence, fps, index); index += 1
            self.ended.emit()
        except Exception as exc: self.worker_error.emit(str(exc))
        finally:
            if capture is not None: capture.release()


class VideoPage(QWidget):
    def __init__(self, device: str):
        super().__init__(); self.device = device; self.worker = None; self._build()

    def _build(self):
        root = QVBoxLayout(self); root.setContentsMargins(28, 24, 28, 24); root.setSpacing(16)
        head = QHBoxLayout(); title = QVBoxLayout(); title.addWidget(QLabel("实时摄像头 / 视频", objectName="pageTitle")); title.addWidget(QLabel("完整匿名化 + 表情识别链路的真实分析帧率", objectName="pageSubtitle")); head.addLayout(title); head.addStretch()
        self.camera_button = QPushButton("打开摄像头"); self.camera_button.setObjectName("primaryButton"); self.camera_button.clicked.connect(self.start_camera); head.addWidget(self.camera_button)
        choose = QPushButton("选择视频"); choose.clicked.connect(self.choose_video); head.addWidget(choose)
        self.stop_button = QPushButton("停止"); self.stop_button.clicked.connect(self.stop); head.addWidget(self.stop_button); root.addLayout(head)
        bar = QHBoxLayout(); bar.addWidget(QLabel("示例视频", objectName="panelNote")); self.examples = QComboBox(); self.examples.addItems([path.name for path in sorted((ROOT / "outputs").glob("example*.mp4"))]); bar.addWidget(self.examples); play = QPushButton("播放示例"); play.clicked.connect(self.play_example); bar.addWidget(play); bar.addStretch(); self.source_label = QLabel("未选择来源"); self.source_label.setObjectName("panelNote"); bar.addWidget(self.source_label); root.addLayout(bar)
        media = QHBoxLayout(); media.setSpacing(16); self.original = self._media_card("ORIGINAL VIDEO", "等待输入", False); self.anonymized = self._media_card("ANONYMIZED VIDEO", "等待推理", True); media.addWidget(self.original[0], 1); media.addWidget(self.anonymized[0], 1); root.addLayout(media, 1)
        metrics = QHBoxLayout(); self.expression = MetricCard("当前表情", "—", "匿名图预测"); self.confidence = MetricCard("置信度", "—", "校准后验"); self.fps = MetricCard("真实分析帧率", "—", "匿名化 + 识别"); self.status = QLabel("等待视频"); self.status.setObjectName("statusChip"); metrics.addWidget(self.expression); metrics.addWidget(self.confidence); metrics.addWidget(self.fps); metrics.addWidget(self.status); root.addLayout(metrics)

    def _media_card(self, heading, empty, accent):
        frame = QFrame(); frame.setProperty("mediaCard", True); layout = QVBoxLayout(frame); layout.setContentsMargins(12, 12, 12, 12); top = QHBoxLayout(); top.addWidget(QLabel(heading, objectName="mediaHeading")); top.addStretch(); top.addWidget(QLabel("FACE-ONLY" if accent else "SOURCE", objectName="mediaTag")); layout.addLayout(top); image = QLabel(empty); image.setObjectName("videoImage"); image.setAlignment(Qt.AlignCenter); image.setMinimumSize(430, 300); image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding); layout.addWidget(image, 1); return frame, image

    def _start(self, source, text):
        self.stop(); self.source_label.setText(text); self.status.setText("正在加载 GPU 模型…"); self.worker = VideoWorker(source, self.device); self.worker.frame_ready.connect(self._frame); self.worker.worker_error.connect(self._error); self.worker.ended.connect(lambda: self.status.setText("视频结束")); self.worker.start()

    def start_camera(self): self._start(("camera", 0), "本机摄像头")
    def choose_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", str(ROOT / "outputs"), "视频 (*.mp4 *.avi *.mov *.mkv *.webm)")
        if path: self._start(("video", path), Path(path).name)
    def play_example(self):
        if self.examples.currentText(): self._start(("video", str(ROOT / "outputs" / self.examples.currentText())), f"示例视频 · {self.examples.currentText()}")
    def stop(self):
        if self.worker and self.worker.isRunning(): self.worker.stop(); self.worker.wait(1200)
        self.worker = None; self.status.setText("已停止")

    @Slot(object, object, str, float, float, int)
    def _frame(self, original, anonymized, expression, confidence, fps, index):
        for image, label in ((original, self.original[1]), (anonymized, self.anonymized[1])):
            pixmap = QPixmap.fromImage(image); pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation); label.setPixmap(pixmap); label.setText("")
        self.expression.set_value(expression); self.confidence.set_value(f"{confidence * 100:.1f}%"); self.fps.set_value(f"{fps:.1f} FPS"); self.status.setText("运行中 · 仅匿名图进入表情模型")

    def _error(self, message): self.status.setText(f"分析失败：{message}")


class PerformancePage(QWidget):
    def __init__(self, device: str):
        super().__init__(); self.device = device; self._build()

    def _build(self):
        root = QVBoxLayout(self); root.setContentsMargins(28, 24, 28, 24); root.setSpacing(18); root.addWidget(QLabel("整体模型表现", objectName="pageTitle")); root.addWidget(QLabel("离线联合评估 · 1100 张匿名样本 · 1681 人身份图库", objectName="pageSubtitle"))
        path = ROOT / "outputs/evaluation_machine_runtime_fast_strong_1681/evaluation.json"
        try: summary = json.loads(path.read_text(encoding="utf-8"))["summary"]
        except Exception: summary = {}
        cards = QGridLayout(); cards.setSpacing(12); root.addLayout(cards)
        values = [("匿名图表情准确率", f"{summary.get('expression_anonymized_accuracy', 0) * 100:.1f}%", "表情模型只接收匿名图"), ("闭集身份准确率", f"{summary.get('identity_anonymized_accuracy', 0) * 100:.1f}%", "身份诊断"), ("1681 人库命中率", f"{summary.get('gallery_original_match_rate', 0) * 100:.1f}%", "越低越好"), ("SFace 平均相似度", f"{summary.get('sface_mean_cosine', 0):.3f}", "脸部紧裁区域"), ("评估样本", str(summary.get("samples", "—")), "KMU 数据集"), ("实时设备", self.device.upper(), "支持 CUDA")]
        for index, value in enumerate(values): cards.addWidget(MetricCard(*value), index // 3, index % 3)
        panel = QFrame(); panel.setProperty("panel", True); panel_layout = QVBoxLayout(panel); panel_layout.setContentsMargins(18, 16, 18, 16); panel_layout.addWidget(QLabel("各表情类别匿名后准确率", objectName="panelTitle")); table = QTableWidget(0, 3); table.setHorizontalHeaderLabels(["类别", "样本数", "准确率"]); table.horizontalHeader().setStretchLastSection(True); table.verticalHeader().setVisible(False); table.setMinimumHeight(250); table.setEditTriggers(QTableWidget.NoEditTriggers); names = ["惊讶", "恐惧", "厌恶", "高兴", "悲伤", "愤怒", "中性"]
        for key, row in sorted(summary.get("expression_anonymized_by_class", {}).items(), key=lambda item: int(item[0])):
            r = table.rowCount(); table.insertRow(r); table.setItem(r, 0, QTableWidgetItem(names[int(key)] if int(key) < len(names) else key)); table.setItem(r, 1, QTableWidgetItem(str(row.get("samples", "—")))); table.setItem(r, 2, QTableWidgetItem(f"{row.get('accuracy', 0) * 100:.1f}%"))
        panel_layout.addWidget(table); root.addWidget(panel)
        chain = QFrame(); chain.setProperty("panel", True); chain_layout = QHBoxLayout(chain); chain_layout.setContentsMargins(20, 18, 20, 18)
        for index, text in enumerate(("01  输入灰度化", "02  局部脸部匿名化", "03  匿名图表情识别")):
            label = QLabel(text); label.setObjectName("pipelineStep"); chain_layout.addWidget(label)
            if index < 2: arrow = QLabel("→"); arrow.setObjectName("pipelineArrow"); chain_layout.addWidget(arrow)
        root.addWidget(chain)


class MainWindow(QMainWindow):
    def __init__(self, device: str):
        super().__init__(); self.device = device; self.setWindowTitle("Expression Lab · 座舱隐私实验台"); self.resize(1440, 920); self.setMinimumSize(1180, 760); self.setStyleSheet(STYLESHEET)
        shell = QWidget(); layout = QHBoxLayout(shell); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(0); self.setCentralWidget(shell)
        nav = QFrame(); nav.setObjectName("sidebar"); nav.setFixedWidth(246); nav_layout = QVBoxLayout(nav); nav_layout.setContentsMargins(22, 24, 22, 22); nav_layout.setSpacing(8)
        brand = QLabel("EXPRESSION\nLAB"); brand.setObjectName("brand"); nav_layout.addWidget(brand); nav_layout.addWidget(QLabel("CABIN PRIVACY / LOCAL", objectName="brandSub")); nav_layout.addSpacing(28)
        self.stack = QStackedWidget(); self.pages = [ExperimentPage(device), VideoPage(device), PerformancePage(device)]
        for page in self.pages: self.stack.addWidget(page)
        self.nav_buttons = []
        for index, text in enumerate(("01   实验图片对照", "02   实时视频分析", "03   整体模型表现")):
            button = QPushButton(text); button.setProperty("nav", True); button.setCheckable(True); button.clicked.connect(lambda checked, i=index: self.select_page(i)); nav_layout.addWidget(button); self.nav_buttons.append(button)
        nav_layout.addStretch(); nav_layout.addWidget(QLabel("LOCAL INFERENCE", objectName="sideCaption")); nav_layout.addWidget(QLabel("匿名结果才进入表情模型", objectName="sideNote")); layout.addWidget(nav); main = QFrame(); main_layout = QVBoxLayout(main); main_layout.setContentsMargins(28, 22, 28, 22); main_layout.setSpacing(0)
        header = QHBoxLayout(); header.addWidget(QLabel("座舱人脸去标识实验台", objectName="appTitle")); header.addStretch(); self.status = QLabel("●  CUDA / LOCAL", objectName="topStatus"); header.addWidget(self.status); main_layout.addLayout(header); main_layout.addSpacing(18); main_layout.addWidget(self.stack, 1); layout.addWidget(main, 1); self.select_page(0)

    def select_page(self, index):
        self.stack.setCurrentIndex(index)
        for i, button in enumerate(self.nav_buttons): button.setChecked(i == index)

    def closeEvent(self, event):
        self.pages[1].stop(); event.accept()


STYLESHEET = f"""
QMainWindow, QWidget {{ background: {BG}; color: {TEXT}; font-family: 'Segoe UI'; font-size: 10pt; }}
#sidebar {{ background: {SIDEBAR}; border-right: 1px solid {BORDER}; }}
#brand {{ color: {MINT}; font-size: 26pt; font-weight: 700; letter-spacing: 2px; line-height: 0.9; }}
#brandSub, #sideCaption {{ color: {FAINT}; font-size: 8pt; letter-spacing: 1px; }}
#sideNote {{ color: {MUTED}; font-size: 9pt; padding-top: 4px; }}
#appTitle {{ color: {TEXT}; font-size: 17pt; font-weight: 600; }}
#topStatus, #statusChip {{ color: {AQUA}; background: #163532; border: 1px solid #2b6d63; border-radius: 5px; padding: 7px 11px; font-size: 8pt; font-weight: 600; }}
QPushButton {{ background: {SURFACE_2}; color: {TEXT}; border: 1px solid {BORDER}; border-radius: 6px; padding: 9px 13px; font-weight: 600; }}
QPushButton:hover {{ background: {SURFACE_3}; border-color: {AQUA}; }}
QPushButton:pressed {{ background: #254c50; }}
QPushButton[nav="true"] {{ background: transparent; color: {MUTED}; border: 0; border-radius: 7px; text-align: left; padding: 12px 13px; font-weight: 500; }}
QPushButton[nav="true"]:hover {{ background: {SURFACE_2}; color: {TEXT}; }}
QPushButton[nav="true"]:checked {{ background: {AQUA}; color: #0a1718; font-weight: 700; }}
#primaryButton {{ background: {MINT}; color: #0d1c1e; border: 0; }}
#primaryButton:hover {{ background: #d6f7dc; }}
#pageTitle {{ color: {TEXT}; font-size: 25pt; font-weight: 650; }}
#pageSubtitle {{ color: {MUTED}; font-size: 10pt; }}
[panel="true"] {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 9px; }}
[mediaCard="true"] {{ background: #0b1518; border: 1px solid {BORDER}; border-radius: 7px; }}
#panelTitle {{ color: {TEXT}; font-size: 11pt; font-weight: 700; }}
#panelNote, #cardNote, #detailLine {{ color: {MUTED}; font-size: 9pt; }}
#mediaHeading {{ color: {TEXT}; font-size: 8pt; font-weight: 700; letter-spacing: 1px; }}
#mediaTag {{ color: {AQUA}; background: #173432; border-radius: 4px; padding: 4px 6px; font-size: 7pt; font-weight: 700; }}
#mediaImage, #videoImage {{ background: #070d0f; color: {FAINT}; border: 1px solid #203538; }}
#metricLine {{ color: {MINT}; font-size: 11pt; font-weight: 650; padding-top: 5px; }}
#cardTitle {{ color: {MUTED}; font-size: 8pt; letter-spacing: 1px; }}
#cardValue {{ color: {AQUA}; font-size: 23pt; font-weight: 700; }}
#cardValueMuted {{ color: {MINT}; font-size: 23pt; font-weight: 700; }}
QListWidget {{ background: transparent; border: 0; outline: 0; color: {MUTED}; }}
QListWidget::item {{ padding: 11px 8px; border-radius: 5px; margin: 2px 0; }}
QListWidget::item:hover {{ background: {SURFACE_2}; color: {TEXT}; }}
QListWidget::item:selected {{ background: #204b4b; color: {MINT}; }}
QComboBox {{ background: {SURFACE_2}; color: {TEXT}; border: 1px solid {BORDER}; border-radius: 5px; padding: 7px 9px; min-width: 190px; }}
QTableWidget {{ background: {SURFACE_2}; alternate-background-color: {SURFACE}; gridline-color: {BORDER}; border: 1px solid {BORDER}; color: {TEXT}; }}
QHeaderView::section {{ background: {SURFACE_3}; color: {AQUA}; border: 0; padding: 8px; font-weight: 700; }}
#pipelineStep {{ color: {MINT}; font-size: 10pt; font-weight: 650; }}
#pipelineArrow {{ color: {AQUA}; font-size: 15pt; padding: 0 12px; }}
QScrollBar:vertical {{ background: {SIDEBAR}; width: 10px; }} QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 5px; }}
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description="Qt Python 桌面座舱隐私实验台")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    app = QApplication([]); app.setApplicationName("Expression Lab"); window = MainWindow(args.device); window.show(); raise SystemExit(app.exec())


if __name__ == "__main__": main()
