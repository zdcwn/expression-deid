"""High-fidelity Qt desktop application for the expression privacy project.

This module is the local Python desktop entry point.  It uses PySide6 for a
product-like desktop shell while reusing the same Python model
and video workers.  The expression classifier receives only the anonymized
face in every realtime frame.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QSize, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFrame, QHBoxLayout, QLabel,
    QListView, QListWidget, QListWidgetItem, QMainWindow, QPushButton,
    QFileDialog, QScrollArea, QStackedWidget, QVBoxLayout, QWidget, QProgressBar,
)

from expression_deid.ui.opencv_demo import _image_frame, _resize_rgb, build_pipeline
from expression_deid.config import EXAMPLE_VIDEO_DIR, KMU_ALIGNED_DIR, ROOT
from expression_deid.runtime.pipeline import ExpressionDeidentifier
from expression_deid.runtime.video import FastVideoRunner
from expression_deid.runtime.expression import QualityExpressionPredictor
from expression_deid.labels import EXPRESSION_ZH
from expression_deid.ui.widgets import (
    AccuracyBars, ConfidenceMeter, FpsTrace, MediaCard, MetricCard,
    STYLESHEET, ensure_fonts, label, line_icon, page_heading,
)


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
    if hasattr(label, 'show_image'):
        label.show_image(image)
        return
    pixmap = QPixmap.fromImage(image)
    if max_size is None:
        max_size = label.size()
    if max_size.width() > 10 and max_size.height() > 10:
        pixmap = pixmap.scaled(max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    label.setPixmap(pixmap)
    label.setText("")


def samples() -> list[dict]:
    manifest = KMU_ALIGNED_DIR / "manifest.jsonl"
    # Register the complete valid gallery. Thumbnail loading is handled lazily
    # by the view, rather than hiding images behind a per-class sample cap.
    result = {}
    if not manifest.is_file():
        return []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        label = int(record.get("label", -1))
        relative = Path(str(record.get("output", "")))
        if relative.is_absolute() or relative.drive or '..' in relative.parts:
            continue
        # Windows may redirect a bundled file separately from its directory.
        # Keep the validated logical path instead of comparing physical paths.
        path = KMU_ALIGNED_DIR / relative
        if record.get('status') != 'ok' or not 0 <= label < 7:
            continue
        if not path.is_file():
            continue
        bucket = result.setdefault(label, [])
        bucket.append({"label": label, "expression": record.get("expression", str(label)), "path": path, "source": record.get("source", path.name), "person": record.get("person", "")})
    return [item for key in sorted(result) for item in result[key]]


_EXPERIMENT_PIPELINES: dict[str, ExpressionDeidentifier] = {}
_EXPERIMENT_PIPELINE_LOCK = threading.Lock()


def _experiment_pipeline(device: str) -> ExpressionDeidentifier:
    """Reuse the validated image pipeline across image selections.

    Preserve the selected GAN visual and expression checkpoints, but skip the
    identity gallery that this page does not display.  The separate diagnostic
    factory remains available to offline audit scripts.
    """
    with _EXPERIMENT_PIPELINE_LOCK:
        pipeline = _EXPERIMENT_PIPELINES.get(device)
        if pipeline is None:
            pipeline = build_pipeline(device, use_gan=True)
            _EXPERIMENT_PIPELINES[device] = pipeline
        return pipeline


def release_experiment_pipeline_cache():
    with _EXPERIMENT_PIPELINE_LOCK:
        pipelines = list(_EXPERIMENT_PIPELINES.values())
        _EXPERIMENT_PIPELINES.clear()
    for pipeline in pipelines:
        pipeline.close()


class ExperimentWorker(QThread):
    result_ready = Signal(object, int, int, float)
    error = Signal(str)
    finished_all = Signal(int)

    def __init__(self, items: list[dict], device: str):
        super().__init__(); self.items = items; self.device = device; self.completed = 0

    def run(self):
        try:
            if self.isInterruptionRequested(): return
            pipeline = _experiment_pipeline(self.device)
            for index, item in enumerate(self.items, start=1):
                if self.isInterruptionRequested(): return
                started = time.perf_counter()
                result = pipeline.process(item["path"], aligned=True, diagnostics=False)
                self.completed = index
                self.result_ready.emit({"item": item, "result": result}, index, len(self.items), time.perf_counter() - started)
            self.finished_all.emit(len(self.items))
        except Exception as exc:  # pragma: no cover - dependent on local model files
            logging.exception('Image analysis failed')
            self.error.emit(str(exc))


class ExperimentPage(QWidget):
    def __init__(self, device: str):
        super().__init__(); self.device = device; self.items = samples(); self.worker = None
        self.results = {}; self.rows_by_path = {}; self.loaded_thumbnails = set()
        self.stop_requested = False
        self.thumbnail_timer = QTimer(self); self.thumbnail_timer.setSingleShot(True)
        self.thumbnail_timer.timeout.connect(self._load_visible_thumbnails)
        self._build()

    def _build(self):
        root = QVBoxLayout(self); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(16)
        heading = QHBoxLayout(); heading.addLayout(page_heading('01 / IMAGE EXPERIMENT', '看见变化，保留表情。', '选择带标签的座舱样本，比较匿名效果与实际识别结果。'), 1)
        self.run_button = QPushButton('分析全部样本'); self.run_button.setObjectName('primaryButton')
        self.run_button.setIcon(line_icon('play', '#ffffff')); self.run_button.clicked.connect(self.run_all)
        heading.addWidget(self.run_button, 0, Qt.AlignBottom)
        self.stop_button = QPushButton('停止分析'); self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_analysis)
        heading.addWidget(self.stop_button, 0, Qt.AlignBottom); root.addLayout(heading)
        content = QHBoxLayout(); content.setSpacing(16); root.addLayout(content, 1)
        sidebar = QFrame(); sidebar.setProperty('panel', True); sidebar.setFixedWidth(198)
        left = QVBoxLayout(sidebar); left.setContentsMargins(12, 16, 12, 14); left.setSpacing(8)
        left.addWidget(label('示例图库', 'panelTitle'))
        self.gallery_count = label(f'{len(self.items):,} 张 · 完整图库', 'panelNote')
        left.addWidget(self.gallery_count)
        self.expression_filter = QComboBox()
        self.expression_filter.addItem('全部表情', None)
        for expression in sorted({item['label'] for item in self.items}):
            self.expression_filter.addItem(EXPRESSION_ZH[expression], expression)
        left.addWidget(self.expression_filter)
        self.sample_list = QListWidget(); self.sample_list.setObjectName("sampleList")
        self.sample_list.setIconSize(QSize(46, 46)); self.sample_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.sample_list.setUniformItemSizes(True)
        # Text rows are cheap; finish their geometry before selection/scrolling.
        # Batched layout can leave late rows unreachable just after filtering.
        self.sample_list.setLayoutMode(QListView.SinglePass)
        for index, item in enumerate(self.items, start=1):
            row = QListWidgetItem(f"{EXPRESSION_ZH[item['label']]}\n样本 {index:02d} · {item['person']}")
            row.setSizeHint(QSize(164, 70))
            row.setData(Qt.UserRole, item); self.sample_list.addItem(row)
            self.rows_by_path[str(item['path'])] = row
        left.addWidget(self.sample_list, 1)
        self.exp_progress = QProgressBar(); self.exp_progress.setTextVisible(False)
        self.exp_progress.setRange(0, max(1, len(self.items))); self.exp_progress.setValue(0); left.addWidget(self.exp_progress)
        self.exp_status = label('点击样本开始分析', 'panelNote', wrap=True); left.addWidget(self.exp_status)
        left.addWidget(label('示例用于演示，非独立测试集。', 'panelNote', wrap=True))
        content.addWidget(sidebar)
        media = QHBoxLayout(); media.setSpacing(14)
        self.original_view = MediaCard('原始画面', 'ORIGINAL / GRAYSCALE', '选择左侧样本').pair()
        self.anonymized_view = MediaCard('匿名化结果', 'ANONYMIZED / FACE EDIT', '等待匿名化处理', True).pair()
        media.addWidget(self.original_view[0], 1); media.addWidget(self.anonymized_view[0], 1); content.addLayout(media, 1)
        cards = QHBoxLayout(); cards.setSpacing(12)
        self.truth_card = MetricCard('真实标签', '—', '样本标注', accent=False)
        self.prediction_card = MetricCard('匿名图表情', '—', '匿名结果识别')
        self.confidence_card = MetricCard('预测置信度', '—', '模型输出概率')
        self.privacy_card = MetricCard('匿名处理', '—', '局部面部编辑已完成', accent=False)
        for card in (self.truth_card, self.prediction_card, self.confidence_card, self.privacy_card): cards.addWidget(card, 1)
        root.addLayout(cards)
        footer = QFrame(); footer.setProperty('panel', True); detail = QVBoxLayout(footer)
        detail.setContentsMargins(16, 10, 16, 10); detail.setSpacing(4)
        self.metric_line = label('对照结果将在分析完成后显示。', 'metricLine', wrap=True)
        self.detail_line = label('匿名化与表情结果将在分析完成后显示。', 'detailLine', wrap=True)
        detail.addWidget(self.metric_line); detail.addWidget(self.detail_line); root.addWidget(footer)
        self.sample_list.itemClicked.connect(self.run_selected)
        self.expression_filter.currentIndexChanged.connect(self._filter_samples)
        self.sample_list.verticalScrollBar().valueChanged.connect(lambda _: self.thumbnail_timer.start(30))
        self.sample_list.verticalScrollBar().rangeChanged.connect(lambda *_: self.thumbnail_timer.start(30))
        if self.items:
            self.sample_list.setCurrentRow(0); self._preview_source(self.items[0])

    def showEvent(self, event):
        super().showEvent(event); self.thumbnail_timer.start(0)

    def resizeEvent(self, event):
        super().resizeEvent(event); self.thumbnail_timer.start(30)

    def _load_visible_thumbnails(self):
        viewport = self.sample_list.viewport()
        first = max(0, self.sample_list.indexAt(QPoint(2, 2)).row())
        budget = max(2, viewport.height() // 70 + 3)
        visible = 0
        for index in range(first, self.sample_list.count()):
            row = self.sample_list.item(index)
            if row.isHidden():
                continue
            if visible >= budget:
                break
            visible += 1
            path = str(row.data(Qt.UserRole)['path'])
            if path not in self.loaded_thumbnails:
                pixmap = QPixmap(path)
                if not pixmap.isNull():
                    row.setIcon(QIcon(pixmap.scaled(46, 46, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
                    self.loaded_thumbnails.add(path)

    def filtered_samples(self):
        expression = self.expression_filter.currentData()
        return [item for item in self.items if expression is None or item['label'] == expression]

    def _filter_samples(self):
        expression = self.expression_filter.currentData()
        for item in self.items:
            self.rows_by_path[str(item['path'])].setHidden(expression is not None and item['label'] != expression)
        items = self.filtered_samples()
        self.gallery_count.setText(f'{len(items):,} / {len(self.items):,} 张 · KMU')
        self.run_button.setText('分析全部样本' if expression is None else '分析当前类别')
        if items:
            row = self.rows_by_path[str(items[0]['path'])]
            self.sample_list.setCurrentItem(row); self.sample_list.scrollToItem(row)
            self._preview_source(items[0])
            cached = self.results.get(str(items[0]['path']))
            if cached:
                self._show_result(*cached)
            else:
                self.anonymized_view[1].clear_image('点击样本开始分析')
                self.anonymized_view[0].set_caption('ANONYMIZED / FACE EDIT')
                for card in (self.prediction_card, self.confidence_card, self.privacy_card): card.set_value('—')
                self.metric_line.setText('对照结果将在分析完成后显示。')
                self.detail_line.setText('匿名化与表情结果将在分析完成后显示。')
        self.exp_progress.setRange(0, max(1, len(items)))
        self.exp_progress.setValue(sum(str(item['path']) in self.results for item in items))
        self.exp_status.setText(f'当前类别 {len(items):,} 张 · 点击样本开始分析')
        self.thumbnail_timer.start(0)

    def _preview_source(self, item):
        set_image(self.original_view[1], np.asarray(_image_frame(item['path'])))
        self.original_view[0].set_caption(item['path'].name)
        self.truth_card.set_value(EXPRESSION_ZH[item['label']])

    @Slot()
    def run_selected(self):
        row = self.sample_list.currentItem()
        if row:
            item = row.data(Qt.UserRole); cached = self.results.get(str(item['path']))
            if cached:
                self._show_result(*cached)
                items = self.filtered_samples()
                self.exp_progress.setRange(0,len(items)); self.exp_progress.setValue(sum(str(i['path']) in self.results for i in items))
                self.exp_status.setText(f'已分析 {len(self.results)} 张 · 正在查看所选样本')
            else: self._start([item])

    @Slot()
    def run_all(self): self._start(self.filtered_samples())

    def stop_analysis(self):
        if self.worker and self.worker.isRunning():
            self.stop_requested = True; self.worker.requestInterruption()
            self.stop_button.setEnabled(False)
            self.exp_status.setText('正在停止，完成当前图片后结束…')

    def _worker_finished(self):
        if self.sender() is not self.worker:
            return
        self.run_button.setEnabled(True); self.stop_button.setEnabled(False)
        self.expression_filter.setEnabled(True)
        if self.stop_requested:
            self.exp_status.setText(f'已停止 · 本轮完成 {self.worker.completed} 张，结果已保留')

    def _start(self, items):
        if self.worker and self.worker.isRunning(): return
        if not items:
            self.exp_status.setText('没有可用样本，请先准备 KMU 数据集。'); return
        self._preview_source(items[0]); self.anonymized_view[1].clear_image('正在加载模型…')
        for card in (self.prediction_card, self.confidence_card, self.privacy_card): card.set_value('—')
        self.metric_line.setText('正在分析匿名结果…'); self.detail_line.setText('匿名化与表情结果将在分析完成后显示。')
        self.exp_progress.setRange(0, len(items)); self.exp_progress.setValue(0)
        self.stop_requested = False
        self.run_button.setEnabled(False); self.stop_button.setEnabled(True)
        self.expression_filter.setEnabled(False); self.exp_status.setText("正在加载匿名化与表情模型…")
        self.worker = ExperimentWorker(items, self.device); self.worker.result_ready.connect(self._show_result); self.worker.error.connect(self._error); self.worker.finished_all.connect(self._finished)
        self.worker.finished.connect(self._worker_finished)
        self.worker.start()

    @Slot(object, int, int, float)
    def _show_result(self, payload, index, total, elapsed):
        item, result = payload["item"], payload["result"]
        self.results[str(item['path'])] = (payload, index, total, elapsed)
        row = self.rows_by_path.get(str(item['path']))
        if row is not None:
            self.sample_list.setCurrentItem(row)
        self._preview_source(item)
        set_image(self.anonymized_view[1], result["anonymized"], max_size=self.anonymized_view[1].size())
        exp = result.get("expression_anonymized", {})
        self.prediction_card.set_value(exp.get('name_zh', exp.get('name', '—')))
        self.confidence_card.set_value(_metric_value(exp.get('confidence'), percent=True))
        self.privacy_card.set_value('已完成')
        self.metric_line.setText(f"样本标签 {EXPRESSION_ZH[item['label']]}  →  匿名图预测 {exp.get('name_zh', exp.get('name', '—'))}     ·     局部匿名化已完成")
        self.detail_line.setText(f"匿名结果已送入表情模型     ·     本张 {elapsed * 1000:.0f} ms     ·     {index}/{total}")
        self.anonymized_view[0].set_caption(f"ANONYMIZED · {result['anonymized'].shape[1]} × {result['anonymized'].shape[0]}")
        self.exp_progress.setRange(0, total); self.exp_progress.setValue(index)
        self.exp_status.setText(f"已完成 {index}/{total} 张样本")

    def _error(self, message):
        self.run_button.setEnabled(True)
        self.exp_status.setText(f"实验失败：{message}")
    def _finished(self, count): self.run_button.setEnabled(True); self.exp_status.setText(f"完成 {count} 张样本的匿名化与表情分析")


class VideoWorker(QThread):
    frame_ready = Signal(object, object, str, float, float, int)
    worker_error = Signal(str)
    ended = Signal()

    def __init__(self, source, device: str, mode: str = "fast"):
        super().__init__(); self.source = source; self.device = device
        self.mode = mode if mode in {"fast", "quality"} else "fast"
        self.running = True
        self.frame_pending = threading.Event()

    def stop(self): self.running = False

    def run(self):
        capture = None
        pipeline = display_pipeline = quality_predictor = None
        try:
            # Both modes keep the privacy boundary: only the local anonymous
            # result reaches the expression classifier.  Fast mode avoids loading
            # the heavyweight GAN and quality checkpoint entirely.  Quality mode
            # opts into those slower display/temporal branches explicitly.
            pipeline = build_pipeline(self.device, use_gan=False)
            if not self.running: return
            if self.mode == "quality":
                display_pipeline = build_pipeline(self.device, use_gan=True)
                if not self.running: return
                quality_manifest = ROOT / 'models' / 'expression' / 'expression_video_quality.json'
                if quality_manifest.is_file() and (ROOT / 'models' / 'expression' / 'expression_video_quality.ts.pt').is_file():
                    quality_predictor = QualityExpressionPredictor(self.device, manifest=quality_manifest)
            else:
                # The fast branch uses the same local pipeline for both output
                # panes, avoiding a second model load and an unnecessary GAN.
                display_pipeline = pipeline
            runner = FastVideoRunner(
                pipeline,
                detect_interval=1 if quality_predictor is not None else 5,
                display_pipeline=display_pipeline,
                display_interval=15 if self.mode == "quality" else 1,
                expression_predictor=quality_predictor,
                track_reset_gap=2.5 if quality_predictor is not None else .75,
            )
            kind, value = self.source; capture = cv2.VideoCapture(int(value), cv2.CAP_DSHOW) if kind == "camera" else cv2.VideoCapture(str(value))
            if not capture.isOpened(): raise RuntimeError("无法打开摄像头或视频文件")
            source_fps = capture.get(cv2.CAP_PROP_FPS)
            if not np.isfinite(source_fps) or source_fps <= 0:
                source_fps = 30.0
            ticks = deque(maxlen=24); index = 0
            next_video_tick = time.perf_counter()
            while self.running:
                ok, bgr = capture.read()
                if not ok: break
                rgb = _resize_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), 960)
                timestamp = index / source_fps if kind != "camera" else time.monotonic()
                analysis_started = time.perf_counter()
                anonymized, info = runner.process_frame(rgb, index, timestamp=timestamp)
                ticks.append((time.perf_counter() - analysis_started, info.get('status') == 'ok'))
                fps = sum(done for _, done in ticks) / max(1e-6, sum(duration for duration, _ in ticks))
                label = info.get("expression_zh") or info.get("expression") or "未检测到人脸"
                confidence = float(info["confidence"]) if info.get("confidence") is not None else float("nan")
                if not self.frame_pending.is_set():
                    self.frame_pending.set()
                    self.frame_ready.emit(qimage_from_array(pipeline._image_gray(rgb)), qimage_from_array(anonymized), label, confidence, fps, index)
                index += 1
                # Example/local videos should be observable at their source
                # cadence.  Without this, the worker decodes as fast as the
                # model permits and a five-second clip appears to jump past.
                if kind != "camera":
                    next_video_tick += 1.0 / source_fps
                    delay = next_video_tick - time.perf_counter()
                    if delay > 0:
                        time.sleep(min(delay, 0.10))
                    elif delay < -1.0:
                        next_video_tick = time.perf_counter()
            self.ended.emit()
        except Exception as exc:
            logging.exception('Video analysis failed')
            self.worker_error.emit(str(exc))
        finally:
            if capture is not None: capture.release()
            if display_pipeline is not None and display_pipeline is not pipeline: display_pipeline.close()
            if pipeline is not None: pipeline.close()


class VideoPage(QWidget):
    def __init__(self, device: str):
        super().__init__(); self.device = device; self.worker = None
        self.pending_source = None; self.accept_frames = False; self._build()

    def _build(self):
        root = QVBoxLayout(self); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(16)
        root.addLayout(page_heading('02 / LIVE ANALYSIS', '每一帧，都有对照。', '接入摄像头或视频，同时查看灰度输入、匿名结果与连续表情分析。'))
        controls = QFrame(); controls.setProperty('panel', True)
        bar = QHBoxLayout(controls); bar.setContentsMargins(14, 12, 14, 12); bar.setSpacing(9)
        self.camera_button = QPushButton('摄像头'); self.camera_button.setObjectName('primaryButton')
        self.camera_button.setIcon(line_icon('video', '#ffffff')); self.camera_button.clicked.connect(self.start_camera); bar.addWidget(self.camera_button)
        choose = QPushButton('本地视频'); choose.setIcon(line_icon('folder')); choose.clicked.connect(self.choose_video); bar.addWidget(choose)
        bar.addWidget(label('运行模式', 'panelNote'))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem('高速模式 · 实时优先', 'fast')
        self.mode_combo.addItem('质量模式 · 准确优先', 'quality')
        self.mode_combo.setToolTip('高速模式不加载 GAN 与质量模型；质量模式启用完整质量分支，启动更慢。')
        bar.addWidget(self.mode_combo)
        self.examples = QComboBox()
        titles = {'example_kmu_expressions.mp4':'六类表情切换','example_mixed.mp4':'混合场景','example_anger.mp4':'愤怒示例','example_surprise.mp4':'惊讶示例'}
        for path in sorted(EXAMPLE_VIDEO_DIR.glob('example*.mp4')):
            self.examples.addItem(titles.get(path.name,path.stem),path.name)
        bar.addWidget(self.examples, 1)
        play = QPushButton('播放示例'); play.setIcon(line_icon('play')); play.clicked.connect(self.play_example); bar.addWidget(play)
        self.stop_button = QPushButton('停止'); self.stop_button.setObjectName('dangerButton'); self.stop_button.setIcon(line_icon('stop','#a95750')); self.stop_button.clicked.connect(self.stop); bar.addWidget(self.stop_button)
        root.addWidget(controls)
        media = QHBoxLayout(); media.setSpacing(16)
        self.original = MediaCard('原始视频', 'SOURCE / GRAYSCALE', '打开摄像头或选择视频').pair()
        self.anonymized = MediaCard('匿名视频', 'ANONYMIZED / LIVE', '分析开始后显示匿名结果', True).pair()
        media.addWidget(self.original[0],1); media.addWidget(self.anonymized[0],1); root.addLayout(media,1)
        metrics = QHBoxLayout(); metrics.setSpacing(12)
        self.expression=MetricCard('当前表情','—','匿名图时序分析')
        self.confidence=MetricCard('置信度','—','连续匿名帧概率')
        self.confidence_meter=ConfidenceMeter(); self.confidence.layout().addWidget(self.confidence_meter)
        self.fps=MetricCard('模型分析帧率','—','匿名化 + 表情识别')
        self.frame_card=MetricCard('当前帧','—','已处理输入帧编号',accent=False)
        for card in (self.expression,self.confidence,self.fps,self.frame_card): metrics.addWidget(card,1)
        root.addLayout(metrics)
        lower=QFrame(); lower.setProperty('panel',True); lower_layout=QHBoxLayout(lower); lower_layout.setContentsMargins(16,10,16,10); lower_layout.setSpacing(22)
        summary=QVBoxLayout(); summary.setSpacing(5)
        self.status=label('等待输入','statusChip'); summary.addWidget(self.status)
        self.source_label=label('未选择来源','panelNote'); self.source_label.setMaximumWidth(300); summary.addWidget(self.source_label)
        lower_layout.addLayout(summary)
        trace=QVBoxLayout(); caption=QHBoxLayout(); caption.addWidget(label('分析帧率走势','panelNote')); caption.addStretch(); caption.addWidget(label('虚线：24 FPS 目标','panelNote')); trace.addLayout(caption)
        self.fps_trace=FpsTrace(); trace.addWidget(self.fps_trace); lower_layout.addLayout(trace,1); root.addWidget(lower)

    def _start(self, source, text):
        self.pending_source = (source, text)
        if self.worker is not None:
            self.accept_frames = False
            self.worker.stop()
            self.status.setText('正在停止上一段分析…')
            return
        self.pending_source = None
        mode_text = self.mode_combo.currentText()
        self.source_label.setText(f'{text} · {mode_text}'); self.status.setText(f"正在加载{mode_text}…")
        self.source_label.setToolTip(text); self.fps_trace.reset(); self.confidence_meter.set_value(None)
        for card in (self.expression,self.confidence,self.fps,self.frame_card): card.set_value('—')
        self.original[1].clear_image('正在打开输入…'); self.anonymized[1].clear_image('正在加载分析模型…')
        # Keep the worker constructor easy to replace in runtime checks while
        # still applying the selected mode before the thread starts.
        self.worker = VideoWorker(source, self.device)
        if hasattr(self.worker, 'mode'):
            self.worker.mode = self.mode_combo.currentData()
        self.worker.setParent(self)
        self.accept_frames = True
        self.worker.frame_ready.connect(self._frame); self.worker.worker_error.connect(self._error)
        self.worker.ended.connect(self._ended); self.worker.finished.connect(self._worker_finished)
        self.worker.start()

    def _ended(self):
        if self.sender() is self.worker and self.accept_frames:
            self.status.setText('视频结束')

    def _worker_finished(self):
        worker = self.sender()
        if worker is not self.worker: return
        self.worker = None
        worker.deleteLater()
        pending, self.pending_source = self.pending_source, None
        if pending is not None:
            self._start(*pending)
        elif not self.accept_frames:
            self.status.setText('已停止')

    def start_camera(self): self._start(("camera", 0), "本机摄像头")
    def choose_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", str(EXAMPLE_VIDEO_DIR), "视频 (*.mp4 *.avi *.mov *.mkv *.webm)")
        if path: self._start(("video", path), Path(path).name)
    def play_example(self):
        if self.examples.currentData(): self._start(("video", str(EXAMPLE_VIDEO_DIR / self.examples.currentData())), f"示例 · {self.examples.currentText()}")
    def stop(self):
        self.pending_source = None; self.accept_frames = False
        if self.worker is not None:
            self.worker.stop()
            self.status.setText('正在停止…')
        else:
            self.status.setText("已停止")

    @Slot(object, object, str, float, float, int)
    def _frame(self, original, anonymized, expression, confidence, fps, index):
        sender = self.sender()
        if sender is not None:
            sender.frame_pending.clear()
            if sender is not self.worker or not self.accept_frames: return
        for image, label in ((original, self.original[1]), (anonymized, self.anonymized[1])):
            label.show_image(image)
        self.expression.set_value(expression); self.confidence.set_value(f"{confidence * 100:.1f}%" if np.isfinite(confidence) else "—")
        self.fps.set_value(f"{fps:.1f}"); self.fps.note_label.setText('FPS · 最近 24 帧模型分析')
        self.frame_card.set_value(f'{index+1:05d}'); self.confidence_meter.set_value(confidence); self.fps_trace.add_value(fps)
        self.original[0].set_caption(f'SOURCE · {original.width()} × {original.height()} · FRAME {index+1:05d}')
        self.anonymized[0].set_caption('ANONYMIZED · LIVE ANALYSIS')
        self.anonymized[1].set_overlay(expression, f'置信度 {confidence*100:.1f}%' if np.isfinite(confidence) else '等待持续有效证据')
        self.status.setText('分析中' if np.isfinite(confidence) else '分析中 · 暂无确定表情')

    def _error(self, message):
        if self.sender() is self.worker and self.accept_frames:
            self.status.setText(f"分析失败：{message}")


def _metric_value(value, *, percent=False, decimals=1):
    if value is None or not np.isfinite(value): return '—'
    return f'{value * 100:.1f}%' if percent else f'{value:.{decimals}f}'


class PerformancePage(QWidget):
    """Show dataset-scoped audited values, never mix them into one accuracy."""
    def __init__(self, device: str):
        super().__init__(); self.device=device
        self.audit_path=ROOT/'docs/model-metrics.json'
        try: self.audit=json.loads(self.audit_path.read_text(encoding='utf-8'))
        except (OSError,ValueError): self.audit={}
        comparisons=self.audit.get('expression_comparison',[])
        self.current=comparisons[-1].get('domains',{}) if comparisons else {}
        self.previous=comparisons[0].get('domains',{}) if comparisons else {}
        self._build()

    def _build(self):
        root=QVBoxLayout(self); root.setContentsMargins(0,0,0,0)
        scroll=QScrollArea(); scroll.setWidgetResizable(True); root.addWidget(scroll)
        body=QWidget(); scroll.setWidget(body); layout=QVBoxLayout(body); layout.setContentsMargins(0,0,8,0); layout.setSpacing(16)
        layout.addLayout(page_heading('03 / MODEL INSIGHTS','用数据，理解表现。','每个数据集单独呈现；示例效果、预测置信度与验证准确率分别记录。'))
        controls=QHBoxLayout(); controls.addWidget(label('验证数据集','panelTitle')); controls.addStretch()
        self.scope=QComboBox()
        for name,key in [('RAF · 自然场景七类','valid_raf'),('KMU · 座舱六类尾帧','valid_kmu'),('PPB · 中性验证集','valid_ppb'),('PPB · 中性测试集','test_ppb')]:
            self.scope.addItem(name,key)
        controls.addWidget(self.scope); controls.addWidget(label(self.audit.get('date','未验证'),'statusChip')); layout.addLayout(controls)
        cards=QHBoxLayout(); cards.setSpacing(12)
        self.accuracy=MetricCard('匿名图验证准确率','—','所选数据集')
        self.sample_count=MetricCard('验证样本','—','不包含训练样本',accent=False)
        privacy=self.audit.get('privacy',{})
        self.privacy=MetricCard('匿名化命中率',_metric_value(privacy.get('anonymization_identity_match_rate'),percent=True),'ratio_v2 强几何 · 仅匿名化 · 越低越好',accent=False)
        self.privacy.value_label.setStyleSheet('color: #ad6157;')
        self.speed=MetricCard('稳态分析帧率','—','离线视频实测范围',accent=False)
        values=[row.get('steady_analysis_fps') for row in self.audit.get('videos',[]) if row.get('steady_analysis_fps') is not None]
        if values: self.speed.set_value(f'{min(values):.1f}–{max(values):.1f}'); self.speed.note_label.setText('FPS · 非每帧保证')
        for card in (self.accuracy,self.sample_count,self.privacy,self.speed): cards.addWidget(card,1)
        layout.addLayout(cards)
        chart_row=QHBoxLayout(); chart_row.setSpacing(16)
        chart=QFrame(); chart.setProperty('panel',True); cl=QVBoxLayout(chart); cl.setContentsMargins(20,18,20,18)
        chart_head=QHBoxLayout(); chart_head.addWidget(label('各表情识别准确率','panelTitle')); chart_head.addStretch(); chart_head.addWidget(label('准确率 / 样本数','panelNote')); cl.addLayout(chart_head)
        self.bars=AccuracyBars(); cl.addWidget(self.bars,1); chart_row.addWidget(chart,3)
        side=QFrame(); side.setProperty('panel',True); sl=QVBoxLayout(side); sl.setContentsMargins(20,18,20,18); sl.setSpacing(12)
        sl.addWidget(label('验证范围','panelTitle'))
        self.scope_note=label('','panelNote',wrap=True); sl.addWidget(self.scope_note)
        sl.addStretch(); sl.addWidget(label('匿名化效果检查','panelTitle'))
        sl.addWidget(label('公开指标采用 ratio_v2 强几何匿名化路径；表情结果采用质量优先分类器。','panelNote',wrap=True))
        sl.addWidget(label(f"{privacy.get('samples','—')} 个查询 · {privacy.get('gallery_size','—')} 人库",'panelNote',wrap=True))
        sl.addWidget(label('模型分析帧率包含匿名化与表情识别；生成刷新仍有耗时峰值。','panelNote',wrap=True))
        chart_row.addWidget(side,2); layout.addLayout(chart_row)
        notice=label('当前结果为 2026-10-09 保存的验证记录；本人摄像头准确率与真实道路场景尚未验收。','notice',wrap=True); layout.addWidget(notice)
        chain=QFrame(); chain.setProperty('panel',True); steps=QHBoxLayout(chain); steps.setContentsMargins(18,14,18,14)
        for i,text in enumerate(('01  输入灰度化','02  面部局部匿名化','03  匿名图表情分析')):
            steps.addWidget(label(text,'pipelineStep'),1)
            if i<2: steps.addWidget(label('→','pipelineArrow'))
        layout.addWidget(chain)
        self.provenance=label('记录来源：docs/model-metrics.json · 静态准确率使用 FP32','panelNote',wrap=True)
        self.provenance.setTextInteractionFlags(Qt.TextSelectableByMouse); layout.addWidget(self.provenance)
        layout.addStretch()
        self.scope.currentIndexChanged.connect(self._update_scope); self._update_scope()

    def _update_scope(self):
        key=self.scope.currentData(); domain=self.current.get(key,{})
        self.accuracy.set_value(_metric_value(domain.get('accuracy'),percent=True))
        self.sample_count.set_value(f"{domain['samples']:,}" if 'samples' in domain else '—')
        self.sample_count.note_label.setText({'valid_raf':'自然场景 · 七类表情','valid_kmu':'尾帧回归 · 非未见身份','valid_ppb':'7 人 · 仅中性','test_ppb':'另 7 人 · 仅中性'}.get(key,'尚未验证'))
        previous=self.previous.get(key,{}).get('accuracy'); accuracy=domain.get('accuracy')
        if accuracy is not None and previous is not None:
            delta=(accuracy-previous)*100
            self.accuracy.note_label.setText(f'同输入较旧模型 {delta:+.2f} 个百分点')
        else: self.accuracy.note_label.setText('该数据集尚未验证')
        rows=[]
        for k,row in sorted(domain.get('by_class',{}).items(),key=lambda v:int(v[0])):
            index=int(k)
            if 0<=index<len(EXPRESSION_ZH): rows.append((EXPRESSION_ZH[index],row.get('samples',0),row.get('accuracy')))
        self.bars.set_rows(rows)
        notes={
            'valid_raf':'RAF-DB\n3068 张自然场景验证图，覆盖七类表情。\n\n原图经过当前匿名链路后，再进入分类器。',
            'valid_kmu':'KMU\n220 张序列尾帧，覆盖六类表情，不含中性。\n\n与训练数据存在同人相邻帧；不能代表陌生人泛化。',
            'valid_ppb':'PPB-Emo\n34 张近红外中性帧，来自 7 人。\n\n只验证中性，不能解释为七类准确率。',
            'test_ppb':'PPB-Emo\n35 张近红外中性帧，来自另外 7 人。\n\n与训练、验证人员分开；只包含中性。',
        }
        self.scope_note.setText(notes.get(key,'暂无验证范围说明') if domain else '没有找到模型指标摘要，请检查 docs/model-metrics.json。')


class MainWindow(QMainWindow):
    def __init__(self, device: str):
        super().__init__(); ensure_fonts(); self.device = device
        self.setWindowTitle('Expression Lab · 座舱隐私实验台'); self.setWindowIcon(line_icon('face'))
        self.resize(1460, 940); self.setMinimumSize(760, 520); self.setStyleSheet(STYLESHEET)
        if self.screen() is not None:
            available=self.screen().availableGeometry()
            self.resize(min(1460,max(760,available.width()-48)),min(940,max(520,available.height()-56)))
            self.move(available.left()+max(0,(available.width()-self.width())//2),available.top()+max(0,(available.height()-self.height())//2))
        shell = QWidget(); layout = QHBoxLayout(shell); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(0); self.setCentralWidget(shell)
        nav = QFrame(); nav.setObjectName('sidebar'); nav.setFixedWidth(202)
        nav_layout=QVBoxLayout(nav); nav_layout.setContentsMargins(18,28,18,24); nav_layout.setSpacing(8)
        logo=QLabel(); logo.setPixmap(line_icon('face','#a9debd',34).pixmap(34,34)); nav_layout.addWidget(logo)
        nav_layout.addWidget(label('expression.','brand')); nav_layout.addWidget(label('座舱隐私实验台','brandSub'))
        nav_layout.addSpacing(36); nav_layout.addWidget(label('WORKSPACE','navSection')); nav_layout.addSpacing(4)
        self.stack = QStackedWidget(); self.pages = [ExperimentPage(device), VideoPage(device), PerformancePage(device)]
        # Keep controls reachable on small/high-DPI displays without clipping.
        for page in self.pages:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.setWidget(page)
            self.stack.addWidget(scroll)
        self.nav_buttons = []
        self.nav_icons=('image','video','chart')
        for index,text in enumerate(('图片实验','实时分析','模型表现')):
            button=QPushButton('  '+text); button.setIconSize(QSize(19,19)); button.setProperty('nav',True)
            button.setCheckable(True); button.clicked.connect(lambda checked,i=index:self.select_page(i))
            nav_layout.addWidget(button); self.nav_buttons.append(button)
        nav_layout.addStretch(); nav_layout.addWidget(label('PRIVATE BY DESIGN','sideCaption'))
        nav_layout.addWidget(label('本地推理 · 匿名图分析','sideNote'))
        nav_layout.addSpacing(10); nav_layout.addWidget(label('EXPRESSION LAB  /  DESKTOP','brandSub'))
        layout.addWidget(nav)
        main=QFrame(); main_layout=QVBoxLayout(main); main_layout.setContentsMargins(24,18,24,22); main_layout.setSpacing(0)
        header_frame=QFrame(); header_frame.setObjectName('workspaceHeader'); header=QHBoxLayout(header_frame)
        header.setContentsMargins(0,0,0,14); self.breadcrumb=label('工作台 / 图片实验','appTitle'); header.addWidget(self.breadcrumb); header.addStretch()
        device_text='自动选择计算设备' if device=='auto' else 'CUDA · GPU 加速' if device=='cuda' else 'CPU · 本地计算'
        self.status=label(device_text,'topStatus'); header.addWidget(self.status)
        main_layout.addWidget(header_frame); main_layout.addSpacing(22); main_layout.addWidget(self.stack,1)
        layout.addWidget(main,1); self.select_page(0)

    def select_page(self, index):
        self.stack.setCurrentIndex(index)
        for i, button in enumerate(self.nav_buttons):
            button.setChecked(i == index); button.setIcon(line_icon(self.nav_icons[i], '#2b6350' if i==index else '#8ca6b2'))
        self.breadcrumb.setText('工作台 / '+('图片实验','实时分析','模型表现')[index])

    def closeEvent(self, event):
        self.pages[1].stop()
        experiment = self.pages[0].worker
        if experiment is not None: experiment.requestInterruption()
        workers = (experiment, self.pages[1].worker)
        if any(worker is not None and worker.isRunning() for worker in workers):
            event.ignore()
            self.status.setText('正在停止分析并释放模型…')
            QTimer.singleShot(100, self.close)
        else:
            release_experiment_pipeline_cache()
            event.accept()



def main(argv=None):
    parser = argparse.ArgumentParser(description="Qt Python 桌面座舱隐私实验台")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    app = QApplication([]); app.setApplicationName("Expression Lab"); window = MainWindow(args.device); window.show(); raise SystemExit(app.exec())


if __name__ == "__main__": main()
