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
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QSize, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFrame, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QMainWindow, QPushButton,
    QFileDialog, QScrollArea, QStackedWidget, QVBoxLayout, QWidget, QProgressBar,
)

from expression_deid.app import _image_frame, _pick_checkpoint, _resize_rgb, build_pipeline
from expression_deid.config import KMU_ALIGNED_DIR, ROOT
from expression_deid.inference import ExpressionDeidentifier
from expression_deid.video_demo import FastVideoRunner
from expression_deid.labels import EXPRESSION_ZH
from expression_deid.desktop_design import (
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
    result = {}
    if not manifest.is_file():
        return []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        label = int(record.get("label", -1))
        path = (KMU_ALIGNED_DIR / str(record.get("output", ""))).resolve()
        if record.get('status') == 'ok' and 0 <= label < 7 and label not in result and path.is_file() and path.is_relative_to(KMU_ALIGNED_DIR.resolve()):
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
        ganonymization_checkpoint=gan_checkpoint if gan_checkpoint.is_file() and gan_repo.is_dir() else None,
        ganonymization_repo=gan_repo if gan_repo.is_dir() else None,
    )


class ExperimentWorker(QThread):
    result_ready = Signal(object, int, int, float)
    error = Signal(str)
    finished_all = Signal(int)

    def __init__(self, items: list[dict], device: str):
        super().__init__(); self.items = items; self.device = device

    def run(self):
        pipeline = None
        try:
            if self.isInterruptionRequested(): return
            pipeline = diagnostic_pipeline(self.device)
            for index, item in enumerate(self.items, start=1):
                if self.isInterruptionRequested(): return
                started = time.perf_counter()
                result = pipeline.process(item["path"], aligned=True, diagnostics=True)
                self.result_ready.emit({"item": item, "result": result}, index, len(self.items), time.perf_counter() - started)
            self.finished_all.emit(len(self.items))
        except Exception as exc:  # pragma: no cover - dependent on local model files
            self.error.emit(str(exc))
        finally:
            if pipeline is not None: pipeline.close()


class ExperimentPage(QWidget):
    def __init__(self, device: str):
        super().__init__(); self.device = device; self.items = samples(); self.worker = None
        self.results = {}; self._build()

    def _build(self):
        root = QVBoxLayout(self); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(16)
        heading = QHBoxLayout(); heading.addLayout(page_heading('01 / IMAGE EXPERIMENT', '看见变化，保留表情。', '选择带标签的座舱样本，比较匿名效果与实际识别结果。'), 1)
        self.run_button = QPushButton('分析全部样本'); self.run_button.setObjectName('primaryButton')
        self.run_button.setIcon(line_icon('play', '#ffffff')); self.run_button.clicked.connect(self.run_all)
        heading.addWidget(self.run_button, 0, Qt.AlignBottom); root.addLayout(heading)
        content = QHBoxLayout(); content.setSpacing(16); root.addLayout(content, 1)
        sidebar = QFrame(); sidebar.setProperty('panel', True); sidebar.setFixedWidth(198)
        left = QVBoxLayout(sidebar); left.setContentsMargins(12, 16, 12, 14); left.setSpacing(8)
        left.addWidget(label('示例图库', 'panelTitle'))
        left.addWidget(label(f'{len(self.items):02d} 个样本 · KMU', 'panelNote'))
        self.sample_list = QListWidget(); self.sample_list.setObjectName("sampleList")
        self.sample_list.setIconSize(QSize(46, 46)); self.sample_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        for index, item in enumerate(self.items, start=1):
            row = QListWidgetItem(f"{EXPRESSION_ZH[item['label']]}\n样本 {index:02d} · {item['person']}")
            row.setIcon(QIcon(str(item['path']))); row.setSizeHint(QSize(164, 70))
            row.setData(Qt.UserRole, item); self.sample_list.addItem(row)
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
        self.identity_card = MetricCard('原身份命中', '—', '独立 SFace 人脸比对', accent=False)
        for card in (self.truth_card, self.prediction_card, self.confidence_card, self.identity_card): cards.addWidget(card, 1)
        root.addLayout(cards)
        footer = QFrame(); footer.setProperty('panel', True); detail = QVBoxLayout(footer)
        detail.setContentsMargins(16, 10, 16, 10); detail.setSpacing(4)
        self.metric_line = label('对照结果将在分析完成后显示。', 'metricLine', wrap=True)
        self.detail_line = label('身份诊断：待分析', 'detailLine', wrap=True)
        detail.addWidget(self.metric_line); detail.addWidget(self.detail_line); root.addWidget(footer)
        self.sample_list.itemClicked.connect(self.run_selected)
        if self.items:
            self.sample_list.setCurrentRow(0); self._preview_source(self.items[0])

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
                self.exp_progress.setRange(0,len(self.items)); self.exp_progress.setValue(len(self.results))
                self.exp_status.setText(f'已分析 {len(self.results)} 张 · 正在查看所选样本')
            else: self._start([item])

    @Slot()
    def run_all(self): self._start(self.items)

    def _start(self, items):
        if self.worker and self.worker.isRunning(): return
        if not items:
            self.exp_status.setText('没有可用样本，请先准备 KMU 数据集。'); return
        self._preview_source(items[0]); self.anonymized_view[1].clear_image('正在加载模型…')
        for card in (self.prediction_card, self.confidence_card, self.identity_card): card.set_value('—')
        self.metric_line.setText('正在分析匿名结果…'); self.detail_line.setText('身份诊断：待分析')
        self.exp_progress.setRange(0, len(items)); self.exp_progress.setValue(0)
        self.run_button.setEnabled(False); self.exp_status.setText("正在加载身份诊断模型…")
        self.worker = ExperimentWorker(items, self.device); self.worker.result_ready.connect(self._show_result); self.worker.error.connect(self._error); self.worker.finished_all.connect(self._finished)
        self.worker.finished.connect(lambda: self.run_button.setEnabled(True))
        self.worker.start()

    @Slot(object, int, int, float)
    def _show_result(self, payload, index, total, elapsed):
        item, result = payload["item"], payload["result"]
        self.results[str(item['path'])] = (payload, index, total, elapsed)
        for row_index in range(self.sample_list.count()):
            row = self.sample_list.item(row_index)
            if row.data(Qt.UserRole)['path'] == item['path']:
                self.sample_list.setCurrentRow(row_index); break
        self._preview_source(item)
        set_image(self.anonymized_view[1], result["anonymized"], max_size=self.anonymized_view[1].size())
        exp = result.get("expression_anonymized", {}); changed = result.get("identity_prediction_changed")
        identity_text = '未验证' if changed is None else '已改变' if changed else '未改变'
        gallery = result.get('gallery_match_original')
        gallery_text = '未验证' if gallery is None else '是' if gallery else '否'
        self.prediction_card.set_value(exp.get('name_zh', exp.get('name', '—')))
        self.confidence_card.set_value(_metric_value(exp.get('confidence'), percent=True))
        self.identity_card.set_value(gallery_text)
        self.metric_line.setText(f"样本标签 {EXPRESSION_ZH[item['label']]}  →  匿名图预测 {exp.get('name_zh', exp.get('name', '—'))}     ·     闭集身份预测：{identity_text}")
        self.detail_line.setText(f"SFace {_metric_value(result.get('sface_cosine'), decimals=3)}     ·     {result.get('gallery_size', '—')} 人库原身份命中：{gallery_text}     ·     本张 {elapsed * 1000:.0f} ms     ·     {index}/{total}")
        self.anonymized_view[0].set_caption(f"ANONYMIZED · {result['anonymized'].shape[1]} × {result['anonymized'].shape[0]}")
        self.exp_progress.setRange(0, total); self.exp_progress.setValue(index)
        self.exp_status.setText(f"已完成 {index}/{total} 张样本")

    def _error(self, message):
        self.run_button.setEnabled(True)
        self.exp_status.setText(f"实验失败：{message}")
    def _finished(self, count): self.run_button.setEnabled(True); self.exp_status.setText(f"完成 {count} 张样本的身份诊断")


class VideoWorker(QThread):
    frame_ready = Signal(object, object, str, float, float, int)
    worker_error = Signal(str)
    ended = Signal()

    def __init__(self, source, device: str):
        super().__init__(); self.source = source; self.device = device; self.running = True
        self.frame_pending = threading.Event()

    def stop(self): self.running = False

    def run(self):
        capture = None
        pipeline = display_pipeline = None
        try:
            # Match the web visual output while keeping the realtime analysis
            # path responsive: the classifier pipeline stays on the fast local
            # edit, and the human-facing GAN is refreshed every fifteen frames for
            # display only.  FastVideoRunner never sends that display branch to
            # the expression classifier.
            pipeline = build_pipeline(self.device, use_gan=False)
            if not self.running: return
            display_pipeline = build_pipeline(self.device, use_gan=True)
            if not self.running: return
            runner = FastVideoRunner(
                pipeline,
                detect_interval=5,
                display_pipeline=display_pipeline,
                display_interval=15,
            )
            kind, value = self.source; capture = cv2.VideoCapture(int(value), cv2.CAP_DSHOW) if kind == "camera" else cv2.VideoCapture(str(value))
            if not capture.isOpened(): raise RuntimeError("无法打开摄像头或视频文件")
            source_fps = capture.get(cv2.CAP_PROP_FPS)
            if not np.isfinite(source_fps) or source_fps <= 0:
                source_fps = 30.0
            ticks = deque(maxlen=24); index = 0
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
            self.ended.emit()
        except Exception as exc: self.worker_error.emit(str(exc))
        finally:
            if capture is not None: capture.release()
            if display_pipeline is not None: display_pipeline.close()
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
        self.examples = QComboBox()
        titles = {'example_kmu_expressions.mp4':'六类表情切换','example_mixed.mp4':'混合场景','example_anger.mp4':'愤怒示例','example_surprise.mp4':'惊讶示例'}
        for path in sorted((ROOT/'outputs').glob('example*.mp4')):
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
        self.source_label.setText(text); self.status.setText("正在加载分析模型…")
        self.source_label.setToolTip(text); self.fps_trace.reset(); self.confidence_meter.set_value(None)
        for card in (self.expression,self.confidence,self.fps,self.frame_card): card.set_value('—')
        self.original[1].clear_image('正在打开输入…'); self.anonymized[1].clear_image('正在加载分析模型…')
        self.worker = VideoWorker(source, self.device); self.worker.setParent(self)
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
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", str(ROOT / "outputs"), "视频 (*.mp4 *.avi *.mov *.mkv *.webm)")
        if path: self._start(("video", path), Path(path).name)
    def play_example(self):
        if self.examples.currentData(): self._start(("video", str(ROOT / "outputs" / self.examples.currentData())), f"示例 · {self.examples.currentText()}")
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
        self.audit_path=ROOT/'docs/audit-results-20261009.json'
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
        privacy=self.audit.get('privacy',{}).get('fast',{})
        self.privacy=MetricCard('原身份命中率',_metric_value(privacy.get('sface_source_match_rate'),percent=True),'快速视频 · SFace · 越低越好',accent=False)
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
        sl.addStretch(); sl.addWidget(label('身份保护检查','panelTitle'))
        sl.addWidget(label('当前快速视频尚未通过隐私验收。','panelNote',wrap=True))
        sl.addWidget(label(f"{privacy.get('samples','—')} 个查询 · {privacy.get('gallery_size','—')} 人库",'panelNote',wrap=True))
        sl.addWidget(label('模型分析帧率包含匿名化与表情识别；生成刷新仍有耗时峰值。','panelNote',wrap=True))
        chart_row.addWidget(side,2); layout.addLayout(chart_row)
        notice=label('当前结果为 2026-10-09 保存的验证记录；本人摄像头准确率与真实道路场景尚未验收。','notice',wrap=True); layout.addWidget(notice)
        chain=QFrame(); chain.setProperty('panel',True); steps=QHBoxLayout(chain); steps.setContentsMargins(18,14,18,14)
        for i,text in enumerate(('01  输入灰度化','02  面部局部匿名化','03  匿名图表情分析')):
            steps.addWidget(label(text,'pipelineStep'),1)
            if i<2: steps.addWidget(label('→','pipelineArrow'))
        layout.addWidget(chain)
        self.provenance=label('记录来源：docs/audit-results-20261009.json · 静态准确率使用 FP32','panelNote',wrap=True)
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
        self.scope_note.setText(notes.get(key,'暂无验证范围说明') if domain else '没有找到验证报告，请检查 docs/audit-results-20261009.json。')


class MainWindow(QMainWindow):
    def __init__(self, device: str):
        super().__init__(); ensure_fonts(); self.device = device
        self.setWindowTitle('Expression Lab · 座舱隐私实验台'); self.setWindowIcon(line_icon('face'))
        self.resize(1460, 940); self.setMinimumSize(1100, 760); self.setStyleSheet(STYLESHEET)
        if self.screen() is not None:
            available=self.screen().availableGeometry()
            self.resize(min(1460,max(1100,available.width()-48)),min(940,max(760,available.height()-56)))
            self.move(available.left()+max(0,(available.width()-self.width())//2),available.top()+max(0,(available.height()-self.height())//2))
        shell = QWidget(); layout = QHBoxLayout(shell); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(0); self.setCentralWidget(shell)
        nav = QFrame(); nav.setObjectName('sidebar'); nav.setFixedWidth(202)
        nav_layout=QVBoxLayout(nav); nav_layout.setContentsMargins(18,28,18,24); nav_layout.setSpacing(8)
        logo=QLabel(); logo.setPixmap(line_icon('face','#a9debd',34).pixmap(34,34)); nav_layout.addWidget(logo)
        nav_layout.addWidget(label('expression.','brand')); nav_layout.addWidget(label('座舱隐私实验台','brandSub'))
        nav_layout.addSpacing(36); nav_layout.addWidget(label('WORKSPACE','navSection')); nav_layout.addSpacing(4)
        self.stack = QStackedWidget(); self.pages = [ExperimentPage(device), VideoPage(device), PerformancePage(device)]
        for page in self.pages: self.stack.addWidget(page)
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
            event.accept()



def main(argv=None):
    parser = argparse.ArgumentParser(description="Qt Python 桌面座舱隐私实验台")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    app = QApplication([]); app.setApplicationName("Expression Lab"); window = MainWindow(args.device); window.show(); raise SystemExit(app.exec())


if __name__ == "__main__": main()
