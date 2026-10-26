"""Native presentation widgets. No model calls or source-image classification."""
from __future__ import annotations

import os
from collections import deque
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

INK = '#1c2b35'
MUTED = '#77858e'
GREEN = '#25816c'
LINE = '#e0e6e8'


def ensure_fonts():
    """Qt offscreen has no Windows font discovery; also keep Chinese readable."""
    if getattr(ensure_fonts, '_loaded', False):
        return
    folder = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts'
    for name in ('msyh.ttc', 'msyhbd.ttc', 'segoeui.ttf', 'segoeuib.ttf'):
        if (folder / name).is_file():
            QFontDatabase.addApplicationFont(str(folder / name))
    # The release remains readable on Windows installations without CJK fonts.
    bundled = Path(__file__).resolve().parent / 'assets' / 'NotoSansSC.ttf'
    if bundled.is_file():
        QFontDatabase.addApplicationFont(str(bundled))
        QFont.insertSubstitutions('Microsoft YaHei UI', ['Noto Sans SC'])
    ensure_fonts._loaded = True


def line_icon(kind: str, color=GREEN, size=24) -> QIcon:
    pix = QPixmap(size, size); pix.fill(Qt.transparent)
    p = QPainter(pix); p.setRenderHint(QPainter.Antialiasing)
    p.scale(size / 24, size / 24)
    p.setPen(QPen(QColor(color), 1.6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    if kind == 'image':
        p.drawRoundedRect(QRectF(3, 4, 18, 16), 3, 3); p.drawEllipse(QPointF(8, 9), 1.2, 1.2)
        path = QPainterPath(QPointF(4, 17)); path.lineTo(10, 12); path.lineTo(14, 16); path.lineTo(17, 13); path.lineTo(20, 16); p.drawPath(path)
    elif kind == 'video':
        p.drawRoundedRect(QRectF(3, 5, 13, 14), 3, 3)
        path = QPainterPath(QPointF(16, 9)); path.lineTo(21, 6); path.lineTo(21, 18); path.lineTo(16, 15); p.drawPath(path)
    elif kind == 'chart':
        p.drawLine(4, 4, 4, 20); p.drawLine(4, 20, 21, 20)
        p.drawLine(8, 15, 8, 18); p.drawLine(13, 10, 13, 18); p.drawLine(18, 5, 18, 18)
    elif kind == 'play':
        path = QPainterPath(QPointF(8, 5)); path.lineTo(19, 12); path.lineTo(8, 19); path.closeSubpath(); p.drawPath(path)
    elif kind == 'stop':
        p.drawRoundedRect(QRectF(6, 6, 12, 12), 2, 2)
    elif kind == 'folder':
        path = QPainterPath(QPointF(3, 18)); path.lineTo(3, 5); path.lineTo(10, 5); path.lineTo(12, 8); path.lineTo(21, 8); path.lineTo(21, 18); path.closeSubpath(); p.drawPath(path)
    else:
        p.drawRoundedRect(QRectF(4, 3, 16, 18), 6, 6)
        p.drawLine(8, 10, 9, 10); p.drawLine(15, 10, 16, 10)
        path = QPainterPath(QPointF(9, 15)); path.quadTo(12, 17, 15, 15); p.drawPath(path)
    p.end(); return QIcon(pix)


def label(text: str, role: str, *, wrap=False) -> QLabel:
    w = QLabel(text); w.setObjectName(role); w.setWordWrap(wrap); return w


def page_heading(kicker: str, title: str, subtitle: str):
    box = QVBoxLayout(); box.setSpacing(5)
    box.addWidget(label(kicker, 'eyebrow'))
    box.addWidget(label(title, 'pageTitle'))
    box.addWidget(label(subtitle, 'pageSubtitle', wrap=True))
    return box


class ImageViewport(QLabel):
    """Keep the full source pixmap so resizing never stretches/crops face images."""
    def __init__(self, empty: str):
        super().__init__(); self.empty = empty; self.source_pixmap = None; self.overlay = None
        self.setObjectName('imageViewport'); self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(180, 170); self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)

    def show_image(self, image):
        self.source_pixmap = QPixmap.fromImage(image); self._fit()

    def clear_image(self, text=None):
        if text is not None: self.empty = text
        self.source_pixmap = None; self.overlay = None; self.clear(); self.update()

    def set_overlay(self, title, subtitle):
        self.overlay = (title, subtitle); self.update()

    def _fit(self):
        if self.source_pixmap is not None and not self.source_pixmap.isNull():
            self.setPixmap(self.source_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event); self._fit()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.source_pixmap is not None:
            if self.overlay:
                p=QPainter(self); p.setRenderHint(QPainter.Antialiasing); p.setPen(Qt.NoPen)
                p.setBrush(QColor(15,30,36,225)); p.drawRoundedRect(QRectF(14,14,min(204,self.width()-28),68),8,8)
                p.setFont(QFont('Microsoft YaHei UI',13,QFont.Bold)); p.setPen(QColor('#e6f6eb'))
                p.drawText(QRectF(28,23,self.width()-56,27),Qt.AlignVCenter,self.overlay[0])
                p.setFont(QFont('Microsoft YaHei UI',9)); p.setPen(QColor('#a7c9bb'))
                p.drawText(QRectF(28,52,self.width()-56,22),Qt.AlignVCenter,self.overlay[1]); p.end()
            return
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        cx, cy = self.width() / 2, self.height() / 2 - 18
        p.setPen(QPen(QColor('#3c505b'), 1.3)); p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(QRectF(cx-30, cy-36, 60, 72), 23, 23)
        p.drawLine(QPointF(cx-13, cy-7), QPointF(cx-8, cy-7)); p.drawLine(QPointF(cx+8, cy-7), QPointF(cx+13, cy-7))
        path=QPainterPath(QPointF(cx-10,cy+14)); path.quadTo(cx,cy+20,cx+10,cy+14); p.drawPath(path)
        p.setFont(QFont('Microsoft YaHei UI', 9)); p.setPen(QColor('#8b9ea7'))
        p.drawText(QRectF(8, cy+57, self.width()-16, 30), Qt.AlignCenter, self.empty); p.end()


class MediaCard(QFrame):
    def __init__(self, title: str, caption: str, empty: str, accent=False):
        super().__init__(); self.setObjectName('mediaCard')
        layout=QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(0)
        top=QFrame(); top.setObjectName('mediaTop'); row=QHBoxLayout(top); row.setContentsMargins(16, 12, 16, 12)
        row.addWidget(label(title,'mediaHeading')); row.addStretch()
        row.addWidget(label('匿名结果' if accent else '灰度输入','mediaTag'))
        layout.addWidget(top)
        self.image=ImageViewport(empty); layout.addWidget(self.image,1)
        self.caption=label(caption,'mediaCaption'); self.caption.setMinimumWidth(0)
        self.caption.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Fixed)
        self.caption.setToolTip(caption); layout.addWidget(self.caption)

    def pair(self): return self, self.image

    def set_caption(self, text):
        self.caption.setText(text); self.caption.setToolTip(text)


class MetricCard(QFrame):
    def __init__(self, title: str, value: str, note: str='', accent=True, parent=None):
        super().__init__(parent); self.setProperty('card',True)
        layout=QVBoxLayout(self); layout.setContentsMargins(18,14,18,14); layout.setSpacing(4)
        self.title_label=label(title,'cardTitle'); layout.addWidget(self.title_label)
        self.value_label=label(value,'cardValue' if accent else 'cardValueMuted')
        self.value_label.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Preferred); layout.addWidget(self.value_label)
        self.note_label=label(note,'cardNote',wrap=True); layout.addWidget(self.note_label)
        self.setMinimumWidth(0)

    def set_value(self,value): self.value_label.setText(value)


class ConfidenceMeter(QWidget):
    def __init__(self):
        super().__init__(); self.value=None; self.setFixedHeight(6)
    def set_value(self,value):
        self.value=float(value) if value is not None and np.isfinite(value) else None; self.update()
    def paintEvent(self,event):
        p=QPainter(self); p.setRenderHint(QPainter.Antialiasing); p.setPen(Qt.NoPen)
        p.setBrush(QColor('#e5ebea')); p.drawRoundedRect(QRectF(0,0,self.width(),6),3,3)
        if self.value is not None:
            p.setBrush(QColor(GREEN)); p.drawRoundedRect(QRectF(0,0,self.width()*np.clip(self.value,0,1),6),3,3)
        p.end()


class FpsTrace(QWidget):
    def __init__(self):
        super().__init__(); self.values=deque(maxlen=72); self.setMinimumHeight(64)
        self.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
    def add_value(self,value):
        if np.isfinite(value): self.values.append(float(value))
        self.update()
    def reset(self): self.values.clear(); self.update()
    def paintEvent(self,event):
        p=QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w,h=self.width(),self.height(); ceiling=max(40,max(self.values,default=0)*1.1)
        target=h-8-24/ceiling*(h-16)
        p.setPen(QPen(QColor('#c3d2ce'),1,Qt.DashLine)); p.drawLine(QPointF(0,target),QPointF(w,target))
        if len(self.values)>1:
            path=QPainterPath()
            for i,v in enumerate(self.values):
                point=QPointF(i/(len(self.values)-1)*(w-2)+1,h-8-min(v/ceiling,1)*(h-16))
                if i: path.lineTo(point)
                else: path.moveTo(point)
            p.setPen(QPen(QColor(GREEN),2)); p.drawPath(path)
        else:
            p.setPen(QColor(MUTED)); p.setFont(QFont('Microsoft YaHei UI',9))
            p.drawText(self.rect(),Qt.AlignCenter,'开始分析后显示处理帧率')
        p.end()


class AccuracyBars(QWidget):
    def __init__(self):
        super().__init__(); self.rows=[]; self.setMinimumHeight(286)
        self.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
    def set_rows(self,rows): self.rows=rows; self.update()
    def paintEvent(self,event):
        p=QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        if not self.rows:
            p.setPen(QColor(MUTED)); p.drawText(self.rect(),Qt.AlignCenter,'没有该数据集的分类结果'); p.end(); return
        row_height=min(43,(self.height()-8)/len(self.rows)); width=max(20,self.width()-166)
        for i,(name,n,value) in enumerate(self.rows):
            y=i*row_height+8; p.setPen(QColor(INK)); p.setFont(QFont('Microsoft YaHei UI',10))
            p.drawText(QRectF(0,y-2,46,24),Qt.AlignVCenter,name)
            p.setPen(Qt.NoPen); p.setBrush(QColor('#eef2f2')); p.drawRoundedRect(QRectF(57,y+5,width,8),4,4)
            if value is not None and np.isfinite(value):
                p.setBrush(QColor(GREEN)); p.drawRoundedRect(QRectF(57,y+5,width*np.clip(value,0,1),8),4,4)
            p.setPen(QColor(INK)); p.setFont(QFont('Segoe UI',10))
            p.drawText(QRectF(65+width,y-2,52,24),Qt.AlignRight|Qt.AlignVCenter,f'{value*100:.1f}%' if value is not None else '—')
            p.setPen(QColor(MUTED)); p.setFont(QFont('Segoe UI',8))
            p.drawText(QRectF(125+width,y-2,40,24),Qt.AlignRight|Qt.AlignVCenter,str(n))
        p.end()


STYLESHEET = """
QMainWindow, QWidget { background: #f4f6f7; color: #1c2b35; font-family: 'Microsoft YaHei UI', 'Segoe UI'; font-size: 12px; }
QLabel { background: transparent; border: none; }
#sidebar { background: #14222c; border: none; }
#brand { color: #f4f9f7; font-family: 'Segoe UI'; font-size: 23px; font-weight: 700; }
#brandSub { color: #7f969f; font-size: 10px; }
#sideCaption { color: #677f8b; font-size: 10px; }
#sideNote { color: #a0b1b9; font-size: 11px; }
#navSection { color: #64808d; font-size: 10px; padding: 0px 10px; }
#workspaceHeader { border-bottom: 1px solid #dfe5e8; }
#appTitle { font-size: 12px; color: #73858e; }
#topStatus, #statusChip { background: #e7f2ed; color: #287762; border-radius: 12px; padding: 6px 12px; font-size: 11px; }
#eyebrow { color: #25816c; font-family: 'Segoe UI'; font-size: 10px; font-weight: 700; letter-spacing: 2px; }
#pageTitle { font-size: 28px; font-weight: 700; color: #1c2b35; }
#pageSubtitle { color: #77858e; font-size: 12px; }
QPushButton { background: white; color: #344750; border: 1px solid #dce4e6; border-radius: 8px; padding: 10px 15px; font-weight: 600; }
QPushButton:hover { border-color: #73b09e; background: #f2faf6; }
QPushButton:pressed { background: #e0eee8; }
QPushButton:disabled { background: #eef1f2; color: #9aa9af; border-color: #e3e8ea; }
QPushButton[nav="true"] { background: transparent; color: #9bafb9; border: 0; border-radius: 8px; text-align: left; padding: 13px 12px; font-size: 12px; }
QPushButton[nav="true"]:hover { background: #20343f; color: #f0f7f4; }
QPushButton[nav="true"]:checked { background: #def1e7; color: #204b3e; font-weight: 700; }
#primaryButton { background: #25816c; color: white; border-color: #25816c; }
#primaryButton:hover { background: #1d6e5b; }
#primaryButton:disabled { background: #9ebdb3; color: #eef7f2; border-color: #9ebdb3; }
#dangerButton { color: #a95750; }
[panel="true"], [card="true"] { background: white; border: 1px solid #e0e6e8; border-radius: 12px; }
#panelTitle { color: #223741; font-size: 14px; font-weight: 700; }
#panelNote, #cardNote, #detailLine { color: #7a8a93; font-size: 11px; }
#cardTitle { color: #798992; font-size: 11px; }
#cardValue { color: #25816c; font-family: 'Microsoft YaHei UI', 'Segoe UI'; font-size: 27px; font-weight: 700; }
#cardValueMuted { color: #253d48; font-size: 27px; font-weight: 700; }
#metricLine { color: #506974; font-size: 11px; }
#mediaCard { background: #14212a; border: 1px solid #263a46; border-radius: 12px; }
#mediaTop { background: #1b2c36; border-top-left-radius: 11px; border-top-right-radius: 11px; }
#mediaHeading { color: #e1ebe9; font-size: 12px; font-weight: 600; }
#mediaTag { color: #94c9b4; font-size: 10px; }
#mediaCaption { color: #79949f; font-family: 'Segoe UI'; font-size: 10px; padding: 10px 16px; }
#imageViewport { background: #101b22; border: none; }
QListWidget { background: transparent; border: none; outline: 0; color: #4f6671; }
QListWidget::item { padding: 8px; margin: 3px 0; border-radius: 8px; }
QListWidget::item:hover { background: #f0f5f3; }
QListWidget::item:selected { background: #e6f2ec; color: #206b54; }
QComboBox { background: white; border: 1px solid #dce4e6; border-radius: 8px; padding: 8px 12px; min-width: 110px; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox QAbstractItemView { background: white; selection-background-color: #e0f1e8; selection-color: #204b3e; padding: 5px; }
QProgressBar { background: #e7eeeb; border: none; border-radius: 3px; height: 5px; }
QProgressBar::chunk { background: #25816c; border-radius: 3px; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: transparent; width: 8px; margin: 4px 0; }
QScrollBar::handle:vertical { background: #ced8db; border-radius: 4px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
#notice { background: #fff6e7; color: #8b682f; border: 1px solid #f0e2c8; border-radius: 8px; padding: 10px 14px; font-size: 11px; }
#pipelineStep { color: #536d78; font-size: 11px; }
#pipelineArrow { color: #9fb4ae; }
QToolTip { background: #233b46; color: white; border: none; padding: 7px; }
"""
