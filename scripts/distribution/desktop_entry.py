"""Windowed release entry. It never needs the source checkout or a Python install."""
from __future__ import annotations

import ctypes
import faulthandler
import logging
from logging.handlers import RotatingFileHandler
import multiprocessing
import os
from pathlib import Path
import sys
import traceback


def main() -> int:
    multiprocessing.freeze_support()
    # MediaPipe imports plotting helpers; this app renders with Qt, not pyplot.
    os.environ['MPLBACKEND'] = 'Agg'
    from expression_deid.distribution import user_data_dir

    log_dir = user_data_dir() / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / 'desktop.log'
    handler = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=2, encoding='utf-8')
    logging.basicConfig(level=logging.INFO, handlers=[handler], format='%(asctime)s %(levelname)s %(message)s')
    # Windowed executables have no console. Native libraries still expect streams.
    stream = (log_dir / 'native.log').open('w', encoding='utf-8', buffering=1)
    sys.stdout = sys.stderr = stream
    faulthandler.enable(stream)
    logging.info('Launching Expression Lab; frozen=%s', getattr(sys, 'frozen', False))
    splash = None
    try:
        if '--check-package' in sys.argv:
            from expression_deid.distribution_check import main as check
            return check(sys.argv[1:])

        from PySide6.QtCore import Qt
        from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPixmap
        from PySide6.QtWidgets import QApplication, QSplashScreen
        QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
        app = QApplication.instance() or QApplication([])
        app.setApplicationName('Expression Lab')
        from expression_deid.ui.widgets import ensure_fonts
        ensure_fonts()
        pixmap = QPixmap(540, 220)
        pixmap.fill(QColor('#173c33'))
        painter = QPainter(pixmap)
        painter.setPen(QColor('#e8f4ed'))
        painter.setFont(QFont('Segoe UI', 30, QFont.Bold))
        painter.drawText(30, 72, 'expression.')
        painter.setFont(QFont('Microsoft YaHei UI', 13))
        painter.drawText(32, 114, '座舱隐私实验台')
        painter.end()
        splash = QSplashScreen(pixmap)
        splash.showMessage('正在检查运行资源与计算设备…', Qt.AlignBottom | Qt.AlignHCenter, QColor('#c2dacf'))
        splash.show()
        app.processEvents()

        from expression_deid.distribution import resolve_device, verify_assets
        verify_assets()
        requested = 'auto'
        if '--device' in sys.argv:
            requested = sys.argv[sys.argv.index('--device') + 1]
        device, description = resolve_device(requested)
        from expression_deid.ui.desktop import MainWindow
        window = MainWindow(device)
        window.status.setText(description)
        window.show()
        splash.finish(window)
        return app.exec()
    except Exception:
        logging.exception('Desktop startup failed')
        if splash is not None:
            splash.close()
        message = f'软件启动失败。请确认已完整解压软件包。\n\n详细记录：{log_path}\n\n{traceback.format_exc().splitlines()[-1]}'
        if os.name == 'nt':
            ctypes.windll.user32.MessageBoxW(None, message, 'Expression Lab', 0x10)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
