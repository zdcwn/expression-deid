"""Real release checks, executed by the frozen executable without a source tree."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-package', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto')
    args = parser.parse_args(argv)
    out = args.check_package.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    result = {'frozen': bool(getattr(sys, 'frozen', False)), 'checks': [], 'status': 'running'}
    report = out / 'package-check.json'

    def record(name, **details):
        result['checks'].append({'name': name, **details})
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    try:
        import socket
        # Any attempted download fails: every exercised path must work offline.
        def offline(*_args, **_kwargs):
            raise RuntimeError('Release check prohibits network access')
        socket.create_connection = offline
        from expression_deid.distribution import resource_root, resolve_device, verify_assets
        manifest = verify_assets(deep=True)
        result['resource_root'] = str(resource_root())
        record('offline_resource_integrity', files=len(manifest.get('files', [])))
        device, description = resolve_device(args.device)
        result['device'] = device
        record('device_probe', description=description)

        import cv2
        import numpy as np
        import torch
        import mediapipe
        record('native_inference_imports', torch=torch.__version__, mediapipe=mediapipe.__version__)
        from PySide6.QtCore import QEventLoop, QTimer
        from PySide6.QtGui import QFont, QFontMetrics
        from PySide6.QtWidgets import QApplication
        from expression_deid.config import EXAMPLE_VIDEO_DIR, KMU_ALIGNED_DIR
        from expression_deid.ui.desktop import MainWindow, VideoWorker, release_experiment_pipeline_cache, samples
        app = QApplication.instance() or QApplication([])
        app.setQuitOnLastWindowClosed(False)
        window = MainWindow(device)
        window.show()
        app.processEvents()
        if not QFontMetrics(QFont('Microsoft YaHei UI')).inFontUcs4(ord('表')):
            raise AssertionError('Chinese font unavailable')
        for width, height in [(1366, 768), (980, 620)]:
            window.resize(width, height)
            app.processEvents()
            if window.width() > width or window.height() > height:
                raise AssertionError(f'Window exceeds {width}x{height}')
            for index in range(3):
                window.select_page(index)
                app.processEvents()
                if width == 1366:
                    window.grab().save(str(out / f'page_{index + 1}.png'))
        record('three_pages_and_screen_sizes', sizes=['1366x768', '980x620'])
        # Decode every bundled gallery image, not just the items initially shown.
        gallery = [p for p in KMU_ALIGNED_DIR.rglob('*') if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}]
        for image in gallery:
            if cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_GRAYSCALE) is None:
                raise AssertionError(f'Invalid sample: {image.name}')
        videos = sorted(EXAMPLE_VIDEO_DIR.glob('example*.mp4'))
        video_inventory = []
        for video in videos:
            cap = cv2.VideoCapture(str(video))
            try:
                ok, frame = cap.read()
                if not ok:
                    raise AssertionError(f'Cannot decode {video.name}')
                video_inventory.append({'name': video.name, 'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))})
            finally:
                cap.release()
        record('all_example_media_decode', gallery_images=len(gallery), visible_samples=len(samples()), videos=video_inventory)

        page = window.pages[0]
        if len(page.items) != len(gallery) or page.sample_list.count() != len(gallery):
            raise AssertionError('Some bundled images are missing from the visible gallery')
        filter_counts = {}
        for i in range(1, page.expression_filter.count()):
            page.expression_filter.setCurrentIndex(i)
            app.processEvents()
            expression = page.expression_filter.currentData()
            expected = sum(item['label'] == expression for item in page.items)
            visible = sum(not page.sample_list.item(j).isHidden() for j in range(page.sample_list.count()))
            if visible != expected:
                raise AssertionError('Expression filter hides valid gallery images')
            filter_counts[str(expression)] = visible
        page.expression_filter.setCurrentIndex(0)
        window.select_page(0); app.processEvents()
        last = page.sample_list.item(page.sample_list.count() - 1)
        page.sample_list.setCurrentItem(last); page.sample_list.scrollToItem(last)
        app.processEvents(); page._load_visible_thumbnails()
        if last.icon().isNull():
            raise AssertionError('Last gallery thumbnail did not load')
        record('complete_gallery_browsing', rows=page.sample_list.count(), filters=filter_counts)

        window.resize(1366, 768)
        window.select_page(0)
        page = window.pages[0]
        # The gallery is complete; release smoke inference is stratified.
        # The presentation check separately runs the full gallery batch.
        picked = {}
        for item in page.items:
            bucket = picked.setdefault(item['label'], [])
            if len(bucket) < (1 if device == 'cpu' else 4):
                bucket.append(item)
        items = [item for bucket in picked.values() for item in bucket]
        if all(item['path'] != page.items[-1]['path'] for item in items):
            items.append(page.items[-1])
        errors, image_results = [], []
        page._start(items)
        worker = page.worker
        loop = QEventLoop()
        worker.error.connect(errors.append)
        def image_done(payload, *_):
            data = payload['result']
            if data.get('expression_input_source') != 'anonymized_only':
                errors.append('Non-anonymous expression input')
            image_results.append({'file': payload['item']['path'].name,
                                  'expression': data['expression_anonymized']['name'],
                                  'confidence': data['expression_anonymized']['confidence']})
        worker.result_ready.connect(image_done)
        worker.finished.connect(loop.quit)
        deadline = QTimer()
        deadline.setSingleShot(True)
        deadline.timeout.connect(worker.requestInterruption)
        deadline.start(240_000)
        started = time.perf_counter()
        loop.exec()
        deadline.stop()
        app.processEvents()
        if errors or len(image_results) != len(items):
            raise AssertionError(f'Image inference failed: {errors}; {len(image_results)}/{len(items)}')
        window.grab().save(str(out / 'image_actual.png'))
        record('real_image_worker', samples=len(image_results), seconds=time.perf_counter() - started, predictions=image_results)
        release_experiment_pipeline_cache()
        if device == 'cuda':
            torch.cuda.empty_cache()

        window.select_page(1)
        for mode in ['fast', 'quality']:
            video_page = window.pages[1]
            video_page.mode_combo.setCurrentIndex(0 if mode == 'fast' else 1)
            video_page._start(('video', str(videos[0])), f'打包检查 · {mode}')
            worker = video_page.worker
            loop = QEventLoop()
            errors, received = [], []
            worker.worker_error.connect(errors.append)
            def receive(*payload):
                received.append(payload[2])
                if len(received) >= 12:
                    window.grab().save(str(out / f'video_{mode}.png'))
                    video_page.stop()
            worker.frame_ready.connect(receive)
            worker.finished.connect(loop.quit)
            deadline = QTimer()
            deadline.setSingleShot(True)
            deadline.timeout.connect(video_page.stop)
            deadline.start(180_000)
            loop.exec()
            deadline.stop()
            app.processEvents()
            if errors or len(received) < 12 or video_page.worker is not None:
                raise AssertionError(f'Video {mode} failed: {errors}; frames={len(received)}')
            record(f'real_video_{mode}', frames=len(received), expressions=received)
        window.close()
        app.processEvents()
        if window.isVisible():
            raise AssertionError('Window did not close')
        record('clean_shutdown')
        result['status'] = 'pass'
        return 0
    except Exception:
        result['status'] = 'fail'
        result['error'] = traceback.format_exc()
        return 1
    finally:
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    raise SystemExit(main())
