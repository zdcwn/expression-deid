"""Exercise real Qt model workers, video encoding, camera opening and shutdown.

Offscreen mode checks data delivery and thread lifetime, not visual design.
Original camera frames are held in memory only and are never saved here.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import av
import cv2
import numpy as np
import torch
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication
from expression_deid.ui.desktop import ExperimentWorker, MainWindow, VideoWorker, samples
from expression_deid.runtime.video import run_video, FastVideoRunner, _stream_pipelines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda', choices=('cpu', 'cuda'))
    parser.add_argument('--output', default='outputs/checks/desktop_runtime')
    args = parser.parse_args()
    torch.set_num_threads(1)
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    rows = []

    def check(name, callback):
        start = time.perf_counter()
        try:
            detail = callback()
            rows.append(dict(check=name, status='pass', seconds=time.perf_counter()-start, detail=detail))
        except Exception as exc:
            rows.append(dict(check=name, status='fail', seconds=time.perf_counter()-start, error=f'{type(exc).__name__}: {exc}'))
        print(rows[-1], flush=True)

    def worker_video():
        worker = VideoWorker(('video', str(ROOT/'examples/videos/example_kmu_expressions.mp4')), args.device)
        loop = QEventLoop()
        frames, errors = [], []
        def receive(original, anonymous, expression, confidence, fps, index):
            worker.frame_pending.clear()
            assert not original.isNull() and not anonymous.isNull()
            frames.append(dict(index=index, expression=expression, confidence=confidence if np.isfinite(confidence) else None, fps=fps))
            if len(frames) >= 80: worker.stop()
        worker.frame_ready.connect(receive)
        worker.worker_error.connect(errors.append)
        worker.finished.connect(loop.quit)
        worker.start()
        timeout = QTimer(); timeout.setSingleShot(True)
        timeout.timeout.connect(worker.stop); timeout.start(60000)
        loop.exec(); timeout.stop()
        assert worker.wait(10000) and len(frames) >= 80 and not errors, (frames, errors)
        return dict(frames=len(frames), last_frame=frames[-1], worker_finished=worker.isFinished(), visual_review=False)
    check('real_qt_video_worker_delivers_images_and_stops', worker_video)

    def worker_experiments():
        items = samples()
        worker = ExperimentWorker(items, args.device)
        loop = QEventLoop(); received, errors = [], []
        def receive(payload, index, count, seconds):
            result = payload['result']
            assert result['expression_input_source'] == 'anonymized_only'
            received.append(dict(label=payload['item']['label'], predicted=result['expression_anonymized']['name'], seconds=seconds))
        worker.result_ready.connect(receive); worker.error.connect(errors.append); worker.finished.connect(loop.quit)
        worker.start(); loop.exec(); worker.wait(10000)
        assert len(received) == len(items) and not errors, (received, errors)
        return dict(samples=received, worker_finished=worker.isFinished(), note='Examples are not independent validation.')
    check('real_qt_experiment_worker_available_classes_anonymous_only', worker_experiments)

    def close_during_load():
        window = MainWindow(args.device); window.show(); app.processEvents()
        window.pages[1]._start(('video', str(ROOT/'examples/videos/example_mixed.mp4')), 'audit')
        worker = window.pages[1].worker
        state = {'finished': False}
        worker.finished.connect(lambda: state.update(finished=True))
        window.close()
        retained = window.isVisible() and worker.isRunning()
        loop = QEventLoop()
        def poll():
            if state['finished'] and not window.isVisible(): loop.quit()
        timer = QTimer(); timer.timeout.connect(poll); timer.start(50)
        cutoff = QTimer(); cutoff.setSingleShot(True); cutoff.timeout.connect(loop.quit); cutoff.start(60000)
        loop.exec(); timer.stop(); cutoff.stop()
        assert retained and state['finished'] and not window.isVisible()
        return dict(close_deferred_while_running=retained, finished_before_close=True)
    check('real_window_close_waits_for_loading_worker', close_during_load)

    def encode():
        video = out/'encoded.mp4'; telemetry = out/'encoded.jsonl'
        result = run_video(ROOT/'examples/videos/example_kmu_expressions.mp4', video, telemetry_path=telemetry, device=args.device, limit=24)
        with av.open(str(video)) as container:
            decoded = list(container.decode(video=0))
        entries = [json.loads(line) for line in telemetry.read_text(encoding='utf-8').splitlines()]
        assert len(decoded) == len(entries) == result['frames'] == 24
        assert all(row['expression_input_source'] == 'anonymized_only' for row in entries if row['status']=='ok')
        return dict(**result, decoded_frames=len(decoded), dimensions=[decoded[0].width, decoded[0].height])
    check('actual_pyav_encode_decode_and_telemetry', encode)

    def noface():
        source = out/'no_face.mp4'
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 30, (320, 240))
        assert writer.isOpened()
        for _ in range(10): writer.write(np.zeros((240, 320, 3), np.uint8))
        writer.release()
        result = run_video(source, out/'no_face_result.mp4', device=args.device)
        assert result['frames']==10 and result['successful_analysis_frames']==0 and result['analysis_fps']==0
        return result
    check('no_face_video_counts_zero_successful_analysis_fps', noface)

    def invalid_video():
        damaged = out/'damaged.mp4'; damaged.write_bytes(b'not a video')
        errors=[]
        for name, source, target in (
            ('missing', out/'absent.mp4', out/'missing-result.mp4'),
            ('damaged', damaged, out/'damaged-result.mp4'),
            ('same_path', damaged, damaged),
        ):
            try: run_video(source, target, device=args.device)
            except (OSError, ValueError) as exc: errors.append(dict(input=name, error=type(exc).__name__))
            else: raise AssertionError(name+' accepted')
        return errors
    check('missing_damaged_and_conflicting_video_paths_rejected', invalid_video)

    def camera():
        capture=cv2.VideoCapture(0, cv2.CAP_DSHOW)
        pipeline=display=None
        try:
            if not capture.isOpened(): return dict(available=False, verified=False, reason='Camera index 0 could not be opened; no camera quality claim.')
            pipeline,display=_stream_pipelines(args.device,None,None,None)
            runner=FastVideoRunner(pipeline, display_pipeline=display, display_interval=15)
            attempted=detected=0
            start=time.monotonic()
            while attempted < 30 and time.monotonic()-start < 20:
                ok,bgr=capture.read()
                if not ok: break
                _,info=runner.process_frame(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),attempted,timestamp=time.monotonic())
                detected += info['status']=='ok'; attempted+=1
            return dict(available=True, captured_frames=attempted, successful_analysis_frames=detected, ground_truth_accuracy_verified=False, original_frames_saved=False)
        finally:
            capture.release()
            if display is not None and display is not pipeline: display.close()
            if pipeline is not None: pipeline.close()
    check('actual_camera_open_and_bounded_inference', camera)
    (out/'summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    if any(row['status']=='fail' for row in rows): raise SystemExit(1)


if __name__=='__main__': main()
