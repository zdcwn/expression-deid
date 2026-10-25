"""Measure expression decisions and full analysis latency on local videos."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from expression_deid.ui.opencv_demo import build_pipeline, _resize_rgb
from expression_deid.labels import EXPRESSION_NAMES
from expression_deid.runtime.video import FastVideoRunner


def switches(labels):
    return sum(a != b for a, b in zip(labels, labels[1:]))


def track_switches(rows, key):
    previous = None
    previous_track = None
    count = 0
    for row in rows:
        if row['status'] != 'ok':
            previous = None
            continue
        if row.get('track_id') != previous_track:
            previous = None
        previous_track = row.get('track_id')
        current = row.get(key)
        if current in (None, 'uncertain'): continue
        if previous is not None: count += previous != current
        previous = current
    return count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('videos', nargs='+')
    p.add_argument('--checkpoint')
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    p.add_argument('--frames', type=int, default=240)
    p.add_argument('--expected', choices=EXPRESSION_NAMES, help='Only for clips with this known label throughout')
    p.add_argument('--human-display', action='store_true', help='Benchmark the desktop visual GAN at interval 15')
    p.add_argument('--quality-manifest', type=Path, help='验证准确模式；缺省验证原快速模型')
    p.add_argument('--output', default='outputs/video_stability_check')
    args = p.parse_args()
    import torch
    torch.set_num_threads(1)
    pipeline = build_pipeline(args.device, use_gan=False, expression_checkpoint=args.checkpoint)
    display = None
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    summaries = []
    try:
        display = build_pipeline(args.device, use_gan=True) if args.human_display else None
        predictor = None
        if args.quality_manifest:
            from expression_deid.runtime.expression import QualityExpressionPredictor
            predictor = QualityExpressionPredictor(args.device, manifest=args.quality_manifest)
        for path in map(Path, args.videos):
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise RuntimeError(f'Cannot open {path}')
            fps = capture.get(cv2.CAP_PROP_FPS)
            fps = fps if np.isfinite(fps) and fps > 0 else 30
            runner = FastVideoRunner(pipeline, display_pipeline=display, display_interval=1 if predictor is not None else 15,
                                     detect_interval=1 if predictor is not None else 5,
                                     expression_predictor=predictor,
                                     track_reset_gap=2.5 if predictor is not None else .75)
            rows = []
            try:
                for index in range(args.frames):
                    ok, bgr = capture.read()
                    if not ok:
                        break
                    rgb = _resize_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), 960)
                    started = time.perf_counter()
                    _, info = runner.process_frame(rgb, index, timestamp=index / fps)
                    rows.append({'frame': index, 'timestamp': index / fps,
                                 'analysis_seconds': time.perf_counter() - started, **info})
            finally:
                capture.release()
            valid = [r for r in rows if r['status'] == 'ok']
            decided = [r for r in valid if r['expression'] != 'uncertain']
            summary = dict(video=path.name, frames=len(rows), detected_frames=len(valid),
                           decided_frames=len(decided), coverage=len(decided) / max(1, len(valid)),
                           display_landmark_failures=sum(r.get('display_status', 'ok') != 'ok' for r in valid),
                           display_concealed_frames=sum(r.get('display_status') == 'concealed_after_landmark_loss' for r in valid),
                           raw_switches=track_switches(rows, 'raw_expression'),
                           decision_switches=track_switches(rows, 'expression'),
                           counts={name: sum(r['expression'] == name for r in valid) for name in (*EXPRESSION_NAMES, 'uncertain')},
                           attempted_fps=len(rows) / max(1e-6, sum(r['analysis_seconds'] for r in rows)),
                           analysis_fps=len(valid) / max(1e-6, sum(r['analysis_seconds'] for r in rows)),
                           steady_analysis_fps=sum(r['status']=='ok' for r in rows[5:]) / max(1e-6, sum(r['analysis_seconds'] for r in rows[5:])),
                           analysis_ms_p95=float(np.quantile([r['analysis_seconds'] * 1000 for r in rows], .95)) if rows else None,
                           expression_input_source='anonymized_only', human_display=args.human_display,
                           expression_backend='quality_tta_fp32' if predictor is not None else 'compact_single_frame')
            if args.expected:
                summary.update(expected=args.expected,
                               raw_accuracy=sum(r['raw_expression'] == args.expected for r in valid) / max(1, len(valid)),
                               decision_accuracy_all_detected=sum(r['expression'] == args.expected for r in valid) / max(1, len(valid)),
                               decision_accuracy_when_decided=sum(r['expression'] == args.expected for r in decided) / max(1, len(decided)))
            (out / (path.stem + '.json')).write_text(json.dumps({'summary': summary, 'rows': rows}, ensure_ascii=False, indent=2), encoding='utf-8')
            summaries.append(summary); print(json.dumps(summary, ensure_ascii=False), flush=True)
        (out / 'summary.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8')
    finally:
        if display is not None: display.close()
        pipeline.close()


if __name__ == '__main__':
    main()
