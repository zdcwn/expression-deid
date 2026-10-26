"""Render all native pages and exercise real image/video delivery after UI edits.

Screenshots use actual model outputs. No camera capture or synthetic scores.
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

import torch
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtGui import QFont, QFontMetrics
from PySide6.QtWidgets import QApplication
from expression_deid.ui.desktop import MainWindow


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--output',default='outputs/checks/desktop_presentation')
    args=parser.parse_args(); out=ROOT/args.output; out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    app=QApplication.instance() or QApplication([]); app.setQuitOnLastWindowClosed(False)
    window=MainWindow(args.device); window.show(); app.processEvents()
    assert QFontMetrics(QFont('Microsoft YaHei UI')).inFontUcs4(ord('表'))
    results={'device':args.device,'checks':[],'screenshots':[],'model_changes':False}

    def snapshot(name):
        app.processEvents()
        path=out/(name+'.png'); assert window.grab().save(str(path))
        results['screenshots'].append(str(path))

    # Pages must remain inside the requested window, including the minimum size.
    for width,height in ((1440,920),(1280,800),(1100,760)):
        window.resize(width,height); app.processEvents()
        assert (window.width(),window.height())==(width,height)
        for i,name in enumerate(('images','video','performance')):
            window.select_page(i); app.processEvents()
            assert window.stack.currentIndex()==i
            assert window.pages[i].width()<=window.width()-202
            snapshot(f'{name}_{width}x{height}')
    results['checks'].append('three pages at 1440x920 / 1280x800 / 1100x760')
    window.resize(1440,920); window.select_page(2); app.processEvents()
    performance=window.pages[2]
    for i in range(performance.scope.count()):
        performance.scope.setCurrentIndex(i); app.processEvents()
        key=performance.scope.currentData(); expected=performance.current[key]
        assert performance.sample_count.value_label.text()==f"{expected['samples']:,}"
        assert sum(n for _,n,_ in performance.bars.rows)==expected['samples']
    performance.scope.setCurrentIndex(0)
    results['checks'].append('dataset selector matches saved sample counts and class rows')

    window.select_page(0); page=window.pages[0]
    errors=[]; page.run_all(); worker=page.worker
    assert worker is not None
    loop=QEventLoop(); worker.error.connect(errors.append); worker.finished.connect(loop.quit)
    cutoff=QTimer(); cutoff.setSingleShot(True); cutoff.timeout.connect(worker.requestInterruption); cutoff.start(600000)
    loop.exec(); cutoff.stop(); assert worker.wait(10000) and not errors, (errors, len(page.results), len(page.items))
    assert len(page.results)==len(page.items)>0 and page.run_button.isEnabled()
    results['image_samples']=len(page.results)
    results['gallery_rows']=page.sample_list.count()
    for payload,_,_,_ in page.results.values():
        assert payload['result']['expression_input_source']=='anonymized_only'
    # Cached thumbnails must show the selected sample without a new model worker.
    last_worker=page.worker
    happy_index=next(i for i,item in enumerate(page.items) if item['label']==3)
    page.sample_list.setCurrentRow(happy_index); page.run_selected(); app.processEvents()
    assert page.worker is last_worker
    assert page.truth_card.value_label.text()=='高兴'
    assert page.original_view[1].source_pixmap is not None
    assert page.anonymized_view[1].source_pixmap is not None
    snapshot('images_actual')
    results['checks'].append('all real example results, cached selection, anonymous-only provenance')

    window.select_page(1); video=window.pages[1]
    assert [video.mode_combo.itemData(i) for i in range(video.mode_combo.count())] == ['fast', 'quality']
    assert video.mode_combo.currentData() == 'fast'
    index=video.examples.findData('example_kmu_expressions.mp4'); assert index>=0
    video.examples.setCurrentIndex(index); video.play_example(); worker=video.worker
    assert worker.mode == 'fast'
    errors=[]; count=[0]; loop=QEventLoop(); worker.worker_error.connect(errors.append)
    def receive(*payload):
        count[0]+=1
        if count[0]==70: snapshot('video_actual')
        if count[0]>=80: video.stop()
    worker.frame_ready.connect(receive); worker.finished.connect(loop.quit)
    cutoff=QTimer(); cutoff.setSingleShot(True); cutoff.timeout.connect(video.stop); cutoff.start(90000)
    loop.exec(); cutoff.stop(); app.processEvents()
    assert not errors and count[0]>=80 and video.worker is None, (errors,count)
    assert video.anonymized[1].overlay is not None
    assert video.fps_trace.values and video.frame_card.value_label.text()!='—'
    results['video_frames']=count[0]
    results['checks'].append('real video frames, one-frame acknowledgement, overlay, FPS trace and stop')
    window.select_page(2); snapshot('performance_actual')
    window.close(); app.processEvents(); assert not window.isVisible()
    results['checks'].append('clean window close after workers finish')
    results['status']='pass'
    (out/'presentation-check.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
