"""Execute local entrypoints, malformed inputs, datasets and native UI lifecycle.

This is an execution audit, not a model accuracy or real-vehicle benchmark.
It never evaluates source pixels with the expression classifier.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import http.client
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PIL import Image
import torch

from expression_deid.app import build_pipeline
from expression_deid.config import KMU_ALIGNED_DIR, PPB_DIR, FERPLUS_DIR
from expression_deid.labels import EXPRESSION_NAMES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/full_audit_20261008/entrypoints.json')
    args=parser.parse_args(); torch.set_num_threads(1)
    results=[]

    def check(name, callback):
        start=time.perf_counter()
        try:
            detail=callback()
            results.append(dict(check=name,status='pass',seconds=time.perf_counter()-start,detail=detail))
        except Exception:
            results.append(dict(check=name,status='fail',seconds=time.perf_counter()-start,error=traceback.format_exc()))
        print(f'{name}: {results[-1]["status"]}',flush=True)

    def help_entrypoints():
        rows=[]
        for command in ('prepare','prepare-ppb','train-expression','train-identity','train-expression-kmu',
                        'train-expression-joint','train-anonymizer','infer','evaluate','video','camera'):
            argv=[sys.executable,'-m','expression_deid',command,'--help']
            completed=subprocess.run(argv,cwd=ROOT,capture_output=True,timeout=40)
            assert completed.returncode==0,(argv,completed.stderr.decode(errors='replace'))
            rows.append(dict(entry=command,exit=0))
        for module in ('app','video_demo','desktop_ui','desktop_ui_qt','web','prepare','prepare_ppb'):
            completed=subprocess.run([sys.executable,'-m','expression_deid.'+module,'--help'],cwd=ROOT,capture_output=True,timeout=40)
            assert completed.returncode==0,(module,completed.stderr.decode(errors='replace'))
            rows.append(dict(entry=module,exit=0))
        return rows
    check('all_cli_help_parsers',help_entrypoints)

    def fer_quality():
        from expression_deid.data import FERPlusExpressionDataset, FEREnhancedExpressionDataset
        selected={}
        for split in ('train','valid'):
            data=FERPlusExpressionDataset(split)
            counts=Counter(); hashes=set()
            for row in data.records:
                pixels=np.fromstring(row['pixels'],sep=' ',dtype=np.int16)
                assert pixels.size==2304 and pixels.min()>=0 and pixels.max()<=255
                assert row['agreement']>.5
                counts[row['label']]+=1
                hashes.add(hashlib.sha256(pixels.astype(np.uint8).tobytes()).hexdigest())
            selected[split]=(counts,hashes)
        original=FERPLUS_DIR/'original/fer2013.csv'
        with original.open(encoding='utf-8',newline='') as handle: images=list(csv.DictReader(handle))
        with (FERPLUS_DIR/'fer2013new.csv').open(encoding='utf-8',newline='') as handle: labels=list(csv.DictReader(handle))
        assert len(images)==len(labels)
        assert all(a['Usage']==b['Usage'] for a,b in zip(images,labels))
        enhanced=FEREnhancedExpressionDataset('valid')
        return dict(original_pairs=len(images),selected_counts={key:dict(value[0]) for key,value in selected.items()},
                    exact_pixel_overlap_train_valid=len(selected['train'][1]&selected['valid'][1]),
                    enhanced_valid_samples=len(enhanced),enhanced_first_label=enhanced.records[0]['label'])
    check('fer_csv_pairing_votes_pixels_and_label_order',fer_quality)

    def preparation():
        from expression_deid.prepare import align_kmu,_align_one,_load_partial
        from expression_deid.prepare_ppb import align_ppb_neutral
        from expression_deid.config import KMU_DIR
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder); sources=base/'kmu-input'; sources.mkdir()
            for source in sorted(KMU_DIR.glob('*.jpg'))[:2]: shutil.copy2(source,sources/source.name)
            kmu_rows=align_kmu(sources,base/'kmu')
            assert len(kmu_rows)==2 and all(row['status']=='ok' for row in kmu_rows)
            bad=sources/'01_AN_broken_999.jpg';bad.write_bytes(b'not an image')
            row=_align_one(bad,base/'kmu',None); assert row['status']=='fail' and row['reason'].startswith('read_error')
            partial=base/'kmu/_partial.jsonl';partial.write_text(json.dumps(row)+'\n')
            assert not _load_partial(partial,base/'kmu',sources)
            ppb_input=base/'ppb-input';ppb_input.mkdir()
            video=next((PPB_DIR/'raw-cir-neutral').glob('*-P04-ND.mp4'))
            os.link(video,ppb_input/video.name)
            ppb_rows=align_ppb_neutral(ppb_input,base/'ppb',frames_per_video=1)
            assert len(ppb_rows)==1 and ppb_rows[0]['status']=='ok'
            assert Image.open(base/'ppb'/ppb_rows[0]['output']).size==(256,256)
            return dict(kmu_records=kmu_rows,broken_source=row['reason'],ppb_records=ppb_rows)
    check('real_kmu_alignment_ppb_extraction_and_failed_resume',preparation)

    def inference_evaluation():
        from expression_deid.cli import main as cli
        from expression_deid.evaluate import evaluate_joint
        path=next(KMU_ALIGNED_DIR.glob('*.png'))
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder)
            cli(['infer','--anonymizer-checkpoint',str(ROOT/'models/anonymizer_adversarial_v1.pt'),
                 '--expression-checkpoint',str(ROOT/'models/expression_realtime_stable.pt'),
                 '--input',str(path),'--output',str(base/'infer'),'--device','cpu'])
            records=json.loads((base/'infer/results.json').read_text(encoding='utf-8'))
            assert len(records)==1 and records[0]['expression_input_source']=='anonymized_only'
            assert Image.open(base/'infer'/ (path.stem+'.png')).size==(256,256)
            report=evaluate_joint(anonymizer_checkpoint=str(ROOT/'models/anonymizer_adversarial_v1.pt'),
                expression_checkpoint=str(ROOT/'models/expression_realtime_stable.pt'),identity_checkpoint=str(ROOT/'models/identity_faceonly.pt'),
                output_dir=str(base/'evaluation'),limit=2,device='cpu',fast_model_only=True,privacy_selection=False)
            assert report['summary']['samples']==2 and report['provenance']['expression_ensemble_size']==1
            assert report['summary']['gallery_size']==1681 and not report['provenance']['diagnostic_errors']
            assert 'expression_consistency' not in report['summary']
            return dict(infer_samples=1,evaluation=report['summary'],provenance=report['provenance'])
    check('actual_cpu_infer_evaluate_identity_sface_and_gallery',inference_evaluation)

    def malformed_inputs():
        from expression_deid.models import load_checkpoint
        pipeline=build_pipeline('cpu',use_gan=False)
        try:
            attempts=[]
            for name,callback in (
                ('no_face',lambda:pipeline.process(np.zeros((480,640,3),np.uint8),aligned=False)),
                ('missing_image',lambda:pipeline.process(ROOT/'absent.png')),
                ('non_square_aligned',lambda:pipeline.process(np.zeros((100,150),np.uint8),aligned=True)),
                ('wrong_model_kind',lambda:build_pipeline('cpu',use_gan=False,expression_checkpoint=ROOT/'models/identity_faceonly.pt')),
            ):
                try: callback()
                except (RuntimeError,ValueError,OSError) as exc: attempts.append(dict(input=name,rejected=type(exc).__name__))
                else: raise AssertionError('invalid input accepted: '+name)
            return attempts
        finally: pipeline.close()
    check('no_face_missing_image_shape_and_wrong_model_errors',malformed_inputs)

    def web_http():
        from expression_deid.web import create_server
        server=create_server(port=0,device='cpu')
        worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        codes=[]
        def request(method,path,expected,body=None,headers=None):
            client=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=10)
            client.request(method,path,body=body,headers=headers or {})
            response=client.getresponse();response.read();code=response.status;client.close()
            assert code==expected,(method,path,code,expected)
            codes.append(dict(method=method,path=path,status=code))
        try:
            for path in ('/','/app.js','/styles.css','/favicon.svg','/api/status','/api/samples','/api/evaluation'):
                request('GET',path,200)
            sample=server.service.samples()[0];request('GET',sample['image'],200)
            request('GET','/models/identity.pt',404)
            request('GET','/../README.md',404)
            request('POST','/api/process',415,body='x')
            request('POST','/api/process',400,body='{',headers={'Content-Type':'application/json'})
            request('POST','/api/process',403,body='{}',headers={'Content-Type':'application/json','Origin':'http://other.invalid'})
            request('POST','/api/process',400,body='{"sample_id":"missing"}',headers={'Content-Type':'application/json'})
            assert server.service.pipeline is None
            return codes
        finally: server.shutdown();server.server_close();worker.join(3)
    check('unchanged_web_read_only_http_and_invalid_requests',web_http)

    def native_ui():
        from PySide6.QtWidgets import QApplication
        from expression_deid.desktop_ui_qt import MainWindow
        app=QApplication.instance() or QApplication([])
        window=MainWindow('cpu');window.show();app.processEvents()
        # Real UI construction and all three page switches; no camera accuracy claim.
        for index in range(3): window.stack.setCurrentIndex(index);app.processEvents()
        window.close();app.processEvents()
        return dict(pages=3,lifecycle='constructed, switched, closed offscreen',visual_review=False)
    check('real_native_ui_three_pages_construct_switch_close',native_ui)

    def dependency_consistency():
        completed=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,timeout=40)
        return dict(exit=completed.returncode,output=completed.stdout.decode(errors='replace'),
                    note='Shared environment; unrelated dependency conflicts are reported, not automatically upgraded.')
    check('environment_pip_check',dependency_consistency)
    output=ROOT/args.output;output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(results=results,all_checks_passed=all(r['status']=='pass' for r in results)),ensure_ascii=False,indent=2),encoding='utf-8')
    if any(row['status']=='fail' for row in results): raise SystemExit(1)


if __name__=='__main__':main()
