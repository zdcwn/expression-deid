"""Verify the exact per-frame accuracy backend against the deployed compact model."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from expression_deid.runtime.expression import QualityExpressionPredictor
from expression_deid.models import load_checkpoint
from expression_deid.preprocessing.normalize import load_stats
from scripts.training.train_video_quality import validate_cache,summarize,digest


def calibration(probability,labels):
    confidence=probability.max(1);prediction=probability.argmax(1);correct=prediction==labels
    nll=-np.log(probability[np.arange(len(labels)),labels].clip(1e-12)).mean()
    ece=0.
    for low in np.linspace(0,1,16)[:-1]:
        mask=(confidence>low)&(confidence<=low+1/15)
        if mask.any():ece+=mask.mean()*abs(correct[mask].mean()-confidence[mask].mean())
    return dict(nll=float(nll),ece_15_bins=float(ece))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=ROOT/'models/expression/expression_video_quality.json')
    p.add_argument('--anonymous-cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda');p.add_argument('--calibrate',action='store_true')
    args=p.parse_args();torch.set_num_threads(1);data=np.load(args.anonymous_cache);validate_cache(data)
    model=QualityExpressionPredictor(args.device,manifest=args.manifest);meta=model.metadata
    compact,payload=load_checkpoint(args.manifest.parent/meta['compact_model']['file'],device=args.device);stats=load_stats()
    selected=np.flatnonzero(data['splits']!='train');p0=np.zeros((len(data['images']),7),np.float32);p1=np.zeros_like(p0);durations=[]
    # Warmup is excluded from timing and does not consume a source frame.
    for _ in range(10):model.predict_anonymous(np.zeros((100,100),np.uint8),input_source='anonymized_only')
    start=time.perf_counter()
    with torch.inference_mode():
        for offset in range(0,len(selected),64):
            batch_indices=selected[offset:offset+64];images=data['images'][batch_indices]
            x=torch.tensor(images,device=args.device,dtype=torch.float32)[:,None]/255
            p0[batch_indices]=((compact((x-stats.mean)/stats.std)/payload['metrics']['calibration_temperature']).softmax(1).cpu().numpy())
            at=time.perf_counter();p1[batch_indices]=model.predict_anonymous_batch(images,input_source='anonymized_only');elapsed=time.perf_counter()-at
            durations.extend([elapsed/len(batch_indices)]*len(batch_indices))
            if (offset+len(batch_indices))%500==0 or offset+len(batch_indices)==len(selected):print('checked',offset+len(batch_indices),len(selected),round(time.perf_counter()-start,1),flush=True)
    valid=data['splits']=='valid';temperature=1.
    if args.calibrate:
        # Calibration is global, not a per-class neutral preference. Test
        # labels are excluded; positive temperature cannot change the argmax.
        logits=torch.tensor(np.log(p1[valid].clip(1e-12)));labels=torch.tensor(data['labels'][valid],dtype=torch.long)
        value=torch.zeros((),requires_grad=True);opt=torch.optim.LBFGS([value],lr=.1,max_iter=50)
        def closure():
            opt.zero_grad();loss=torch.nn.functional.cross_entropy(logits/value.exp().clamp(.5,5),labels);loss.backward();return loss
        opt.step(closure);temperature=float(value.detach().exp().clamp(.5,5))
        recalibrated=np.zeros_like(p1);l=np.log(p1[selected].clip(1e-12))/temperature;e=np.exp(l-l.max(1,keepdims=True));recalibrated[selected]=e/e.sum(1,keepdims=True)
        assert np.array_equal(p1[selected].argmax(1),recalibrated[selected].argmax(1))
        meta['final_temperature']*=temperature
        args.manifest.write_text(json.dumps(meta,indent=2),encoding='utf-8')
        before_calibration=calibration(p1[valid],data['labels'][valid]);p1=recalibrated
    else:before_calibration=calibration(p1[valid],data['labels'][valid])
    report=dict(expression_input_source='anonymized_only',original_classifier_calls=0,manifest=str(args.manifest),
        model_sha256=meta['quality_model']['sha256'],cache_sha256=digest(args.anonymous_cache),
        baseline={'valid':summarize(p0,data),'test':summarize(p0,data,'test')},
        quality={'valid':summarize(p1,data),'test':summarize(p1,data,'test')},
        calibration=dict(relative_temperature=temperature,before=before_calibration,after=calibration(p1[valid],data['labels'][valid])),
        classifier_only_fps=len(selected)/sum(durations),classifier_ms_p95=float(np.quantile(durations,.95)*1000),
        scope='RAF validation used in selection; KMU adjacent-tail regression; PPB unseen-participant neutral only; not camera accuracy')
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    np.savez_compressed(args.output/'predictions.npz',indices=selected,baseline=p0[selected],quality=p1[selected],labels=data['labels'][selected],domains=data['domains'][selected],splits=data['splits'][selected])
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
