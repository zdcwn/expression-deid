"""Export a locally trained accuracy model; verify the graph before deployment."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
from scripts.training.train_video_quality import load_backbone,digest,UPSTREAM_URL,UPSTREAM_SHA
from expression_deid.runtime.expression import anonymous_tensor
from expression_deid.labels import EXPRESSION_NAMES
from expression_deid.models import load_checkpoint
from expression_deid.preprocessing.normalize import load_stats

class QualityGraph(nn.Module):
    def __init__(self,backbone,head,mean,std):
        super().__init__();self.backbone=backbone;self.head=head
        self.register_buffer('mean',mean);self.register_buffer('std',std)
    def forward(self,x):return self.head((self.backbone(x)-self.mean)/self.std)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--onnx',type=Path,required=True)
    p.add_argument('--compact',type=Path,default=ROOT/'models/expression/expression_realtime_stable.pt')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=p.parse_args();torch.set_num_threads(1);args.output.mkdir(parents=True,exist_ok=True)
    payload=torch.load(args.checkpoint,map_location=args.device,weights_only=True)
    backbone=load_backbone(args.onnx,args.device);backbone.load_state_dict(payload['backbone']);backbone.eval()
    head=nn.Linear(1408,7,device=args.device);head.load_state_dict(payload['head']);head.eval()
    graph=QualityGraph(backbone,head,payload['mean'].to(args.device),payload['std'].to(args.device)).eval()
    fixture=np.random.default_rng(0).integers(0,256,(100,100),dtype=np.uint8)
    _,x=anonymous_tensor(fixture,device=args.device)
    with torch.inference_mode():
        traced=torch.jit.trace(graph,x,check_trace=False).eval()
        errors=[]
        for batch in (x[:1],x,x.repeat(2,1,1,1)):
            original=graph(batch);converted=traced(batch);err=float((original-converted).abs().max())
            if err>1e-4 or not torch.isfinite(converted).all() or not torch.equal(original.argmax(1),converted.argmax(1)):
                raise RuntimeError(f'导出图对照失败，最大误差 {err}')
            errors.append(err)
    target=args.output/'expression_video_quality.ts.pt';pending=target.with_suffix('.pending');torch.jit.save(traced,str(pending));pending.replace(target)
    compact,metadata=load_checkpoint(args.compact);del compact
    stats=load_stats();row=payload['row']
    compact_name=args.compact.name
    if args.compact.resolve().parent!=args.output.resolve():
        # Candidate bundles must be self-contained; identical existing assets
        # may be reused when exporting directly into the models directory.
        import shutil
        shutil.copy2(args.compact,args.output/compact_name)
    manifest=dict(schema=1,expression_input_source='anonymized_only',machine_privacy={},class_names=list(EXPRESSION_NAMES),
        preprocessing='float_bilinear_260_imagenet_gray_v1',quality_weight=row.get('quality_weight',.5),
        quality_temperature=row.get('quality_temperature',row.get('temperature',1.)),final_temperature=1.,
        compact_temperature=metadata['metrics']['calibration_temperature'],compact_mean=stats.mean,compact_std=stats.std,
        quality_model={'file':target.name,'sha256':digest(target)},compact_model={'file':compact_name,'sha256':digest(args.compact)},
        pretrained={'url':UPSTREAM_URL,'sha256':UPSTREAM_SHA,'repository':'https://github.com/sb-ai-lab/EmotiEffLib'},
        training_checkpoint_sha256=digest(args.checkpoint),validation=row,export_max_abs_errors=errors)
    (args.output/'expression_video_quality.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'model':str(target),'bytes':target.stat().st_size,'export_errors':errors,'selected':row}),flush=True)

if __name__=='__main__':main()
