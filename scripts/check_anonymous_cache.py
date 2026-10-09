"""Compare every cached classifier input with the real video runner.

The aligned templates are controlled detector fixtures. This verifies pixel
domain equivalence, not detection accuracy or expression quality.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from expression_deid.app import build_pipeline
from expression_deid.normalize import load_gray
from expression_deid.align import template_for
from expression_deid.video_demo import FastVideoRunner


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--limit',type=int,default=0,help='0 检查全部缓存输入')
    args=parser.parse_args(); torch.set_num_threads(1)
    with np.load(args.cache) as archive:
        data={key:archive[key] for key in archive.files}
    signature=json.loads(str(data['signature']))
    pipeline=build_pipeline('cuda',use_gan=False,expression_checkpoint=args.checkpoint)
    pipeline.machine_privacy=signature['machine_privacy']
    captured=[]
    def capture(image,aligned):
        captured.append(image.copy())
        return dict(name='neutral',confidence=1.,probabilities=[0,0,0,0,0,0,1])
    pipeline.predict_expression=capture
    checks=[]
    indices=range(len(data['images'])) if not args.limit else np.linspace(0,len(data['images'])-1,args.limit,dtype=int)
    try:
        for index in indices:
            gray=load_gray(ROOT/str(data['sources'][index]));size=gray.shape[0]
            face=np.zeros(15);face[:4]=(0,0,size,size);face[4:14]=template_for(size).ravel();face[14]=1.
            class Detector:
                def detect(self,rgb): return face[None].copy()
            pipeline.detector=Detector()
            runner=FastVideoRunner(pipeline)
            captured.clear()
            runner.process_frame(np.repeat(gray[:,:,None],3,axis=2),0,timestamp=0.)
            assert len(captured)==1
            difference=int(np.abs(captured[0].astype(int)-data['images'][index].astype(int)).max())
            checks.append(dict(source=str(data['sources'][index]),max_pixel_diff=difference))
            if (index+1)%500==0: print(f'input equivalence {index+1}/{len(data["images"])}',flush=True)
    finally: pipeline.close()
    result=dict(checked=len(checks),total=len(data['images']),max_pixel_difference=max(r['max_pixel_diff'] for r in checks),
                all_inputs_checked=len(checks)==len(data['images']),cache_signature=signature,rows=checks)
    Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print({key:value for key,value in result.items() if key not in ('rows','cache_signature')},flush=True)
    if result['max_pixel_difference']>1: raise SystemExit(1)


if __name__=='__main__':main()
