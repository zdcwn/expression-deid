"""Execute every training entry on tiny anonymous fixtures, not a benchmark."""
import json
import argparse
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
import torch
import expression_deid.normalize as normal
from expression_deid.train import train_expression,train_identity,train_anonymizer,train_expression_kmu,train_expression_joint
from expression_deid.models import load_checkpoint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',default=str(ROOT/'outputs/realtime_training/anonymous_cache.npz'))
    args=parser.parse_args()
    torch.set_num_threads(1)
    cache=np.load(args.cache)
    if str(cache['privacy_source'])!='anonymized_only' or cache['images'].dtype!=np.uint8:
        raise ValueError('训练检查只允许匿名 uint8 缓存')
    pictures={label:cache['images'][np.flatnonzero((cache['labels']==label)&(cache['splits']=='train'))[0]] for label in range(7)}
    rows=[]
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);raf=root/'raf';kmu=root/'kmu';ppb=root/'ppb';gan=root/'gan'
        for split,count in (('train',3),('valid',1)):
            for label in range(7):
                target=raf/split/str(label);target.mkdir(parents=True)
                for i in range(count): Image.fromarray(pictures[label]).save(target/f'{i}.jpg')
        kmu.mkdir();records=[];gan.mkdir();generated=[]
        for person in ('01','02'):
            for code,label in (('AN',5),('DI',2),('HA',3)):
                for frame in range(1,5):
                    stem=f'{person}_{code}_anon_{frame:03d}';name=stem+'.png'
                    Image.fromarray(pictures[label]).save(kmu/name)
                    record=dict(source=stem+'.jpg',output=name,person=person,expression=code,frame=frame,label=label,status='ok')
                    records.append(record)
                    Image.fromarray(pictures[label]).save(gan/name)
                    generated.append({**record,'domain':'kmu','split':'train'})
        # Intentionally keep original validation sources in an old GAN train
        # manifest; train_expression_joint must remove them before sampling.
        (kmu/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        (gan/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in generated))
        ppb.mkdir();records=[]
        for person in ('01','02','03','04','05'):
            for frame in range(2):
                name=f'ppb_{person}_{frame}.png';Image.fromarray(pictures[6]).save(ppb/name)
                records.append(dict(source=name,output=name,person=person,frame=frame,label=6,status='ok'))
        (ppb/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        def check(name,callback,path):
            result=callback();model,meta=load_checkpoint(path)
            assert meta['epoch']==1
            assert all(torch.isfinite(v).all() for v in model.state_dict().values())
            rows.append(dict(path=name,status='pass',epoch=meta['epoch'],metrics=meta['metrics']))
        with patch.object(normal,'RAF_DIR',raf):
            expression=root/'expression.pt';identity=root/'identity.pt';anonymizer=root/'anonymizer.pt';adapted=root/'adapted.pt';joint=root/'joint.pt'
            check('train-expression',lambda:train_expression(output=str(expression),epochs=1,batch_size=8,device='cpu'),expression)
            check('train-identity',lambda:train_identity(root=kmu,output=str(identity),epochs=1,batch_size=8,device='cpu'),identity)
            check('train-anonymizer',lambda:train_anonymizer(root=kmu,output=str(anonymizer),identity_checkpoint=str(identity),expression_checkpoint=str(expression),epochs=1,batch_size=8,device='cpu'),anonymizer)
            assert rows[-1]['metrics']['expression_input_source']=='anonymized_only'
            assert rows[-1]['metrics']['train_samples']==18 and rows[-1]['metrics']['valid_samples']==6
            check('train-expression-kmu',lambda:train_expression_kmu(root=kmu,base_checkpoint=str(expression),output=str(adapted),epochs=1,batch_size=8,device='cpu'),adapted)
            check('train-expression-joint',lambda:train_expression_joint(root=kmu,ppb_root=ppb,gan_root=gan,gan_fraction=.1,ferplus_root=root/'absent-fer',ferplus_fraction=0,ppb_fraction=.1,kmu_fraction=.25,base_checkpoint=str(expression),output=str(joint),epochs=1,batch_size=8,device='cpu'),joint)
    (ROOT/'outputs/full_audit_20261008/training_paths.json').write_text(json.dumps(dict(note='One-epoch execution and gradient smoke on tiny anonymous fixtures; accuracy is not a benchmark.',rows=rows),indent=2))
    print('all five training paths passed',flush=True)

if __name__=='__main__':main()
