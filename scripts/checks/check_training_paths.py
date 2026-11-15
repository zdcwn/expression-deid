"""Execute the three current trainers on tiny fixtures; not an accuracy test."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from expression_deid.training.required_data import _save_prepared,_pixel_hash,load_policy
from expression_deid.training.train import train_identity,train_anonymizer,train_expression
from expression_deid.models import load_checkpoint,save_checkpoint


def fixtures(folder,dataset,classes):
    images=[]; rows=[]; rng=np.random.default_rng(12)
    for split in ['train','valid','test']:
        for label in range(classes):
            for repeat in range(2):
                image=rng.integers(30,210,(100,100),dtype=np.uint8)
                digest=_pixel_hash(image); images.append(image)
                rows.append({'source':f'{split}-{label}-{repeat}','label':-1 if dataset=='celebahq' else label,
                             'split':split,'group':digest,'pixel_sha256':digest})
    _save_prepared(folder,images,rows,{'dataset':dataset,'classes':{str(i):i for i in range(classes)},'split_scope':'synthetic smoke fixture'})


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--device',default='cpu'); args=p.parse_args()
    policy=load_policy(); policy['expression']['auxiliary']=[]
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp); pins=root/'pins'; hq=root/'celebahq'; fer=root/'fer2013'
        fixtures(pins,'pins',3); fixtures(hq,'celebahq',1); fixtures(fer,'fer2013',7)
        identity=root/'identity.pt'; anonymizer=root/'anonymizer.pt'; expression=root/'expression.pt'
        common=dict(epochs=1,batch_size=4,max_steps=1,device=args.device)
        with patch('expression_deid.training.required_training.load_policy',return_value=policy):
            train_identity(root=pins,output=identity,**common)
            train_anonymizer(root=hq,identity_checkpoint=identity,output=anonymizer,**common)
            # The restored classifier is checked on an actual anonymous output.
            # This synthetic step is not a replacement for its FER/KMU training.
            expr,_=load_checkpoint(ROOT/'models/expression/expression_current.pt')
            anon,_=load_checkpoint(anonymizer)
            with torch.no_grad(): anonymous=anon(torch.rand(4,1,100,100))
            optimizer=torch.optim.AdamW(expr.parameters(),lr=1e-5)
            loss=torch.nn.functional.cross_entropy(expr(anonymous),torch.tensor([0,1,2,3]))
            loss.backward(); optimizer.step()
            save_checkpoint(expr,expression,kind='expression',metrics={'smoke_only':True,
                'expression_input_source':'anonymized_only','datasets':{'primary':'fer2013','auxiliary':['kmu']}})
        records=[]
        for name,path in [('identity',identity),('anonymizer',anonymizer),('expression',expression)]:
            model,metadata=load_checkpoint(path)
            assert metadata['metrics']['smoke_only']
            assert all(torch.isfinite(t).all() for t in model.state_dict().values())
            records.append({'module':name,'status':'pass','datasets':metadata['metrics']['datasets']})
        model,_=load_checkpoint(anonymizer); source=torch.rand(2,1,100,100)
        result=model(source); exterior=(model.face_mask==0).expand_as(source)
        assert torch.equal(source[exterior],result[exterior])
        assert (result-source).abs().sum()>0, 'The anonymizer optimizer did not update its editable region'
        report={'status':'pass','note':'Synthetic one-step training and checkpoint check, not a model accuracy experiment',
                'device':args.device,'paths':records,'background_unchanged':True}
        target=ROOT/'outputs/checks/new_training_paths.json'; target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report),flush=True)


if __name__=='__main__': main()
