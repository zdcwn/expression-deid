"""Independent required-dataset metrics; never treat unknown CelebA IDs as Pins classes."""
from pathlib import Path
import json

import numpy as np
import torch

from expression_deid.config import ROOT
from expression_deid.models import load_checkpoint
from expression_deid.training.required_data import AUXILIARY_FOLDERS, verify_prepared
from expression_deid.training.required_training import (
    PreparedDataset,anonymous_cache,classification_metrics,face_only,file_digest,loader,normalize,setup,
)


def evaluate_required(*,identity_checkpoint,anonymizer_checkpoint,expression_checkpoint,split='valid',device='auto',
                      output='outputs/required_training/evaluation.json',batch_size=128,full_photos=False,
                      data_root=None):
    if split not in {'valid','test'}: raise ValueError('Evaluation requires an independent held-out partition')
    target=setup(42,device)
    identity,id_meta=load_checkpoint(identity_checkpoint,device=target)
    anonymizer,anon_meta=load_checkpoint(anonymizer_checkpoint,device=target)
    from expression_deid.runtime.restored_expression import MainExpressionPredictor
    expression=MainExpressionPredictor(str(target),checkpoint=expression_checkpoint)
    expr_meta=expression.metadata
    for metadata,name in [(id_meta,'pins'),(anon_meta,'celebahq')]:
        if metadata['metrics'].get('datasets',{}).get('primary')!=name: raise ValueError('Wrong primary dataset in checkpoint')
        if metadata['metrics'].get('smoke_only'): raise ValueError('Smoke checkpoints are not valid accuracy experiments')
    if (expr_meta.get('metrics',{}).get('anonymizer_sha256')
            and expr_meta['metrics']['anonymizer_sha256']!=file_digest(anonymizer_checkpoint)):
        raise ValueError('Expression and anonymization checkpoints do not share the same image contract')
    if anon_meta['metrics']['identity_checkpoint_sha256']!=file_digest(identity_checkpoint):
        raise ValueError('Identity diagnostics must use the same frozen Pins recognizer as anonymization training')
    data_root=Path(data_root or ROOT/'datasets')
    pins=data_root/'pins/prepared'; fer=data_root/'fer2013/prepared'
    verify_prepared(pins); verify_prepared(fer)
    pins_loader=loader(PreparedDataset(pins,split),batch_size,0)
    original=classification_metrics(identity,pins_loader,target,id_meta['metrics']['normalization'],identity=True)
    correct=seen=original_correct=success=0; posterior=[]; changed=0; exterior_correct=True
    with torch.inference_mode():
        for x,y in pins_loader:
            x=x.to(target); y=y.to(target)
            raw=identity(normalize(face_only(x),id_meta['metrics']['normalization'])).float()
            anonymous=anonymizer(x)
            exterior=(anonymizer.face_mask==0).expand_as(x)
            exterior_correct=exterior_correct and torch.equal(x[exterior],anonymous[exterior])
            # Quantize the actual saved pixels before identity evaluation.
            anonymous=anonymous.clamp(0,1).mul(255).round()/255
            logits=identity(normalize(face_only(anonymous),id_meta['metrics']['normalization'])).float()
            raw_ok=raw.argmax(1)==y; private_ok=logits.argmax(1)==y
            original_correct+=int(raw_ok.sum()); success+=int((raw_ok&~private_ok).sum())
            correct+=int(private_ok.sum()); seen+=len(y); changed+=int((raw.argmax(1)!=logits.argmax(1)).sum())
            posterior.extend(logits.softmax(1).gather(1,y[:,None]).squeeze(1).cpu().tolist())
    images,provenance=anonymous_cache(fer,anonymizer_checkpoint,target,batch_size)
    expression_result=evaluate_expression_pixels(expression,fer,images,split,batch_size)
    result={'split':split,'identity_original':original,
            'identity_anonymized':{'accuracy':correct/seen,'samples':seen,'mean_true_identity_probability':float(np.mean(posterior)),
                'prediction_change_rate':changed/seen,'suppression_on_originally_correct':success/original_correct if original_correct else None,
                'scope':'project Pins classifier; aligned face oval only'},
            'expression_anonymized':expression_result,'expression_input_source':provenance['expression_input_source'],
            'background_unchanged':exterior_correct,'anonymizer_training_dataset':'celebahq',
            'checkpoints':{k:file_digest(v) for k,v in [('identity',identity_checkpoint),('anonymizer',anonymizer_checkpoint),('expression',expression_checkpoint)]}}
    auxiliary=expr_meta.get('metrics',{}).get('datasets',{}).get('auxiliary',[])
    result['expression_scheme']=expr_meta.get('metrics',{}).get('recipe','compact')
    result['expression_weights_retrained']=False
    result['evaluation_scope']='Aligned residual-component diagnostic; this input domain does not replace the full desktop runtime expression evaluation'
    if auxiliary:
        if len(auxiliary)!=1 or auxiliary[0] not in AUXILIARY_FOLDERS:
            raise ValueError('The expression checkpoint does not select a single supported auxiliary')
        aux=data_root/Path(AUXILIARY_FOLDERS[auxiliary[0]]).relative_to('datasets')
        verify_prepared(aux)
        values,_=anonymous_cache(aux,anonymizer_checkpoint,target,batch_size)
        result['expression_auxiliary']=evaluate_expression_pixels(expression,aux,values,split,batch_size)
    if full_photos:
        result['identity_full_photos']=evaluate_full_photos(identity_checkpoint,anonymizer_checkpoint,
            expression_checkpoint,split=split,device=str(target),data_root=data_root)
    target_file=Path(output); target_file.parent.mkdir(parents=True,exist_ok=True)
    target_file.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


def evaluate_expression_pixels(predictor,folder,images,split,batch_size):
    labels=np.load(Path(folder)/'labels.npy',allow_pickle=False)
    partitions=np.load(Path(folder)/'splits.npy',allow_pickle=False)
    selected=np.flatnonzero(partitions==split)
    probabilities=[]
    for offset in range(0,len(selected),batch_size):
        batch=selected[offset:offset+batch_size]
        probabilities.append(predictor.predict_anonymous_batch(np.asarray(images[batch]),input_source='anonymized_only'))
    probability=np.concatenate(probabilities); truth=labels[selected]; guess=probability.argmax(1)
    confusion=np.zeros((7,7),dtype=np.int64); np.add.at(confusion,(truth,guess),1)
    recall=np.diag(confusion)/np.maximum(confusion.sum(1),1)
    precision=np.diag(confusion)/np.maximum(confusion.sum(0),1)
    f1=2*precision*recall/np.maximum(precision+recall,1e-12)
    return {'accuracy':float((truth==guess).mean()),'balanced_accuracy':float(recall.mean()),
            'macro_f1':float(f1.mean()),'samples':len(selected),'confusion_matrix':confusion.tolist(),
            'by_class':{str(i):{'samples':int(confusion[i].sum()),'accuracy':float(recall[i])} for i in range(7)}}


def evaluate_full_photos(identity_checkpoint,anonymizer_checkpoint,expression_checkpoint,*,split,device,data_root=None):
    """Exercise detection, inverse composition and re-alignment on held-out Pins photos."""
    from expression_deid.runtime.required_pipeline import RequiredPipeline
    pipeline=RequiredPipeline(anonymizer_checkpoint,expression_checkpoint,identity_checkpoint,device=device)
    folder=Path(data_root or ROOT/'datasets')/'pins'
    records=[json.loads(s) for s in (folder/'prepared/manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    selected=[r for r in records if r['split']==split]
    original_correct=anonymous_correct=protected=0
    failures=[]
    try:
        for row in selected:
            relative=Path(row['source'])
            if relative.is_absolute() or relative.drive or '..' in relative.parts:
                raise ValueError('Invalid Pins source path')
            try:
                prediction=pipeline.process(folder/'raw'/relative,diagnostics=True)
            except (OSError,ValueError) as exc:
                failures.append({'source':row['source'],'error':str(exc)})
                continue
            original_correct+=int(prediction['identity_original']['label']==row['label'])
            anonymous_correct+=int(prediction['identity_anonymized']['label']==row['label'])
            protected+=int(prediction['protected_pixels_unchanged'])
    finally:
        pipeline.close()
    processed=len(selected)-len(failures)
    return {'selected':len(selected),'processed':processed,'failed':len(failures),'failures':failures,
            'original_accuracy':original_correct/processed if processed else None,
            'anonymized_accuracy':anonymous_correct/processed if processed else None,
            'protected_pixels_unchanged':processed==protected and processed>0,
            'scope':'project Pins classifier; full source photo -> anonymous composite -> re-aligned face oval'}


def build_parser(subparsers):
    p=subparsers.add_parser('evaluate',help='Aligned residual-component diagnostics; use check_current_expression.py for application expression scores')
    p.add_argument('--identity-checkpoint',required=True); p.add_argument('--anonymizer-checkpoint',required=True)
    p.add_argument('--expression-checkpoint',required=True); p.add_argument('--split',choices=['valid','test'],default='valid')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--output',default='outputs/required_training/evaluation.json'); p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--full-photos',action='store_true',help='Also verify full Pins photos through detection, composition and re-alignment')
    p.add_argument('--data-root',type=Path,help='Root containing prepared Pins and FER-2013 datasets')
    p.set_defaults(func=lambda a:evaluate_required(**{k:v for k,v in vars(a).items() if k not in {'func','command'}}))
