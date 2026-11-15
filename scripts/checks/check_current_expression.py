"""Re-evaluate the fixed self-trained checkpoint on complete held-out domains."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch
from expression_deid.models import load_checkpoint
from expression_deid.training.required_data import verify_prepared
from expression_deid.training.required_training import file_digest


def summarize(truth,prediction):
    cm=np.zeros((7,7),np.int64); np.add.at(cm,(truth,prediction),1)
    counts=cm.sum(1); recall=np.diag(cm)/np.maximum(counts,1)
    precision=np.diag(cm)/np.maximum(cm.sum(0),1)
    f1=2*precision*recall/np.maximum(precision+recall,1e-12)
    present=counts>0
    return {'samples':len(truth),'accuracy':float((truth==prediction).mean()),
            'balanced_accuracy':float(recall[present].mean()),'macro_f1':float(f1[present].mean()),
            'confusion':cm.tolist(),'by_class':{str(i):{'samples':int(counts[i]),'accuracy':float(recall[i])}
                                             for i in np.flatnonzero(present)}}


def check(data_root,output,device,checkpoint=None,domains=('fer2013','kmu','raf')):
    torch.set_num_threads(2)
    checkpoint=Path(checkpoint or ROOT/'models/expression/expression_current.pt')
    model,metadata=load_checkpoint(checkpoint,device=device); spec=metadata['metrics']
    anonymizer=ROOT/'models/anonymization/anonymizer_celebahq.pt'
    if spec.get('datasets',{}).get('primary') != 'fer2013':
        raise ValueError('Evaluate a FER-2013-primary checkpoint; a RAF/KMU checkpoint cannot stand in for it')
    if spec.get('expression_input_source')!='anonymized_only' or spec['anonymizer_sha256']!=file_digest(anonymizer):
        raise ValueError('The restored expression checkpoint does not match the frozen anonymizer')
    report={'checkpoint_sha256':file_digest(checkpoint),'anonymizer_sha256':file_digest(anonymizer),
            'expression_input_source':'anonymized_only','checkpoint_fixed_before_evaluation':True,
            'training_datasets':spec['datasets'], 'recipe':spec.get('recipe'),
            'selection_scope':'The checkpoint is fixed before comparing complete dataset partitions',
            'initialization_checkpoint_sha256':spec.get('parent_checkpoint_sha256'),
            'splits':{},'runtime_cache_equivalence':{}}
    pipeline=None
    try:
        from expression_deid.runtime.pipeline import ExpressionDeidentifier
        pipeline=ExpressionDeidentifier(anonymizer,checkpoint,device=device,fast_mode=False)
        for domain in domains:
            relative={'fer2013':'fer2013/prepared','raf':'raf-auxiliary/prepared','kmu':'kmu-auxiliary/prepared'}[domain]
            folder=Path(data_root)/relative; verify_prepared(folder)
            manifest=json.loads((folder/'anonymous-runtime.json').read_text(encoding='utf-8'))
            for key,name in [('source_images_sha256','images.npy'),('labels_sha256','labels.npy'),
                             ('splits_sha256','splits.npy'),('output_sha256','anonymous-runtime.npy')]:
                if manifest[key]!=file_digest(folder/name): raise ValueError('Anonymous cache digest mismatch: '+name)
            if manifest['anonymizer_sha256']!=report['anonymizer_sha256']: raise ValueError('Wrong anonymous cache producer')
            images=np.load(folder/'anonymous-runtime.npy',mmap_mode='r'); labels=np.load(folder/'labels.npy'); partitions=np.load(folder/'splits.npy')
            source_images=np.load(folder/'images.npy',mmap_mode='r')
            # Reproduce selected cache entries through the same restored app
            # anonymizer; the classifier is never called on source pixels.
            selected=np.flatnonzero(partitions=='test')
            probes=np.unique(np.linspace(0,len(selected)-1,min(14,len(selected)),dtype=int))
            exact=0; max_error=0
            for position in probes:
                i=selected[position]
                pipeline.anonymize(np.asarray(source_images[i]),aligned=True,purpose='machine')
                actual=pipeline._last_anonymized_aligned
                error=int(np.abs(actual.astype(np.int16)-images[i].astype(np.int16)).max())
                exact+=error==0; max_error=max(max_error,error)
            report['runtime_cache_equivalence'][domain]={'probes':len(probes),'exact':exact,'max_pixel_error':max_error}
            if max_error: raise ValueError('Cache differs from the restored runtime: '+domain)
            for split in ('valid','test'):
                indices=np.flatnonzero(partitions==split); predictions=[]
                with torch.inference_mode():
                    for start in range(0,len(indices),128):
                        ids=indices[start:start+128]
                        x=torch.as_tensor(np.asarray(images[ids]),device=device,dtype=torch.float32)[:,None]/255
                        logits=model((x-float(spec['normalization']['mean']))/float(spec['normalization']['std']))
                        predictions.extend(logits.argmax(1).cpu().tolist())
                report['splits'][split+'_'+domain]=summarize(labels[indices],np.asarray(predictions))
            print(json.dumps({'domain':domain,'test':report['splits']['test_'+domain],
                              'cache_equivalence':report['runtime_cache_equivalence'][domain]}),flush=True)
    finally:
        if pipeline is not None: pipeline.close()
    candidates={key:value for key,value in report['splits'].items() if key.startswith('valid_')}
    best=max(candidates,key=lambda key:candidates[key]['accuracy'])
    report['best_validation_domain']={'split':best,**candidates[best],
        'scope':'Highest full validation-domain accuracy; the same frozen model is used for every domain'}
    report['evaluation_scope']='FER-2013 primary training, KMU auxiliary. Six-class KMU and seven-class FER/RAF remain separate. Initial weights have earlier multi-domain training history; current-stage held-out results are regression measurements, not strict never-seen-person claims.'
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/verification/expression-validation.json')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--domains',nargs='+',choices=('fer2013','kmu','raf'),default=['fer2013','kmu','raf'])
    args=parser.parse_args(); check(args.data_root,args.output,args.device,args.checkpoint,args.domains)
