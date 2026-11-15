"""FER-2013 primary training on the frozen anonymous runtime pixel domain."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from expression_deid.config import ROOT
from expression_deid.models import load_checkpoint, save_checkpoint
from expression_deid.training.required_data import load_policy, verify_prepared


CURRENT_CHECKPOINT = ROOT / 'models/expression/expression_current.pt'
ANONYMIZER_CHECKPOINT = ROOT / 'models/anonymization/anonymizer_celebahq.pt'
KMU_FOLDER = ROOT / 'datasets/kmu-auxiliary/prepared'
MEAN = 0.481619
STD = 0.242606


def digest(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load_domain(folder: Path, domain: str, cache_name: str, anonymizer_checkpoint: Path):
    cache = folder / cache_name
    if cache_name == 'images.npy' or Path(cache_name).name != cache_name:
        raise ValueError('Use a named anonymous cache within the prepared dataset')
    provenance = json.loads(cache.with_suffix('.json').read_text(encoding='utf-8'))
    if provenance.get('expression_input_source') != 'anonymized_only':
        raise ValueError('The expression classifier cannot train on source pixels')
    for key, path in [('source_images_sha256', folder/'images.npy'), ('labels_sha256', folder/'labels.npy'),
                      ('splits_sha256', folder/'splits.npy'), ('output_sha256', cache),
                      ('anonymizer_sha256', Path(anonymizer_checkpoint))]:
        if provenance.get(key) != digest(path):
            raise ValueError('Anonymous training cache provenance mismatch: '+key)
    images = np.load(folder / cache_name, mmap_mode='r', allow_pickle=False)
    labels = np.load(folder / 'labels.npy', allow_pickle=False)
    splits = np.load(folder / 'splits.npy', allow_pickle=False)
    metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    if metadata.get('dataset') != domain or images.shape[1:] != (100, 100):
        raise ValueError(f'Invalid anonymous {domain} cache')
    if images.dtype != np.uint8 or len(images) != len(labels) or len(images) != len(splits):
        raise ValueError(f'Invalid anonymous {domain} arrays')
    if not np.isin(labels, np.arange(7)).all():
        raise ValueError(f'Invalid {domain} labels')
    return images, labels.astype(np.int64), splits.astype(str), metadata


def weak_views(x: torch.Tensor, generator: torch.Generator):
    """Two light views that preserve expression geometry."""
    views = []
    for _ in range(2):
        view = x.clone()
        flip = torch.rand((), generator=generator, device='cpu') < .5
        if bool(flip):
            view = view.flip(-1)
        gain = torch.empty((len(view), 1, 1, 1), device=x.device).uniform_(.94, 1.06)
        offset = torch.empty_like(gain).uniform_(-.03, .03)
        views.append((view * gain + offset).clamp(0, 1))
    return views


@torch.inference_mode()
def evaluate(model, domains, device, split):
    model.eval()
    result = {}
    for name, (images, labels, splits, _metadata) in domains.items():
        indices = np.flatnonzero(splits == split)
        predictions = []
        for start in range(0, len(indices), 128):
            batch = torch.as_tensor(np.asarray(images[indices[start:start + 128]]), device=device,
                                    dtype=torch.float32)[:, None] / 255
            logits = model((batch - MEAN) / STD)
            predictions.extend(logits.argmax(1).cpu().numpy().tolist())
        truth = labels[indices]
        prediction = np.asarray(predictions)
        per_class = {}
        for label in sorted(set(truth.tolist())):
            mask = truth == label
            per_class[str(int(label))] = {
                'samples': int(mask.sum()),
                'accuracy': float((prediction[mask] == truth[mask]).mean()),
            }
        result[name] = {
            'samples': int(len(indices)),
            'accuracy': float((prediction == truth).mean()),
            'balanced_accuracy': float(np.mean([row['accuracy'] for row in per_class.values()])),
            'by_class': per_class,
        }
    return result


def sampling_weights(labels, domain_ids, primary_fraction, neutral_boost):
    if not .75 <= primary_fraction <= 1:
        raise ValueError('FER-2013 must supply at least 75 percent of training draws')
    weights = np.zeros(len(labels), dtype=np.float64)
    domains = sorted(set(domain_ids.tolist()))
    fractions = {0: 1.0} if len(domains) == 1 else {0: primary_fraction, 1: 1-primary_fraction}
    for domain, fraction in fractions.items():
        selected = domain_ids == domain
        counts = np.bincount(labels[selected], minlength=7).astype(float)
        class_weight = 1 / np.sqrt(np.maximum(counts[labels[selected]], 1))
        class_weight *= np.where(labels[selected] == 6, neutral_boost, 1.0)
        if not selected.any() or class_weight.sum() <= 0:
            raise ValueError('Empty training domain')
        weights[selected] = fraction * class_weight / class_weight.sum()
    return weights / weights.sum()


@torch.inference_mode()
def calibrate_validation(model, domains, device, primary_fraction):
    """Fit confidence scale on training-validation domains, never on test."""
    model.eval(); logits_by_domain = {}
    for name,(images,labels,splits,_) in domains.items():
        ids=np.flatnonzero(splits=='valid'); pieces=[]
        for start in range(0,len(ids),128):
            batch=torch.as_tensor(np.asarray(images[ids[start:start+128]]),device=device,dtype=torch.float32)[:,None]/255
            pieces.append(model((batch-MEAN)/STD).float())
        logits_by_domain[name]=(torch.cat(pieces),torch.as_tensor(labels[ids],device=device,dtype=torch.long))
    temperatures=sorted(set([1.0]+np.linspace(.75,2.5,71).tolist()))
    losses=[]
    for temperature in temperatures:
        total=0.0
        for name,(logits,truth) in logits_by_domain.items():
            weight=1.0 if len(domains)==1 else primary_fraction if name=='fer2013' else 1-primary_fraction
            total+=weight*float(F.cross_entropy(logits/temperature,truth))
        losses.append(total)
    best=int(np.argmin(losses))
    return {'temperature':float(temperatures[best]),'nll_before':losses[temperatures.index(1.0)],
            'nll_after':losses[best],'source':'FER-2013 and selected auxiliary validation only'}


def train(args):
    if not np.isfinite(args.neutral_boost) or args.neutral_boost <= 0:
        raise ValueError('Neutral sampling boost must be finite and positive')
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError('Invalid training settings')
    policy = load_policy()
    if policy['expression']['primary'] != 'fer2013':
        raise ValueError('FER-2013 is the required primary expression dataset')
    auxiliary = policy['expression'].get('auxiliary', [])
    if auxiliary not in ([], ['kmu']):
        raise ValueError('This compact recipe supports KMU as its only auxiliary')
    import random
    random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    output = Path(args.output).resolve()
    if output.is_relative_to((ROOT/'models').resolve()) and not output.is_relative_to((ROOT/'models/training').resolve()):
        raise ValueError('Training must not overwrite deployed model weights')
    output.mkdir(parents=True, exist_ok=True)
    initialization = Path(args.initial_checkpoint).resolve()
    model, parent = load_checkpoint(initialization, device=device)
    parent_spec = parent.get('metrics', {})
    if (parent.get('kind') != 'expression' or parent_spec.get('expression_input_source') != 'anonymized_only'
            or parent_spec.get('datasets', {}).get('primary') != 'fer2013' or parent_spec.get('smoke_only')):
        raise ValueError('Initialize from a verified FER-2013 expression checkpoint; RAF-only weights are not a FER restoration')
    data_root = Path(args.data_root)
    paths = {'fer2013': data_root/'fer2013/prepared'}
    if auxiliary: paths['kmu'] = data_root/'kmu-auxiliary/prepared'
    domains = {}
    for name, folder in paths.items():
        verify_prepared(folder)
        domains[name] = load_domain(folder, name, args.cache_name, Path(args.anonymizer_checkpoint))
    train_rows = [(name, domain, int(i), int(values[1][i]))
                  for domain, (name, values) in enumerate(domains.items())
                  for i in np.flatnonzero(values[2] == 'train')]
    labels = np.asarray([row[3] for row in train_rows], np.int64)
    domain_ids = np.asarray([row[1] for row in train_rows], np.int64)
    fraction = args.primary_fraction
    weights = sampling_weights(labels, domain_ids, fraction, args.neutral_boost)
    generator = torch.Generator(device='cpu').manual_seed(args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr*.03)
    scaler = torch.amp.GradScaler('cuda', init_scale=512, enabled=device.type=='cuda')
    initialization_digest = digest(initialization)
    anonymizer_digest = digest(Path(args.anonymizer_checkpoint))
    data_spec = {'primary': 'fer2013', 'auxiliary': auxiliary, 'primary_sampling_fraction': fraction if auxiliary else 1.0}
    baseline = evaluate(model, domains, device, 'valid')
    best = -float('inf'); stalled = 0; history = []
    run = {'recipe':'runtime_domain_fer2013_expression', 'datasets':data_spec,
           'initialization_checkpoint_sha256': initialization_digest,
           'anonymizer_sha256': anonymizer_digest, 'expression_input_source':'anonymized_only',
           'selection_scope':'FER-2013 and selected auxiliary validation only; external test sets never select epochs',
           'baseline_valid':baseline,'history':history}
    (output/'training.json').write_text(json.dumps(run,ensure_ascii=False,indent=2),encoding='utf-8')
    for epoch in range(1,args.epochs+1):
        started=time.perf_counter(); model.train(); total=correct=seen=0
        chosen=torch.multinomial(torch.as_tensor(weights,dtype=torch.double),len(train_rows),
                                 replacement=True,generator=generator).tolist()
        for start in range(0,len(chosen),args.batch_size):
            rows=[train_rows[i] for i in chosen[start:start+args.batch_size]]
            batch=np.asarray([domains[name][0][i] for name,_,i,_ in rows])
            x=torch.as_tensor(batch,device=device,dtype=torch.float32)[:,None]/255
            y=torch.as_tensor([label for _,_,_,label in rows],device=device,dtype=torch.long)
            first,second=weak_views(x,generator)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type,enabled=device.type=='cuda'):
                a=model((first-MEAN)/STD); b=model((second-MEAN)/STD)
                loss=F.cross_entropy(a,y,label_smoothing=.03)+args.consistency_weight*F.mse_loss(a.softmax(1),b.softmax(1).detach())
            if not torch.isfinite(loss): raise ValueError('Non-finite expression loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.0)
            scaler.step(optimizer);scaler.update()
            seen+=len(y);total+=float(loss.detach())*len(y);correct+=int((a.argmax(1)==y).sum())
        valid=evaluate(model,domains,device,'valid')
        score=fraction*valid['fer2013']['accuracy']+(1-fraction)*valid['kmu']['accuracy'] if auxiliary else valid['fer2013']['accuracy']
        row={'epoch':epoch,'train_loss':total/seen,'train_accuracy':correct/seen,'valid':valid,
             'selection_score':score,'seconds':time.perf_counter()-started,'learning_rate':optimizer.param_groups[0]['lr']}
        metadata={**row,'datasets':data_spec,'recipe':'runtime_domain_fer2013_expression',
                  'expression_input_source':'anonymized_only','anonymizer_sha256':anonymizer_digest,
                  'parent_checkpoint_sha256':initialization_digest,
                  'normalization':{'mean':MEAN,'std':STD,'channels':1,'space':'gray_0_1','source':'fixed runtime gray normalization'},
                  'machine_privacy':{},'consistency_weight':args.consistency_weight,
                  'neutral_sampling_boost':args.neutral_boost,'smoke_only':False,
                  'preprocessing':'local component edit + frozen learned residual + final composite and realignment'}
        if score > best+args.min_delta:
            best=score;stalled=0;save_checkpoint(model,output/'best.pt',kind='expression',epoch=epoch,metrics=metadata)
        else:stalled+=1
        history.append(row);run['best_valid_score']=best;run['completed_epochs']=epoch
        (output/'training.json').write_text(json.dumps(run,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(row,ensure_ascii=False),flush=True);scheduler.step()
        if epoch>=args.min_epochs and stalled>=args.patience:
            run['stop_reason']='No meaningful improvement on the fixed validation criterion';break
    run['status']='complete'
    model,payload=load_checkpoint(output/'best.pt',device=device)
    calibration=calibrate_validation(model,domains,device,fraction)
    payload['metrics']['calibration_temperature']=calibration['temperature']
    payload['metrics']['calibration_source']=calibration['source']
    save_checkpoint(model,output/'best.pt',kind='expression',epoch=payload['epoch'],metrics=payload['metrics'])
    run['calibration']=calibration
    (output/'training.json').write_text(json.dumps(run,ensure_ascii=False,indent=2),encoding='utf-8')
    return {'checkpoint':str(output/'best.pt'),'best_valid_score':best,'datasets':data_spec,
            'test_used_for_selection':False,'device':str(device)}


def build_parser(subparsers):
    p=subparsers.add_parser('train-expression',help='Train the compact expression model primarily on anonymous FER-2013')
    p.add_argument('--output',default='models/training/expression_fer2013')
    p.add_argument('--data-root',default=str(ROOT/'datasets'))
    p.add_argument('--anonymizer-checkpoint',default=str(ANONYMIZER_CHECKPOINT))
    p.add_argument('--cache-name',default='anonymous-runtime.npy')
    p.add_argument('--initial-checkpoint',default=str(CURRENT_CHECKPOINT))
    p.add_argument('--epochs',type=int,default=24);p.add_argument('--min-epochs',type=int,default=8)
    p.add_argument('--batch-size',type=int,default=128);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--weight-decay',type=float,default=1e-4);p.add_argument('--patience',type=int,default=5)
    p.add_argument('--min-delta',type=float,default=.001)
    p.add_argument('--primary-fraction',type=float,default=load_policy()['expression'].get('primary_sampling_fraction',.85))
    p.add_argument('--consistency-weight',type=float,default=.3);p.add_argument('--neutral-boost',type=float,default=1.2)
    p.add_argument('--seed',type=int,default=21);p.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    p.set_defaults(func=train)
