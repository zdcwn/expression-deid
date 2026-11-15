"""One training workflow: Pins identity -> CelebA-HQ edits -> anonymous FER-2013."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from expression_deid.config import ROOT
from expression_deid.models import load_checkpoint, model_class, save_checkpoint, save_checkpoint_payload
from expression_deid.training.required_data import AUXILIARY_FOLDERS, load_policy, verify_prepared



def file_digest(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def setup(seed, device):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(4)
    target=torch.device('cuda' if device=='auto' and torch.cuda.is_available() else 'cpu' if device=='auto' else device)
    if target.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA is unavailable')
    return target


def guarded_output(path):
    output=Path(path).resolve()
    if output.is_relative_to((ROOT/'models').resolve()) and not output.is_relative_to((ROOT/'models/training').resolve()):
        raise ValueError('Training must not overwrite deployed weights; use models/training')
    output.parent.mkdir(parents=True,exist_ok=True)
    return output


def face_only(x):
    """Differentiable tight oval, shared by Pins training and privacy evaluation."""
    crop=x[:,:,12:100,12:88]
    y,z=torch.meshgrid(torch.arange(88,device=x.device),torch.arange(76,device=x.device),indexing='ij')
    radius=((z-37.5)/36)**2+((y-43.5)/44)**2
    mask=((1-radius)*12).clamp(0,1).to(x.dtype)[None,None]
    fill=crop[:,:,mask[0,0]>.5].median(2).values[:,:,None,None]
    return F.interpolate(crop*mask+fill*(1-mask),(100,100),mode='bilinear',align_corners=False)


def normalize(x, stats):
    return (x-float(stats['mean']))/float(stats['std'])


def initialize_transfer(model,*,smoke=False):
    if smoke: return {'source':'random initialization for smoke check only'}
    from torchvision.models import ResNet18_Weights
    weights=ResNet18_Weights.IMAGENET1K_V1
    model.backbone.initialize_imagenet(weights.get_state_dict(progress=False,check_hash=True))
    return {'source':type(weights).__name__+' ImageNet-1K V1 generic feature initialization',
            'url':weights.url,'classification_head':'new head trained on the required project dataset'}


def transfer_optimizer(model,lr,weight_decay):
    head=list(model.classifier.parameters())
    head_ids={id(p) for p in head}
    return torch.optim.AdamW([
        {'params':[p for p in model.parameters() if id(p) not in head_ids],'lr':lr},
        {'params':head,'lr':lr*5},
    ],weight_decay=weight_decay)




def motion_loss(source,anonymous):
    """Preserve eye/mouth edges and low-frequency facial motion without a raw teacher."""
    region=torch.zeros_like(source[:1])
    region[:,:,23:47,18:82]=1
    region[:,:,58:88,24:77]=1
    horizontal=(anonymous[:,:,:,1:]-anonymous[:,:,:,:-1])-(source[:,:,:,1:]-source[:,:,:,:-1])
    vertical=(anonymous[:,:,1:,:]-anonymous[:,:,:-1,:])-(source[:,:,1:,:]-source[:,:,:-1,:])
    edges=(horizontal.abs()*region[:,:,:,1:]).mean()+(vertical.abs()*region[:,:,1:,:]).mean()
    shape=F.l1_loss(F.avg_pool2d(anonymous,7,stride=1,padding=3),F.avg_pool2d(source,7,stride=1,padding=3))
    return edges+shape


def anonymous_supervision(logits,labels,*,input_source,class_weights=None):
    if input_source!='anonymized_only': raise ValueError('原图不可进入表情分类器')
    return F.cross_entropy(logits,labels,weight=class_weights,label_smoothing=.02)


class PreparedDataset(torch.utils.data.Dataset):
    def __init__(self,folder,split,*,images=None,limit_per_class=0):
        self.folder=Path(folder)
        self.images=np.load(self.folder/'images.npy',mmap_mode='r',allow_pickle=False) if images is None else images
        self.labels=np.load(self.folder/'labels.npy',allow_pickle=False)
        partitions=np.load(self.folder/'splits.npy',allow_pickle=False)
        self.indices=np.flatnonzero(partitions==split)
        if limit_per_class:
            self.indices=np.concatenate([self.indices[self.labels[self.indices]==label][:limit_per_class]
                                         for label in np.unique(self.labels[self.indices])])
        if not len(self.indices): raise ValueError(f'Empty {self.folder} {split}')
    def __len__(self): return len(self.indices)
    def __getitem__(self,index):
        i=self.indices[index]
        return torch.from_numpy(np.array(self.images[i],dtype=np.float32)[None]/255),int(self.labels[i])




def loader(dataset,batch_size,workers,shuffle=False,sampler=None):
    return torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=shuffle if sampler is None else False,
        sampler=sampler,num_workers=workers,pin_memory=torch.cuda.is_available(),persistent_workers=workers>0)


def augment(x,*,strength='standard'):
    flipped=torch.rand(len(x),device=x.device)<.5
    x[flipped]=x[flipped].flip(-1)
    if strength=='weak': gain_low,gain_high,bias=0.93,1.07,.03
    else: gain_low,gain_high,bias=.85,1.15,.06
    gain=torch.empty((len(x),1,1,1),device=x.device).uniform_(gain_low,gain_high)
    bias=torch.empty_like(gain).uniform_(-bias,bias)
    return (x*gain+bias).clamp(0,1)




def classification_metrics(model,data,device,stats,*,identity=False,temperature=1):
    model.eval(); correct=seen=0; losses=[]; predictions=[]; targets=[]; confidences=[]
    with torch.inference_mode():
        for x,y in data:
            x=x.to(device,non_blocking=True); y=y.to(device)
            if identity: x=face_only(x)
            logits=model(normalize(x,stats)).float()/temperature
            probs=logits.softmax(1); guess=probs.argmax(1)
            correct+=int((guess==y).sum()); seen+=len(y)
            losses.append(float(F.cross_entropy(logits,y,reduction='sum')))
            predictions.extend(guess.cpu().tolist()); targets.extend(y.cpu().tolist())
            confidences.extend(probs.max(1).values.cpu().tolist())
    by_class={str(label):{'samples':targets.count(label),'accuracy':float(np.mean(np.asarray(predictions)[np.asarray(targets)==label]==label))}
              for label in sorted(set(targets))}
    ece=0.; c=np.asarray(confidences); hit=np.asarray(predictions)==np.asarray(targets)
    for lo in np.arange(0,1,.1):
        selected=(c>=lo)&(c<lo+.1 if lo<.9 else c<=1)
        if selected.any(): ece+=selected.mean()*abs(hit[selected].mean()-c[selected].mean())
    return {'accuracy':correct/seen,'balanced_accuracy':float(np.mean([v['accuracy'] for v in by_class.values()])),
            'loss':sum(losses)/seen,'samples':seen,'by_class':by_class,'expected_calibration_error':float(ece)}


def train_identity(*,root=None,output='models/training/identity_pins.pt',epochs=40,batch_size=64,
                   workers=0,lr=.0003,seed=42,device='auto',limit_per_class=0,max_steps=0,initial_checkpoint=None):
    policy=load_policy(); target=setup(seed,device); output=guarded_output(output)
    if policy['identity'].get('auxiliary'): raise ValueError('No identity auxiliary is configured for this workflow')
    folder=Path(root or ROOT/'datasets/pins/prepared'); verify_prepared(folder)
    metadata=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    if metadata['dataset']!='pins': raise ValueError('Identity training requires Pins Face Recognition')
    if metadata.get('limited_preparation') and not (max_steps or limit_per_class):
        raise ValueError('Limited preparation cannot be used as a full Pins experiment')
    train=PreparedDataset(folder,'train',limit_per_class=limit_per_class)
    valid=PreparedDataset(folder,'valid',limit_per_class=limit_per_class)
    stats={'mean':.5,'std':.25,'channels':1,'space':'gray_0_1'}
    model=model_class('TransferIdentityNet')(len(metadata['classes'])).to(target)
    initial_sha=None
    initialization=None
    if initial_checkpoint:
        initial_sha=file_digest(initial_checkpoint)
        model,initial_meta=load_checkpoint(initial_checkpoint,device=target)
        if initial_meta['metrics'].get('datasets',{}).get('primary')!='pins': raise ValueError('Identity initialization must come from the same Pins workflow')
    else: initialization=initialize_transfer(model,smoke=bool(max_steps or limit_per_class))
    optimizer=transfer_optimizer(model,lr,.0001)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)
    scaler=torch.amp.GradScaler('cuda',init_scale=512,enabled=target.type=='cuda')
    training=loader(train,batch_size,workers,True); validation=loader(valid,batch_size,workers)
    best=-1.; history=[]
    for epoch in range(1,epochs+1):
        model.train(); total=correct=seen=0; started=time.perf_counter()
        for step,(x,y) in enumerate(training,1):
            x=augment(x.to(target,non_blocking=True)); y=y.to(target)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(target.type,enabled=target.type=='cuda'):
                logits=model(normalize(face_only(x),stats)); loss=F.cross_entropy(logits,y,label_smoothing=.03)
            if not torch.isfinite(loss): raise ValueError('Non-finite identity training loss')
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(),2)
            scaler.step(optimizer); scaler.update()
            total+=float(loss.detach())*len(y); correct+=int((logits.argmax(1)==y).sum()); seen+=len(y)
            if max_steps and step>=max_steps: break
        metrics=classification_metrics(model,validation,target,stats,identity=True)
        row={'epoch':epoch,'train_loss':total/seen,'train_accuracy':correct/seen,'valid':metrics,'seconds':time.perf_counter()-started,
             'datasets':{'primary':'pins','auxiliary':[]},'normalization':stats,'person_to_id':metadata['classes'],
             'identity_region':'face_only_ellipse','split_scope':metadata['split_scope'],'smoke_only':bool(max_steps or limit_per_class)}
        row.update(face_input_contract='interior_median_v1',initial_checkpoint_sha256=initial_sha,
                   initialization=initialization,
                   train_samples=len(train),valid_samples=len(valid),prepared_manifest_sha256=file_digest(folder/'manifest.jsonl'))
        if metrics['balanced_accuracy']>best:
            best=metrics['balanced_accuracy']; save_checkpoint(model,output,kind='identity',epoch=epoch,metrics=row)
        history.append(row); schedule.step(); print(json.dumps(row),flush=True)
    output.with_suffix('.history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    return {'output':str(output),'best_valid_balanced_accuracy':best,'device':str(target)}


def anonymizer_objective(model,identity,x,stats):
    anonymous=model(x)
    # The backbone can use mixed precision, while normalization, cosine and
    # classifier margins must stay FP32 to retain the privacy gradient.
    with torch.autocast(x.device.type,enabled=x.device.type=='cuda'):
        with torch.no_grad():
            original_features=identity.backbone(normalize(face_only(x.float()),stats))
        features=identity.backbone(normalize(face_only(anonymous.float()),stats))
    with torch.autocast(x.device.type,enabled=False):
        original_logits=identity.classifier(original_features.float())
        logits=identity.classifier(features.float())
        embedding=F.normalize(original_features.float(),dim=1)
        cosine=F.cosine_similarity(embedding,F.normalize(features.float(),dim=1),dim=1)
    source_label=original_logits.argmax(1)
    true_logit=logits.gather(1,source_label[:,None]).squeeze(1)
    others=logits.masked_fill(F.one_hot(source_label,logits.shape[1]).bool(),-10000).max(1).values
    margin=F.relu(true_logit-others+.5).mean()
    identity_loss=margin+F.relu(cosine-.2).mean()
    motion=motion_loss(x,anonymous); pixels=F.l1_loss(anonymous,x)
    loss=identity_loss+8*motion+.5*pixels
    return loss,{'identity_margin':float(margin.detach()),'identity_cosine':float(cosine.mean().detach()),
                 'motion_loss':float(motion.detach()),'pixel_loss':float(pixels.detach()),
                 'prediction_changed':float((logits.argmax(1)!=source_label).float().mean().detach())}


def train_anonymizer(*,identity_checkpoint,root=None,output='models/training/anonymizer_celebahq.pt',epochs=15,
                     batch_size=32,workers=0,lr=.0002,seed=42,device='auto',limit_per_class=0,max_steps=0,
                     consistency_weight=0,teacher_label_weight=0,**unknown):
    initial_checkpoint=unknown.pop('initial_checkpoint',None)
    if consistency_weight or teacher_label_weight: raise ValueError('源图表情教师已禁用')
    if unknown: raise TypeError('Unsupported old training arguments: '+','.join(unknown))
    policy=load_policy(); target=setup(seed,device); output=guarded_output(output)
    if policy['anonymization'].get('auxiliary'): raise ValueError('No anonymization auxiliary is configured for this workflow')
    folder=Path(root or ROOT/'datasets/celebahq/prepared'); verify_prepared(folder)
    metadata=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    if metadata['dataset']!='celebahq': raise ValueError('Anonymization training requires CelebA-HQ')
    if metadata.get('limited_preparation') and not (max_steps or limit_per_class):
        raise ValueError('Limited preparation cannot be used as a full CelebA-HQ experiment')
    identity,identity_meta=load_checkpoint(identity_checkpoint,device=target)
    identity_digest=file_digest(identity_checkpoint)
    if identity_meta['kind']!='identity' or identity_meta['metrics'].get('datasets',{}).get('primary')!='pins':
        raise ValueError('Use the identity model trained on Pins')
    if identity_meta['metrics'].get('smoke_only') and not max_steps:
        raise ValueError('A smoke-test identity checkpoint cannot be used for full training')
    if not max_steps and identity_meta['metrics']['valid']['accuracy'] < .6:
        raise ValueError('The Pins identity baseline is too weak for meaningful privacy training; improve it first')
    identity.eval()
    for parameter in identity.parameters(): parameter.requires_grad_(False)
    stats=identity_meta['metrics']['normalization']
    train=loader(PreparedDataset(folder,'train',limit_per_class=limit_per_class),batch_size,workers,True)
    valid=loader(PreparedDataset(folder,'valid',limit_per_class=limit_per_class),batch_size,workers)
    model=model_class('MaskedAnonymizer')().to(target)
    initial_sha=None
    if initial_checkpoint:
        initial_sha=file_digest(initial_checkpoint)
        model,initial_meta=load_checkpoint(initial_checkpoint,device=target)
        if initial_meta['metrics'].get('datasets',{}).get('primary')!='celebahq': raise ValueError('Anonymizer initialization must come from the same CelebA-HQ workflow')
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.00001)
    scaler=torch.amp.GradScaler('cuda',init_scale=512,enabled=target.type=='cuda'); best=float('inf'); history=[]
    for epoch in range(1,epochs+1):
        model.train(); total=seen=0; started=time.perf_counter()
        for step,(x,_) in enumerate(train,1):
            x=x.to(target,non_blocking=True); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(target.type,enabled=target.type=='cuda'):
                loss,components=anonymizer_objective(model,identity,x,stats)
            if not torch.isfinite(loss): raise ValueError('Non-finite anonymizer loss')
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(),2)
            scaler.step(optimizer); scaler.update(); total+=float(loss.detach())*len(x); seen+=len(x)
            if step%100==0:
                print(json.dumps({'stage':'anonymizer','epoch':epoch,'seen':seen,
                                  'total':len(train.dataset),'loss':float(loss.detach())}),flush=True)
            if max_steps and step>=max_steps: break
        model.eval(); sums={}; valid_seen=0; valid_loss=0.
        with torch.no_grad():
            for x,_ in valid:
                x=x.to(target); loss,components=anonymizer_objective(model,identity,x,stats)
                valid_loss+=float(loss)*len(x); valid_seen+=len(x)
                for key,value in components.items(): sums[key]=sums.get(key,0)+value*len(x)
        row={'epoch':epoch,'train_loss':total/seen,'valid_loss':valid_loss/valid_seen,
             **{key:value/valid_seen for key,value in sums.items()},'seconds':time.perf_counter()-started,
             'datasets':{'primary':'celebahq','auxiliary':[]},'identity_checkpoint_sha256':identity_digest,
             'expression_input_source':'anonymized_only','source_expression_teacher':False,
             'background_contract':'exactly unchanged outside the face mask','smoke_only':bool(max_steps or limit_per_class)}
        row.update(initial_checkpoint_sha256=initial_sha,face_input_contract='interior_median_v1',
                   prepared_manifest_sha256=file_digest(folder/'manifest.jsonl'))
        if row['valid_loss']<best:
            best=row['valid_loss']; save_checkpoint(model,output,kind='anonymizer',epoch=epoch,metrics=row)
        history.append(row); print(json.dumps(row),flush=True)
    output.with_suffix('.history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    return {'output':str(output),'best_valid_loss':best,'device':str(target)}


def validate_anonymous_cache(images,metadata,*,expected_producer):
    if metadata.get('expression_input_source')!='anonymized_only': raise ValueError('Expression cache contains original pixels')
    if images.dtype!=np.uint8 or images.ndim!=3 or images.shape[1:]!=(100,100): raise ValueError('Invalid anonymous grayscale cache')
    if metadata.get('anonymizer_sha256')!=expected_producer: raise ValueError('Anonymous cache producer differs from the selected model')


def anonymous_cache(folder,checkpoint,device,batch_size=128):
    folder=Path(folder); fingerprint=file_digest(checkpoint)
    cache=folder/('anonymous-'+fingerprint[:16]+'.npy'); manifest=cache.with_suffix('.json')
    images=np.load(folder/'images.npy',mmap_mode='r',allow_pickle=False)
    source_digest=file_digest(folder/'manifest.jsonl')
    source_images_digest=file_digest(folder/'images.npy')
    if cache.is_file() and manifest.is_file():
        saved=json.loads(manifest.read_text())
        if (saved.get('anonymizer_sha256')==fingerprint and saved.get('source_manifest_sha256')==source_digest
                and saved.get('source_images_sha256')==source_images_digest
                and saved.get('cache_sha256')==file_digest(cache)):
            values=np.load(cache,mmap_mode='r',allow_pickle=False)
            validate_anonymous_cache(values,saved,expected_producer=fingerprint)
            if values.shape==images.shape: return values,saved
    model,metadata=load_checkpoint(checkpoint,device=device)
    if metadata['kind']!='anonymizer' or metadata['metrics'].get('datasets',{}).get('primary')!='celebahq':
        raise ValueError('Expression inputs require the CelebA-HQ anonymizer')
    target=np.lib.format.open_memmap(cache.with_suffix('.partial.npy'),mode='w+',dtype=np.uint8,shape=images.shape)
    with torch.inference_mode():
        for start in range(0,len(images),batch_size):
            x=torch.as_tensor(np.array(images[start:start+batch_size]),device=device,dtype=torch.float32)[:,None]/255
            output=model(x).clamp(0,1)
            outside=(model.face_mask==0).expand_as(x)
            if not torch.equal(output[outside],x[outside]): raise ValueError('Anonymization changed the protected exterior')
            target[start:start+len(x)]=output[:,0].mul(255).round().to(torch.uint8).cpu().numpy()
    target.flush(); del target
    cache.with_suffix('.partial.npy').replace(cache)
    saved={'expression_input_source':'anonymized_only','anonymizer_sha256':fingerprint,
           'source_manifest_sha256':source_digest,'source_images_sha256':source_images_digest,
           'dataset':json.loads((folder/'metadata.json').read_text())['dataset'],
           'samples':len(images),'shape':list(images.shape),'cache_sha256':file_digest(cache)}
    manifest.write_text(json.dumps(saved,indent=2),encoding='utf-8')
    return np.load(cache,mmap_mode='r',allow_pickle=False),saved
