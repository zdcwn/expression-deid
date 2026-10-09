"""Fine tune an AffectNet backbone using ONLY anonymous grayscale cache images.

Requires requirements-quality-train.txt. Use a new output directory. Validation
selects checkpoints; the PPB test partition is never consulted in selection.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from expression_deid.expression_quality import anonymous_tensor
from expression_deid.models import load_checkpoint
from expression_deid.normalize import load_stats

UPSTREAM_COMMIT='520a051c64cd191521e5934655314e769a319684'
UPSTREAM_SHA='7863032d8c0c0bc259aa0dd7b8ebf6b607af454e7a0ea82be2b37767914ecd0f'
UPSTREAM_URL=f'https://raw.githubusercontent.com/sb-ai-lab/EmotiEffLib/{UPSTREAM_COMMIT}/models/affectnet_emotions/onnx/enet_b2_7.onnx'


def digest(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def load_backbone(onnx_path,device):
    import onnx
    from onnx2torch import convert
    from onnx2torch.node_converters.registry import add_converter
    from onnx2torch.utils.common import OperationConverterResult,onnx_mapping_from_node
    # The published graph uses opset-1 max pooling, absent from onnx2torch.
    class GlobalMaxPool(nn.Module):
        def forward(self,x):return x.flatten(2).amax(-1).reshape(x.shape[0],x.shape[1],1,1)
    from onnx2torch.node_converters.registry import get_converter
    try:get_converter('GlobalMaxPool',1)
    except NotImplementedError:
        @add_converter(operation_type='GlobalMaxPool',version=1)
        def converter(node,graph):
            return OperationConverterResult(torch_module=GlobalMaxPool(),onnx_mapping=onnx_mapping_from_node(node))
    if digest(onnx_path)!=UPSTREAM_SHA:raise ValueError('预训练 ONNX 权重的 SHA-256 不匹配')
    graph=onnx.load(str(onnx_path));node=graph.graph.node[-1]
    if node.op_type!='Gemm':raise ValueError('预训练图结构发生变化')
    feature_name=node.input[0];graph.graph.node.remove(node);graph.graph.output.remove(graph.graph.output[0])
    graph.graph.output.append(onnx.helper.make_tensor_value_info(feature_name,onnx.TensorProto.FLOAT,['batch_size',1408]))
    return convert(graph).eval().to(device)


def training_tensor(images,device,augment=False):
    x=images.to(device=device,dtype=torch.float32)/255
    if augment:
        if torch.rand(())<.5:x=x.flip(-1)
        gain=torch.empty(len(x),1,1,1,device=device).uniform_(.94,1.06)
        offset=torch.empty_like(gain).uniform_(-.03,.03)
        x=(x*gain+offset).clamp(0,1)
    x=F.interpolate(x,size=(260,260),mode='bilinear',align_corners=False)
    mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
    return (x.expand(-1,3,-1,-1)-mean)/std


def summarize(probabilities,data,split='valid'):
    result={}
    for domain in ('raf','kmu','ppb'):
        mask=(data['splits']==split)&(data['domains']==domain)
        if not mask.any():continue
        truth=data['labels'][mask];pred=probabilities[mask].argmax(1)
        cm=np.zeros((7,7),int);np.add.at(cm,(truth,pred),1)
        result[domain]=dict(samples=int(mask.sum()),accuracy=float((pred==truth).mean()),
            neutral_recall=float(cm[6,6]/cm[6].sum()) if cm[6].sum() else None,
            neutral_as_sad_angry=int(cm[6,4]+cm[6,5]),confusion=cm.tolist())
    return result


def softmax(a):
    e=np.exp(a-a.max(1,keepdims=True));return e/e.sum(1,keepdims=True)


def validate_cache(data):
    if (str(data['privacy_source'])!='anonymized_only' or data['images'].dtype!=np.uint8
            or data['images'].shape[1:]!=(100,100) or not np.isin(data['labels'],range(7)).all()):
        raise ValueError('训练只允许匿名化后的灰度缓存')
    sources={}
    for source,split in zip(data['sources'],data['splits']):sources.setdefault(str(source),set()).add(str(split))
    if any(len(s)>1 for s in sources.values()):raise ValueError('同一来源出现在多个数据划分')
    if 'ppb_people' in data:
        people=json.loads(str(data['ppb_people']));flat=[p for group in people.values() for p in group]
        if len(flat)!=len(set(flat)):raise ValueError('PPB 人员划分重复')
    for split in ('train','valid'):
        for domain in ('raf','kmu','ppb'):
            if not ((data['splits']==split)&(data['domains']==domain)).any():raise ValueError(f'{split} 缺少 {domain}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--anonymous-cache',type=Path,required=True)
    p.add_argument('--initial',type=Path,required=True,help='冻结特征头或上一阶段的 restricted PyTorch checkpoint')
    p.add_argument('--onnx',type=Path,default=ROOT/'models/enet_b2_7.onnx')
    p.add_argument('--compact',type=Path,default=ROOT/'models/expression_realtime_stable.pt')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=4);p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--seed',type=int,default=21);p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--class-balance',type=float,default=.5,help='类别采样频次的负幂，1 为完全均衡')
    p.add_argument('--backbone-lr',type=float,default=3e-5);p.add_argument('--head-lr',type=float,default=2e-4)
    args=p.parse_args()
    if args.epochs<1 or args.batch_size<1:raise ValueError('训练轮数和批大小必须为正')
    if not 0<=args.class_balance<=1 or min(args.backbone_lr,args.head_lr)<=0:raise ValueError('类别均衡和学习率参数无效')
    args.output.mkdir(parents=True,exist_ok=True)
    if args.initial.resolve()==(args.output/'best.pt').resolve():raise ValueError('保留初始权重，使用新的输出目录')
    if not args.onnx.is_file():
        args.onnx.parent.mkdir(parents=True,exist_ok=True);pending=args.onnx.with_suffix('.part')
        with urllib.request.urlopen(UPSTREAM_URL,timeout=60) as response,pending.open('wb') as handle:
            while chunk:=response.read(1024*1024):handle.write(chunk)
        if digest(pending)!=UPSTREAM_SHA:raise ValueError('预训练模型下载校验失败')
        pending.replace(args.onnx)
    torch.set_num_threads(1);torch.manual_seed(args.seed);np.random.seed(args.seed)
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    data=np.load(args.anonymous_cache);validate_cache(data)
    images=torch.tensor(data['images'].copy(),device=args.device)[:,None];labels=torch.tensor(data['labels'],device=args.device,dtype=torch.long)
    train=np.flatnonzero(data['splits']=='train');valid=np.flatnonzero(data['splits']=='valid')
    compact,metadata=load_checkpoint(args.compact,device=args.device);stats=load_stats()
    temperature=metadata['metrics']['calibration_temperature'];cp=np.zeros((len(images),7),np.float32)
    reference=np.zeros_like(cp)
    with torch.inference_mode():
        for indices in np.array_split(valid,max(1,int(np.ceil(len(valid)/64)))):
            x=images[indices].float()/255
            a=compact((x-stats.mean)/stats.std)/temperature;b=compact((x.flip(-1)-stats.mean)/stats.std)/temperature
            cp[indices]=((a.softmax(1)+b.softmax(1))/2).cpu().numpy()
            reference[indices]=a.softmax(1).cpu().numpy()
    baseline=summarize(reference,data);backbone=load_backbone(args.onnx,args.device)
    initial=torch.load(args.initial,map_location=args.device,weights_only=True)
    head=nn.Linear(1408,7,device=args.device)
    if 'backbone' in initial:backbone.load_state_dict(initial['backbone'])
    head.load_state_dict(initial.get('head',initial.get('state_dict')))
    mean=initial['mean'].to(args.device);std=initial['std'].to(args.device)
    for name,param in backbone.named_parameters():
        param.requires_grad_(name.startswith('Conv_') and int(name.split('.')[0].split('_')[1])>=470)
    weights=np.zeros(len(train),np.float32)
    for domain,fraction in (('raf',.75),('kmu',.2),('ppb',.05)):
        mask=data['domains'][train]==domain;count=np.bincount(data['labels'][train[mask]],minlength=7)
        values=count[data['labels'][train[mask]]].clip(1).astype(float)**(-args.class_balance)
        weights[mask]=fraction*values/values.sum()
    weights=torch.tensor(weights);indices=torch.tensor(train,device=args.device)
    opt=torch.optim.AdamW([{'params':[v for v in backbone.parameters() if v.requires_grad],'lr':args.backbone_lr},
                          {'params':head.parameters(),'lr':args.head_lr}],weight_decay=.001)
    scale=torch.amp.GradScaler('cuda',enabled=args.device=='cuda')
    history=[];best=-1;started=time.perf_counter()
    for epoch in range(args.epochs):
        backbone.train();head.train();loss_sum=0
        chosen=indices[torch.multinomial(weights,len(train),replacement=True).to(args.device)]
        for batch in chosen.split(args.batch_size):
            x=training_tensor(images[batch],args.device,True)
            with torch.autocast(args.device,dtype=torch.float16,enabled=args.device=='cuda'):
                logits=head((backbone(x)-mean)/std);loss=F.cross_entropy(logits,labels[batch],label_smoothing=.03)
            opt.zero_grad();scale.scale(loss).backward();scale.unscale_(opt)
            nn.utils.clip_grad_norm_(list(backbone.parameters())+list(head.parameters()),3);scale.step(opt);scale.update()
            loss_sum+=float(loss.detach())*len(batch)
        backbone.eval();head.eval();a=[];b=[]
        with torch.inference_mode():
            for batch in torch.tensor(valid,device=args.device).split(32):
                x=training_tensor(images[batch],args.device)
                a.append(head((backbone(x)-mean)/std).cpu().numpy());b.append(head((backbone(x.flip(-1))-mean)/std).cpu().numpy())
        a=np.concatenate(a);b=np.concatenate(b);rows=[]
        for qt in (1.,2.):
            for weight in (.5,.75,1.):
                probability=cp.copy();probability[valid]=(1-weight)*cp[valid]+weight*(softmax(a/qt)+softmax(b/qt))/2
                metrics=summarize(probability,data);score=.75*metrics['raf']['accuracy']+.2*metrics['kmu']['accuracy']+.05*metrics['ppb']['accuracy']
                old_cm=np.array(baseline['raf']['confusion']);new_cm=np.array(metrics['raf']['confusion'])
                recalls=new_cm.diagonal()/new_cm.sum(1).clip(1)
                old_recalls=old_cm.diagonal()/old_cm.sum(1).clip(1)
                eligible=bool(metrics['raf']['accuracy']>=baseline['raf']['accuracy'] and metrics['raf']['neutral_recall']>=.716
                              and metrics['raf']['neutral_as_sad_angry']<=64
                              and (recalls>=old_recalls-.05).all()
                              and metrics['kmu']['accuracy']>=.98 and metrics['ppb']['accuracy']>=.97)
                row=dict(epoch=epoch+1,quality_temperature=qt,quality_weight=weight,score=score,eligible=eligible,
                         loss=loss_sum/len(train),metrics=metrics,expression_input_source='anonymized_only')
                rows.append(row)
                if eligible and score>best:
                    best=score;pending=args.output/'best.pending'
                    torch.save(dict(backbone=backbone.state_dict(),head=head.state_dict(),mean=mean.cpu(),std=std.cpu(),row=row),pending)
                    pending.replace(args.output/'best.pt')
                    np.savez_compressed(args.output/'best_valid.npz',logits=a,logits_flip=b,indices=valid)
        history+=rows
        (args.output/'training.json').write_text(json.dumps(dict(baseline=baseline,history=history,cache_sha256=digest(args.anonymous_cache),
                            pretrained_sha256=UPSTREAM_SHA,seed=args.seed,class_balance=args.class_balance,
                            backbone_lr=args.backbone_lr,head_lr=args.head_lr),indent=2),encoding='utf-8')
        print(json.dumps(max(rows,key=lambda r:r['score'])),round(time.perf_counter()-started,1),flush=True)
    if best<0:raise RuntimeError('没有候选通过验证，未部署权重')

if __name__=='__main__':main()
