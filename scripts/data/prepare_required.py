"""Prepare the required datasets and the single configured auxiliary."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from expression_deid.training.required_data import prepare_required,load_policy

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['celebahq','fer2013','pins','kmu','ppb-emo','raf','affectnet','all'],default='all')
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--limit',type=int,default=0)
    args=p.parse_args(); policy=load_policy()
    names=['fer2013','pins','celebahq']+policy['expression']['auxiliary'] if args.dataset=='all' else [args.dataset]
    for name in names:
        print(json.dumps(prepare_required(name,workers=args.workers,limit=args.limit),ensure_ascii=False),flush=True)
