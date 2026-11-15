"""Training commands for the selected identity, anonymizer and expression recipes."""
from expression_deid.training.required_training import (
    train_anonymizer, train_identity,
)
from expression_deid.training.expression_current import train as train_expression, build_parser as build_expression_parser


def build_parser(subparsers):
    build_expression_parser(subparsers)
    functions={'identity':train_identity,'anonymizer':train_anonymizer}
    for module,function in functions.items():
        p=subparsers.add_parser('train-'+module)
        default={'identity':'identity_pins.pt','anonymizer':'anonymizer_celebahq.pt'}[module]
        p.add_argument('--output',default='models/training/'+default)
        p.add_argument('--root'); p.add_argument('--epochs',type=int,default=15 if module=='anonymizer' else 40)
        p.add_argument('--batch-size',type=int,default=64); p.add_argument('--workers',type=int,default=0)
        p.add_argument('--lr',type=float,default=.0002 if module=='anonymizer' else .0003)
        p.add_argument('--seed',type=int,default=42); p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
        p.add_argument('--limit-per-class',type=int,default=0); p.add_argument('--max-steps',type=int,default=0)
        p.add_argument('--initial-checkpoint')
        if module=='anonymizer': p.add_argument('--identity-checkpoint',required=True)
        p.set_defaults(func=lambda a,fn=function: fn(**{k:v for k,v in vars(a).items() if k not in {'func','command'}}))

__all__=['build_parser','train_identity','train_anonymizer','train_expression']
