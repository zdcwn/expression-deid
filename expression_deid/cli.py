"""Command line interface for preparation, training, inference and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from expression_deid.config import ROOT
from expression_deid.training.required_evaluate import build_parser as build_evaluate_parser
from expression_deid.runtime.pipeline import ExpressionDeidentifier
from expression_deid.training.train import build_parser as build_train_parser
from expression_deid.runtime.video import run_camera, run_video


def _run_prepare(args):
    from expression_deid.training.required_data import prepare_required,load_policy
    policy=load_policy()
    names=['pins','celebahq','fer2013']+policy['expression']['auxiliary'] if args.dataset=='all' else [args.dataset]
    return {name:prepare_required(name,workers=args.workers,limit=args.limit) for name in names}


def _run_infer(args):
    pipeline = ExpressionDeidentifier(
        args.anonymizer_checkpoint, args.expression_checkpoint,
        args.identity_checkpoint, device=args.device, fast_mode=False,
    )
    try:
        source = Path(args.input)
        paths = [source] if source.is_file() else sorted(source.glob(args.pattern))
        if not paths:
            raise FileNotFoundError(f"没有找到输入图像: {source}")
        target = Path(args.output)
        target.mkdir(parents=True, exist_ok=True)
        rows = []
        for path in paths:
            result = pipeline.process(path,aligned=args.aligned,diagnostics=args.identity_checkpoint is not None)
            anonymized = result.pop("anonymized")
            result.pop("original", None)
            out_image = target / f"{path.stem}.png"
            Image.fromarray(anonymized, mode="L").save(out_image)
            result["source"] = str(path)
            result["output"] = str(out_image)
            rows.append(result)
            print(json.dumps({k: v for k, v in result.items() if k != "identity_original" and k != "identity_anonymized"}, ensure_ascii=False))
        (target / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"samples": len(rows), "output": str(target)}
    finally:
        pipeline.close()


def _run_video(args):
    return run_video(
        args.input, args.output, telemetry_path=args.telemetry,
        device=args.device, detect_interval=args.detect_interval, limit=args.limit,
        ganonymization_checkpoint=args.ganonymization_checkpoint,
        ganonymization_repo=args.ganonymization_repo,
        expression_checkpoint=args.expression_checkpoint,
    )


def _run_camera(args):
    return run_camera(
        args.index, device=args.device, detect_interval=args.detect_interval,
        expression_checkpoint=args.expression_checkpoint,
    )


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m expression_deid", description="座舱人脸去标识项目工具")
    subparsers = parser.add_subparsers(dest="command", required=True)
    p = subparsers.add_parser("prepare", help="准备指定的 CelebA-HQ、FER-2013、Pins 和已选辅助集")
    p.add_argument('--dataset',choices=['pins','celebahq','fer2013','kmu','ppb-emo','raf','affectnet','all'],default='all')
    p.add_argument('--workers',type=int,default=6); p.add_argument('--limit',type=int,default=0); p.set_defaults(func=_run_prepare)
    build_train_parser(subparsers)
    build_evaluate_parser(subparsers)
    p = subparsers.add_parser("infer", help="对单张图或目录做匿名化和表情识别")
    p.add_argument('--anonymizer-checkpoint',default=str(ROOT/'models/anonymization/anonymizer_celebahq.pt'))
    p.add_argument('--expression-checkpoint',default=str(ROOT/'models/expression/expression_current.pt'))
    p.add_argument('--identity-checkpoint'); p.add_argument('--input',required=True)
    p.add_argument('--output',default='outputs/inference'); p.add_argument('--pattern',default='*.jpg')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto'); p.add_argument('--aligned',action='store_true')
    p.set_defaults(func=_run_infer)
    p = subparsers.add_parser("video", help="Python 视频匿名化与匿名图表情演示")
    p.add_argument("--input", required=True); p.add_argument("--output", default="outputs/video_demo.mp4"); p.add_argument("--telemetry"); p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto"); p.add_argument("--detect-interval", type=int, default=5); p.add_argument("--limit", type=int, default=0); p.add_argument("--ganonymization-checkpoint", help="可选；模型文件存在时默认使用 models/anonymization/ganonymization_50.ckpt"); p.add_argument("--ganonymization-repo", help="可选；默认使用项目内 third_party/ganonymization"); p.add_argument("--expression-checkpoint", default=None); p.set_defaults(func=_run_video)
    p = subparsers.add_parser("camera", help="Python 本地摄像头实时匿名化与表情演示")
    p.add_argument("--index", type=int, default=0, help="摄像头索引，默认 0")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--detect-interval", type=int, default=5)
    p.add_argument("--expression-checkpoint", default=None)
    p.set_defaults(func=_run_camera)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = args.func(args)
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2))
