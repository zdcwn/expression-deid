"""Command line interface for preparation, training, inference and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from expression_deid import prepare
from expression_deid import prepare_ppb
from expression_deid.config import ROOT
from expression_deid.evaluate import build_parser as build_evaluate_parser
from expression_deid.inference import ExpressionDeidentifier
from expression_deid.train import build_parser as build_train_parser
from expression_deid.video_demo import run_camera, run_video


def _run_prepare(args):
    argv = []
    if args.limit:
        argv += ["--limit", str(args.limit)]
    if args.recompute_stats:
        argv.append("--recompute-stats")
    if args.stats_only:
        argv.append("--stats-only")
    prepare.main(argv)


def _run_prepare_ppb(args):
    argv = ["--frames-per-video", str(args.frames_per_video)]
    if args.source_dir:
        argv += ["--source-dir", args.source_dir]
    if args.output_dir:
        argv += ["--output-dir", args.output_dir]
    return prepare_ppb.main(argv)


def _run_infer(args):
    pipeline = ExpressionDeidentifier(
        args.anonymizer_checkpoint, args.expression_checkpoint,
        args.identity_checkpoint, device=args.device,
        ganonymization_checkpoint=args.ganonymization_checkpoint,
        ganonymization_repo=args.ganonymization_repo,
    )
    source = Path(args.input)
    paths = [source] if source.is_file() else sorted(source.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"没有找到输入图像: {source}")
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in paths:
        result = pipeline.process(path)
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


def _run_video(args):
    expression_checkpoint = args.expression_checkpoint
    if not Path(expression_checkpoint).is_file():
        for candidate in (ROOT / "models" / "expression_machine_selector_texture.pt", ROOT / "models" / "expression_gan_strength100.pt", ROOT / "models" / "expression_gan_geometry.pt", ROOT / "models" / "expression_gan_adapted.pt", ROOT / "models" / "expression_anonymized.pt", ROOT / "models" / "expression.pt"):
            if candidate.is_file():
                expression_checkpoint = str(candidate)
                break
    gan_checkpoint = args.ganonymization_checkpoint
    gan_repo = args.ganonymization_repo
    if gan_checkpoint is None:
        candidate = ROOT / "models" / "ganonymization_50.ckpt"
        if candidate.is_file():
            gan_checkpoint = str(candidate)
    if gan_repo is None:
        candidate = ROOT / "external_ganonymization"
        if candidate.is_dir():
            gan_repo = str(candidate)
    return run_video(
        args.input, args.output, telemetry_path=args.telemetry,
        device=args.device, detect_interval=args.detect_interval, limit=args.limit,
        ganonymization_checkpoint=gan_checkpoint,
        ganonymization_repo=gan_repo,
        expression_checkpoint=expression_checkpoint,
    )


def _run_camera(args):
    return run_camera(
        args.index, device=args.device, detect_interval=args.detect_interval,
        expression_checkpoint=args.expression_checkpoint,
    )


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m expression_deid", description="座舱人脸去标识项目工具")
    subparsers = parser.add_subparsers(dest="command", required=True)
    p = subparsers.add_parser("prepare", help="计算统计量并对齐 KMU")
    p.add_argument("--limit", type=int, default=0); p.add_argument("--recompute-stats", action="store_true"); p.add_argument("--stats-only", action="store_true"); p.set_defaults(func=_run_prepare)
    p = subparsers.add_parser("prepare-ppb", help="抽取并对齐 PPB-Emo 中央红外中性帧")
    p.add_argument("--source-dir", default=None); p.add_argument("--output-dir", default=None); p.add_argument("--frames-per-video", type=int, default=5)
    p.set_defaults(func=_run_prepare_ppb)
    build_train_parser(subparsers)
    build_evaluate_parser(subparsers)
    p = subparsers.add_parser("infer", help="对单张图或目录做匿名化和表情识别")
    p.add_argument("--anonymizer-checkpoint", required=True); p.add_argument("--expression-checkpoint", required=True); p.add_argument("--identity-checkpoint"); p.add_argument("--ganonymization-checkpoint", help="可选：GANonymization 的 landmark-only 生成器检查点"); p.add_argument("--ganonymization-repo", help="GANonymization 源码目录"); p.add_argument("--input", required=True); p.add_argument("--output", default="outputs/inference"); p.add_argument("--pattern", default="*.jpg"); p.add_argument("--device", default="cpu"); p.set_defaults(func=_run_infer)
    p = subparsers.add_parser("video", help="Python 视频匿名化与匿名图表情演示")
    p.add_argument("--input", required=True); p.add_argument("--output", default="outputs/video_demo.mp4"); p.add_argument("--telemetry"); p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto"); p.add_argument("--detect-interval", type=int, default=5); p.add_argument("--limit", type=int, default=0); p.add_argument("--ganonymization-checkpoint", help="可选；模型文件存在时默认使用 models/ganonymization_50.ckpt"); p.add_argument("--ganonymization-repo", help="可选；默认使用项目内 external_ganonymization"); p.add_argument("--expression-checkpoint", default="models/expression_machine_selector_texture.pt"); p.set_defaults(func=_run_video)
    p = subparsers.add_parser("camera", help="Python 本地摄像头实时匿名化与表情演示")
    p.add_argument("--index", type=int, default=0, help="摄像头索引，默认 0")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--detect-interval", type=int, default=5)
    p.add_argument("--expression-checkpoint", default="models/expression_machine_selector_texture.pt")
    p.set_defaults(func=_run_camera)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = args.func(args)
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2))
