"""Python desktop application for the privacy-first expression demo.

The window is intentionally local and Python-only: the left pane shows the
grayscale source frame, the right pane shows the face-only anonymized frame,
and the expression result is inferred from the anonymized face only.  Camera
and video modes reuse :class:`FastVideoRunner` so landmarks are tracked across
frames instead of running a full detector on every frame.

Examples::

    python -m expression_deid.app --mode camera --device cuda
    python -m expression_deid.app --mode video --input demo.mp4 --device cuda
    python -m expression_deid.app --mode image --input datasets/kmu-fed-aligned/01_AN_mr_001.png --device cuda
    python -m expression_deid.app --mode examples --device cuda

Press ``q`` or ``Esc`` to close.  In ``examples`` mode, keys ``1`` through
``7`` switch the displayed expression sample.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import time
from pathlib import Path

import cv2
import numpy as np

from expression_deid.config import KMU_ALIGNED_DIR, ROOT
from expression_deid.inference import ExpressionDeidentifier
from expression_deid.video_demo import FastVideoRunner


def _pick_checkpoint(candidates: tuple[Path, ...]) -> Path:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[-1]


def build_pipeline(device: str, *, use_gan: bool = True, expression_checkpoint: str | Path | None = None) -> ExpressionDeidentifier:
    """Load the validated fast models without loading the identity verifier."""
    anonymizer = _pick_checkpoint((
        ROOT / "models/anonymizer_adversarial_v1.pt",
        ROOT / "models/anonymizer.pt",
    ))
    gan_checkpoint = ROOT / "models/ganonymization_50.ckpt"
    gan_repo = ROOT / "external_ganonymization"
    use_gan = bool(use_gan and gan_checkpoint.is_file() and gan_repo.is_dir())
    # Keep fallback selection inside the published runtime set.  Historical
    # GAN-adapted checkpoints are intentionally not part of the main branch.
    candidates = (ROOT / "models/expression_machine_selector_texture.pt",)
    if not use_gan:
        candidates = (ROOT / "models/expression_realtime_stable.pt", *candidates)
    expression = Path(expression_checkpoint) if expression_checkpoint else _pick_checkpoint(candidates)
    import torch

    resolved_device = device
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    if resolved_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前 Python 环境没有可用的 CUDA。请改用 --device cpu 或检查 PyTorch/CUDA 安装。")
    pipeline = ExpressionDeidentifier(
        anonymizer,
        expression,
        identity_checkpoint=None,
        device=resolved_device,
        fast_mode=True,
        ganonymization_checkpoint=gan_checkpoint if use_gan else None,
        ganonymization_repo=gan_repo if use_gan else None,
    )
    # Warm the expression model before opening the window.
    pipeline.predict_expression(np.zeros((100, 100), dtype=np.uint8), aligned=True)
    return pipeline


def _resize_rgb(rgb: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0 or rgb.shape[1] <= max_width:
        return rgb
    scale = max_width / float(rgb.shape[1])
    return cv2.resize(rgb, (max(1, int(round(rgb.shape[1] * scale))), max(1, int(round(rgb.shape[0] * scale)))), interpolation=cv2.INTER_AREA)


def _gray_bgr(gray: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)


def _panel(original_gray: np.ndarray, anonymized_gray: np.ndarray, *, expression: str,
           confidence: float | None, fps: float, frame_index: int, device: str,
           status: str = "ok", example_name: str = "") -> np.ndarray:
    """Build a side-by-side desktop view with an English overlay for OpenCV fonts."""
    target_h = 540
    aspect = max(0.25, float(original_gray.shape[1]) / max(1, original_gray.shape[0]))
    target_w = max(240, int(round(target_h * aspect)))
    left = _gray_bgr(original_gray, (target_w, target_h))
    right = _gray_bgr(anonymized_gray, (target_w, target_h))
    canvas = np.hstack([left, right])
    bar = np.zeros((72, canvas.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, "ORIGINAL GRAYSCALE", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (220, 235, 235), 2, cv2.LINE_AA)
    cv2.putText(bar, "ANONYMIZED FACE / EXPRESSION", (target_w + 16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (100, 245, 215), 2, cv2.LINE_AA)
    cv2.putText(bar, f"{device.upper()}   {fps:5.1f} FPS   frame {frame_index}", (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 185, 185), 1, cv2.LINE_AA)
    label = expression if confidence is None else f"{expression}   confidence {confidence * 100:.1f}%"
    if status != "ok":
        label = status
    cv2.putText(bar, label, (target_w + 16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (100, 245, 215) if status == "ok" else (80, 190, 255), 2, cv2.LINE_AA)
    canvas = np.vstack([bar, canvas])
    if example_name:
        cv2.putText(canvas, f"example: {example_name}", (16, canvas.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (185, 205, 205), 1, cv2.LINE_AA)
    return canvas


def _load_examples() -> list[tuple[str, Path]]:
    manifest = KMU_ALIGNED_DIR / "manifest.jsonl"
    if not manifest.is_file():
        return []
    result: dict[int, tuple[str, Path]] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        label = int(record.get("label", -1))
        path = (KMU_ALIGNED_DIR / str(record.get("output", ""))).resolve()
        if label not in result and path.is_file() and path.is_relative_to(KMU_ALIGNED_DIR.resolve()):
            result[label] = (str(record.get("expression", label)), path)
    return [result[label] for label in sorted(result)]


def _image_frame(path: Path) -> np.ndarray:
    # ``cv2.imread`` can fail on Windows when the project path contains
    # non-ASCII characters.  Decode the bytes explicitly so the desktop
    # examples work from any user profile and checkout location.
    encoded = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
    if bgr is None:
        raise RuntimeError(f"无法读取图片：{path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def run_image(pipeline: ExpressionDeidentifier, path: Path, *, device: str, example_name: str = "") -> int:
    rgb = _image_frame(path)
    aligned = tuple(rgb.shape[:2]) in {(100, 100), (256, 256)}
    started = time.perf_counter()
    result = pipeline.process(rgb, aligned=aligned, diagnostics=False)
    elapsed = max(1e-6, time.perf_counter() - started)
    exp = result["expression_anonymized"]
    original = pipeline._image_gray(rgb)
    panel = _panel(original, result["anonymized"], expression=exp["name"], confidence=float(exp["confidence"]), fps=1.0 / elapsed, frame_index=0, device=str(pipeline.device), example_name=example_name)
    cv2.imshow("Expression De-identification · Python", cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    print(json.dumps({"mode": "image", "input": str(path), "expression": exp["name"], "confidence": float(exp["confidence"]), "elapsed_ms": round(elapsed * 1000, 1), "device": str(pipeline.device), "expression_input_source": result["expression_input_source"]}, ensure_ascii=False, indent=2))
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (27, ord("q")):
            break
    return 0


def run_stream(pipeline: ExpressionDeidentifier, capture: cv2.VideoCapture, *, device: str, detect_interval: int, max_width: int, source_fps: float | None = None) -> int:
    display = build_pipeline(device, use_gan=True) if pipeline.ganonymization_backend is None else pipeline
    runner = FastVideoRunner(pipeline, detect_interval=detect_interval, display_pipeline=display, display_interval=15)
    frame_index = 0
    started = time.perf_counter()
    analysis_times = deque(maxlen=24)
    analysis_seconds = 0.0
    successful = 0
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            rgb = _resize_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), max_width)
            timestamp = frame_index / source_fps if source_fps else time.monotonic()
            analysis_started = time.perf_counter()
            anonymized, info = runner.process_frame(rgb, frame_index, timestamp=timestamp)
            duration = time.perf_counter() - analysis_started
            analysis_times.append((duration, info.get('status') == 'ok'))
            analysis_seconds += duration
            successful += info.get('status') == 'ok'
            fps = sum(done for _, done in analysis_times) / max(1e-6, sum(duration for duration, _ in analysis_times))
            panel = _panel(pipeline._image_gray(rgb), anonymized, expression=info.get("expression") or "NO FACE", confidence=info.get("confidence"), fps=fps, frame_index=frame_index, device=str(pipeline.device), status=info.get("status", "ok"))
            cv2.imshow("Expression De-identification · Python", cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
            frame_index += 1
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        capture.release()
        if display is not pipeline: display.close()
        cv2.destroyAllWindows()
    print(json.dumps({"mode": "stream", "frames": frame_index, "fps": frame_index / max(1e-6, time.perf_counter() - started), "successful_analysis_frames": successful, "analysis_fps": successful / max(1e-6, analysis_seconds), "device": str(pipeline.device), "expression_input_source": "anonymized_only"}, ensure_ascii=False, indent=2))
    return 0


def run_examples(pipeline: ExpressionDeidentifier, *, device: str) -> int:
    examples = _load_examples()
    if not examples:
        raise RuntimeError("没有找到 datasets/kmu-fed-aligned/manifest.jsonl 示例图片")
    index = 0
    while True:
        label, path = examples[index]
        rgb = _image_frame(path)
        started = time.perf_counter()
        result = pipeline.process(rgb, aligned=True, diagnostics=False)
        elapsed = max(1e-6, time.perf_counter() - started)
        exp = result["expression_anonymized"]
        panel = _panel(pipeline._image_gray(rgb), result["anonymized"], expression=exp["name"], confidence=float(exp["confidence"]), fps=1.0 / elapsed, frame_index=index, device=str(pipeline.device), example_name=f"{index + 1}: {label} / {path.name}")
        cv2.imshow("Expression De-identification · Python", cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        print(f"example {index + 1}/{len(examples)}  {label}  {exp['name']}  {float(exp['confidence']) * 100:.1f}%")
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord("q")):
            break
        if ord("1") <= key <= ord("7"):
            candidate = key - ord("1")
            if candidate < len(examples):
                index = candidate
        elif key in (ord("n"), 32, 13):
            index = (index + 1) % len(examples)
    cv2.destroyAllWindows()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Python 本地人脸匿名化与表情实时应用")
    parser.add_argument("--mode", choices=("camera", "video", "image", "examples"), default="camera")
    parser.add_argument("--input", help="video/image 模式的输入文件")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--detect-interval", type=int, default=5, help="视频模式每隔多少帧重新检测人脸")
    parser.add_argument("--max-width", type=int, default=960, help="处理前把视频帧缩到的最大宽度，降低延迟")
    parser.add_argument("--stream-quality", choices=("fast", "quality"), default="fast", help="视频/摄像头：fast 使用轻量局部编辑，quality 启用 landmark-only 生成器")
    parser.add_argument("--expression-checkpoint", help="显式指定表情权重，用于复测或回退")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode in {"video", "image"} and not args.input:
        raise SystemExit(f"--mode {args.mode} 需要同时提供 --input")
    use_gan = args.mode in {"image", "examples"} or args.stream_quality == "quality"
    pipeline = build_pipeline(args.device, use_gan=use_gan, expression_checkpoint=args.expression_checkpoint)
    try:
        if args.mode == "image":
            return run_image(pipeline, Path(args.input), device=args.device)
        if args.mode == "examples":
            return run_examples(pipeline, device=args.device)
        if args.mode == "video":
            capture = cv2.VideoCapture(str(Path(args.input)))
        else:
            capture = cv2.VideoCapture(int(args.camera_index), cv2.CAP_DSHOW)
        if not capture.isOpened():
            raise SystemExit("无法打开视频或摄像头，请检查输入路径、摄像头权限和索引。")
        source_fps = None
        if args.mode == "video":
            rate = capture.get(cv2.CAP_PROP_FPS)
            source_fps = rate if np.isfinite(rate) and rate > 0 else 30.0
        return run_stream(pipeline, capture, device=args.device, detect_interval=args.detect_interval, max_width=args.max_width, source_fps=source_fps)
    finally:
        pipeline.close()


if __name__ == "__main__":
    raise SystemExit(main())
