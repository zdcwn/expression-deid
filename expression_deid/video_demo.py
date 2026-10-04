"""Python-first video demonstration for the privacy-first expression pipeline.

The runner decodes a video with PyAV, converts every frame to grayscale, edits
only local eye/nose/mouth regions, and sends the anonymized aligned face to the
expression model.  It writes a side-by-side MP4 and a JSONL telemetry file so
the original frame, anonymized frame, expression and confidence can be checked
without relying on the web UI.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from expression_deid.align import landmarks_of, select_face, similarity_matrix, template_for, warp_gray
from expression_deid.config import ROOT
from expression_deid.inference import ExpressionDeidentifier


def _font(size: int = 18):
    for candidate in (
        Path(r"C:\\Windows\\Fonts\\msyh.ttc"),
        Path(r"C:\\Windows\\Fonts\\simhei.ttf"),
        Path(r"C:\\Windows\\Fonts\\arial.ttf"),
    ):
        if candidate.is_file():
            try:
                return ImageFont.truetype(str(candidate), size=size)
            except OSError:
                pass
    return ImageFont.load_default()


class FastVideoRunner:
    """Process a video while reusing face alignment between detector passes."""

    def __init__(
        self,
        pipeline: ExpressionDeidentifier,
        detect_interval: int = 5,
        *,
        display_pipeline: ExpressionDeidentifier | None = None,
        display_interval: int = 1,
    ):
        self.pipeline = pipeline
        # The optional display pipeline lets a desktop client use the same
        # human-facing GAN as the web UI without sending that branch into the
        # expression classifier.  Existing callers keep the original single
        # pipeline behavior when this is omitted.
        self.display_pipeline = display_pipeline or pipeline
        self.display_interval = max(1, int(display_interval))
        self.detect_interval = max(1, int(detect_interval))
        self.face = None
        self.landmarks = None
        self.matrix = None
        self.frames_since_detection = self.detect_interval
        self.runtime_donor = None
        self.display_aligned = None
        self.pipeline._runtime_donor = None
        self.display_pipeline._runtime_donor = None

    def reset(self):
        """Forget the previous stream's face track before a new source."""
        self.face = None
        self.landmarks = None
        self.matrix = None
        self.frames_since_detection = self.detect_interval
        self.runtime_donor = None
        self.display_aligned = None
        self.pipeline._runtime_donor = None
        self.display_pipeline._runtime_donor = None

    def _detect(self, rgb: np.ndarray):
        if self.pipeline.detector is None:
            from expression_deid.detect import YuNet
            self.pipeline.detector = YuNet()
        faces = self.pipeline.detector.detect(rgb)
        face, reason = select_face(faces, rgb.shape[:2])
        if face is None:
            self.face = self.landmarks = self.matrix = None
            return reason
        landmarks = landmarks_of(face)
        self.face = face
        self.landmarks = landmarks
        self.matrix = similarity_matrix(landmarks, template_for(256))
        return ""

    def process_frame(self, rgb: np.ndarray, frame_index: int):
        gray = self.pipeline._image_gray(rgb)
        should_detect = self.face is None or frame_index % self.detect_interval == 0
        reason = ""
        if should_detect:
            reason = self._detect(rgb)
        if self.face is None or self.landmarks is None or self.matrix is None:
            return gray, {"status": reason or "no_face", "expression": None, "confidence": None}
        aligned256, fit_error = warp_gray(gray, self.landmarks)
        if fit_error > 25.0:
            return gray, {"status": "landmarks", "expression": None, "confidence": None}
        aligned100 = self.pipeline._resize(aligned256)
        if self.pipeline.ganonymization_backend is None:
            # Appearance-based donor search is stable during a face track;
            # refresh it only when the detector refreshes the landmarks.
            if should_detect or self.runtime_donor is None:
                self.runtime_donor = self.pipeline._select_component_donor(aligned100)
            self.pipeline._runtime_donor = self.runtime_donor
        # Machine path: local component edit plus the learned residual.  It
        # retains source expression geometry and is the only frame sent to the
        # expression model.
        machine100 = self.pipeline._machine_runtime_candidate(aligned256, variant=3)
        machine_frame = self.pipeline._composite_component_edit(
            gray, machine100, self.face, self.matrix, mask_mode="features"
        )
        edited100 = self.pipeline._resize(warp_gray(machine_frame, self.landmarks)[0])
        expression = self.pipeline.predict_expression(edited100, aligned=True)
        # Human-facing path: a separate local face edit used only for display.
        # A desktop client can refresh an expensive visual GAN every few frames
        # and re-composite its aligned result onto the current background.  The
        # expression result above is still computed on every frame from the
        # machine-targeted anonymized face.
        if self.display_pipeline is self.pipeline:
            display100 = self.pipeline._human_facing_candidate(aligned256)
        elif self.display_aligned is None or frame_index % self.display_interval == 0:
            display100 = self.display_pipeline._human_facing_candidate(aligned256)
            self.display_aligned = np.asarray(display100, dtype=np.uint8)
        else:
            display100 = self.display_aligned
        display_frame = self.pipeline._composite_component_edit(
            gray, display100, self.face, self.matrix,
            mask_mode="face" if self.display_pipeline.ganonymization_backend is not None else "features",
        )
        return display_frame, {
            "status": "ok",
            "expression": expression["name"],
            "expression_zh": expression["name_zh"],
            "confidence": float(expression["confidence"]),
            "expression_input_source": "anonymized_only",
            "expression_privacy_path": "machine_targeted_local_edit",
            "display_privacy_path": "human_facing_local_face_edit",
            "fit_error": float(fit_error),
        }


def _side_by_side(original_gray: np.ndarray, anonymized_gray: np.ndarray, info: dict, frame_index: int):
    original = Image.fromarray(original_gray, mode="L").convert("RGB")
    anonymized = Image.fromarray(anonymized_gray, mode="L").convert("RGB")
    width, height = original.size
    canvas = Image.new("RGB", (width * 2, height), (16, 22, 24))
    canvas.paste(original, (0, 0))
    canvas.paste(anonymized, (width, 0))
    draw = ImageDraw.Draw(canvas)
    font = _font(max(14, min(24, width // 45)))
    small = _font(max(12, min(18, width // 60)))
    draw.rectangle((0, 0, width, 34), fill=(8, 12, 14))
    draw.rectangle((width, 0, width * 2, 34), fill=(8, 12, 14))
    draw.text((12, 8), "ORIGINAL GRAYSCALE", fill=(226, 238, 238), font=small)
    draw.text((width + 12, 8), "ANONYMIZED / EXPRESSION INPUT", fill=(91, 245, 211), font=small)
    if info.get("status") == "ok":
        label = f"{info['expression']}  {info['confidence'] * 100:.1f}%"
        draw.rectangle((width + 10, height - 42, width * 2 - 10, height - 10), fill=(8, 12, 14))
        draw.text((width + 18, height - 35), label, fill=(91, 245, 211), font=font)
    else:
        draw.text((width + 12, height - 32), info.get("status", "no_face"), fill=(255, 191, 117), font=small)
    draw.text((12, height - 24), f"frame {frame_index}", fill=(190, 205, 205), font=small)
    return np.asarray(canvas)


def run_video(
    input_path: str | Path,
    output_path: str | Path,
    *,
    telemetry_path: str | Path | None = None,
    device: str = "auto",
    detect_interval: int = 5,
    limit: int = 0,
    ganonymization_checkpoint: str | Path | None = None,
    ganonymization_repo: str | Path | None = None,
    expression_checkpoint: str | Path | None = None,
):
    input_path = str(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if ganonymization_checkpoint is None:
        candidate = ROOT / "models" / "ganonymization_50.ckpt"
        if candidate.is_file():
            ganonymization_checkpoint = candidate
    if ganonymization_repo is None:
        candidate = ROOT / "external_ganonymization"
        if candidate.is_dir():
            ganonymization_repo = candidate
    if expression_checkpoint is None or not Path(expression_checkpoint).is_file():
        for candidate in (ROOT / "models" / "expression_machine_selector_texture.pt", ROOT / "models" / "expression_gan_strength100.pt", ROOT / "models" / "expression_gan_geometry.pt", ROOT / "models" / "expression_gan_adapted.pt", ROOT / "models" / "expression_anonymized.pt", ROOT / "models" / "expression.pt"):
            if candidate.is_file():
                expression_checkpoint = candidate
                break
    pipeline = ExpressionDeidentifier(
        ROOT / "models/anonymizer_adversarial_v1.pt" if (ROOT / "models/anonymizer_adversarial_v1.pt").is_file() else ROOT / "models/anonymizer.pt",
        expression_checkpoint or (ROOT / "models/expression.pt"),
        identity_checkpoint=None,
        device=device,
        fast_mode=True,
        ganonymization_checkpoint=ganonymization_checkpoint,
        ganonymization_repo=ganonymization_repo,
    )
    warmup_started = time.perf_counter()
    # Warm CUDA kernels before the first camera frame so the displayed stream
    # does not contain a one-off multi-second expression delay.
    pipeline.predict_expression(np.zeros((100, 100), dtype=np.uint8), aligned=True)
    warmup_seconds = time.perf_counter() - warmup_started
    runner = FastVideoRunner(pipeline, detect_interval=detect_interval)
    telemetry_path = Path(telemetry_path) if telemetry_path else output_path.with_suffix(".jsonl")
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    reader = av.open(input_path)
    input_stream = reader.streams.video[0]
    fps_rate = input_stream.average_rate if input_stream.average_rate else 15
    fps = float(fps_rate)
    writer = None
    telemetry = telemetry_path.open("w", encoding="utf-8")
    started = time.perf_counter()
    processing_seconds = 0.0
    count = 0
    try:
        for frame in reader.decode(input_stream):
            if limit and count >= limit:
                break
            rgb = frame.to_rgb().to_ndarray()
            processing_started = time.perf_counter()
            anonymized, info = runner.process_frame(rgb, count)
            processing_seconds += time.perf_counter() - processing_started
            original_gray = pipeline._image_gray(rgb)
            composite = _side_by_side(original_gray, anonymized, info, count)
            if writer is None:
                writer = av.open(str(output_path), mode="w")
                stream = writer.add_stream("libx264", rate=fps_rate)
                stream.width, stream.height = composite.shape[1], composite.shape[0]
                stream.pix_fmt = "yuv420p"
                stream.options = {"preset": "ultrafast", "crf": "28"}
            for packet in writer.streams.video[0].encode(av.VideoFrame.from_ndarray(composite, format="rgb24")):
                writer.mux(packet)
            telemetry.write(json.dumps({"frame": count, **info}, ensure_ascii=False) + "\n")
            if count % 10 == 0:
                expression = info.get("expression") or info.get("status")
                confidence = info.get("confidence")
                suffix = "" if confidence is None else f" {confidence * 100:.1f}%"
                print(f"frame={count} expression={expression}{suffix}", flush=True)
            count += 1
    finally:
        if writer is not None and writer.streams.video:
            for packet in writer.streams.video[0].encode():
                writer.mux(packet)
            writer.close()
        telemetry.close()
        reader.close()
    elapsed = time.perf_counter() - started
    result = {"frames": count, "seconds": elapsed, "fps": count / elapsed if elapsed else 0.0, "analysis_seconds": processing_seconds, "analysis_fps": count / processing_seconds if processing_seconds else 0.0, "warmup_seconds": warmup_seconds, "device": device, "output": str(output_path), "telemetry": str(telemetry_path)}
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def run_camera(
    camera_index: int = 0,
    *,
    device: str = "auto",
    detect_interval: int = 5,
    expression_checkpoint: str | Path | None = None,
    ganonymization_checkpoint: str | Path | None = None,
    ganonymization_repo: str | Path | None = None,
):
    """Run the same privacy-first pipeline on a local webcam."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on optional camera package
        raise RuntimeError("摄像头演示需要 OpenCV；先安装 requirements-anonymization.txt") from exc
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if ganonymization_checkpoint is None:
        candidate = ROOT / "models" / "ganonymization_50.ckpt"
        if candidate.is_file():
            ganonymization_checkpoint = candidate
    if ganonymization_repo is None:
        candidate = ROOT / "external_ganonymization"
        if candidate.is_dir():
            ganonymization_repo = candidate
    if expression_checkpoint is None or not Path(expression_checkpoint).is_file():
        for candidate in (ROOT / "models" / "expression_machine_selector_texture.pt", ROOT / "models" / "expression_gan_strength100.pt", ROOT / "models" / "expression_gan_geometry.pt", ROOT / "models" / "expression_gan_adapted.pt", ROOT / "models" / "expression_anonymized.pt", ROOT / "models" / "expression.pt"):
            if candidate.is_file():
                expression_checkpoint = candidate
                break
    pipeline = ExpressionDeidentifier(
        ROOT / "models/anonymizer_adversarial_v1.pt" if (ROOT / "models/anonymizer_adversarial_v1.pt").is_file() else ROOT / "models/anonymizer.pt", expression_checkpoint,
        identity_checkpoint=None, device=device, fast_mode=True,
        ganonymization_checkpoint=ganonymization_checkpoint,
        ganonymization_repo=ganonymization_repo,
    )
    pipeline.predict_expression(np.zeros((100, 100), dtype=np.uint8), aligned=True)
    runner = FastVideoRunner(pipeline, detect_interval=detect_interval)
    capture = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)
    if not capture.isOpened():
        raise RuntimeError(f"打不开摄像头索引 {camera_index}")
    frame_index = 0
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            anonymized, info = runner.process_frame(rgb, frame_index)
            composite = _side_by_side(pipeline._image_gray(rgb), anonymized, info, frame_index)
            cv2.imshow("Expression De-identification", cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
            frame_index += 1
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return {"frames": frame_index, "device": device, "camera_index": int(camera_index), "expression_input_source": "anonymized_only"}


def build_parser():
    parser = argparse.ArgumentParser(description="Python 本地视频匿名化与表情演示")
    parser.add_argument("--input", required=True, help="输入视频文件")
    parser.add_argument("--output", default="outputs/video_demo.mp4", help="并排输出视频")
    parser.add_argument("--telemetry", help="逐帧 JSONL 输出路径")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--detect-interval", type=int, default=5, help="每隔多少帧重新检测一次人脸")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 帧，0 表示全部")
    parser.add_argument("--ganonymization-checkpoint", help="可选：landmark-only GANonymization 检查点；质量高但速度慢")
    parser.add_argument("--ganonymization-repo", help="GANonymization 源码目录")
    parser.add_argument("--expression-checkpoint", default=str(ROOT / "models/expression_machine_selector_texture.pt"), help="表情模型检查点；默认使用机器匿名域适配模型")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_video(args.input, args.output, telemetry_path=args.telemetry, device=args.device, detect_interval=args.detect_interval, limit=args.limit, ganonymization_checkpoint=args.ganonymization_checkpoint, ganonymization_repo=args.ganonymization_repo, expression_checkpoint=args.expression_checkpoint)


if __name__ == "__main__":
    main()
