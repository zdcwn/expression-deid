"""Python-first video demonstration for the privacy-first expression pipeline.

The runner decodes a video with PyAV, converts every frame to grayscale, edits
only local eye/nose/mouth regions, and sends the anonymized aligned face to the
expression model.  It writes a side-by-side MP4 and a JSONL telemetry file so
the original frame, anonymized frame, expression and confidence can be checked
without relying on a browser UI.
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
from expression_deid.inference import ExpressionDeidentifier, AnonymizationUnavailable
from expression_deid.ganonymization_backend import FaceLandmarksUnavailable
from expression_deid.labels import EXPRESSION_NAMES, EXPRESSION_ZH
from expression_deid.temporal import TemporalExpressionFilter


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
        expression_predictor=None,
        track_reset_gap: float = .75,
    ):
        self.pipeline = pipeline
        # The optional display pipeline lets a desktop client use the same
        # human-facing GAN without sending that branch into the
        # expression classifier.  Existing callers keep the original single
        # pipeline behavior when this is omitted.
        self.display_pipeline = display_pipeline or pipeline
        self.display_interval = max(1, int(display_interval))
        self.detect_interval = max(1, int(detect_interval))
        self.expression_predictor = expression_predictor
        if expression_predictor is not None and hasattr(expression_predictor, 'validate_pipeline'):
            expression_predictor.validate_pipeline(pipeline)
        self.expression_filter = TemporalExpressionFilter(reset_gap=track_reset_gap)
        self.last_frame_index = None
        self.last_timestamp = None
        self.track_id = 0
        self.frame_shape = None
        self.face = None
        self.landmarks = None
        self.matrix = None
        self.frames_since_detection = self.detect_interval
        self.runtime_donor = None
        self.display_aligned = None
        self.display_timestamp = None
        self.display_frame_index = None
        self.pipeline._runtime_donor = None
        self.display_pipeline._runtime_donor = None

    def reset(self):
        """Forget the previous stream's face track before a new source."""
        self.expression_filter.reset()
        self.last_frame_index = None
        self.last_timestamp = None
        self.frame_shape = None
        self.face = None
        self.landmarks = None
        self.matrix = None
        self.frames_since_detection = self.detect_interval
        self.runtime_donor = None
        self.display_aligned = None
        self.display_timestamp = None
        self.display_frame_index = None
        self.pipeline._runtime_donor = None
        self.display_pipeline._runtime_donor = None

    def _detect(self, rgb: np.ndarray):
        if self.pipeline.detector is None:
            from expression_deid.detect import YuNet
            self.pipeline.detector = YuNet()
        faces = self.pipeline.detector.detect(rgb)
        face, reason = select_face(faces, rgb.shape[:2])
        if face is None:
            self.reset()
            return reason
        if self.face is not None and not self._same_track(self.face, face):
            self.reset()
        landmarks = landmarks_of(face)
        if self.face is None: self.track_id += 1
        self.face = face
        self.landmarks = landmarks
        self.matrix = similarity_matrix(landmarks, template_for(256))
        return ""

    @staticmethod
    def _same_track(previous, current):
        """Reset state after a large box jump or scale change, without an ID model."""
        ax, ay, aw, ah = previous[:4]
        bx, by, bw, bh = current[:4]
        overlap = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
        union = aw * ah + bw * bh - overlap
        ratio = (bw * bh) / max(1.0, aw * ah)
        return union > 0 and overlap / union >= 0.25 and 0.35 <= ratio <= 2.8

    def process_frame(self, rgb: np.ndarray, frame_index: int, *, timestamp: float | None = None):
        timestamp = time.monotonic() if timestamp is None else float(timestamp)
        if not np.isfinite(timestamp): raise ValueError('视频时间戳必须为有限数值')
        if (self.last_frame_index is not None and frame_index <= self.last_frame_index
                or self.frame_shape is not None and self.frame_shape != rgb.shape[:2]
                or self.last_timestamp is not None and (timestamp <= self.last_timestamp or timestamp - self.last_timestamp > self.expression_filter.reset_gap)):
            self.reset()
        self.last_frame_index, self.frame_shape = frame_index, rgb.shape[:2]
        self.last_timestamp = timestamp
        gray = self.pipeline._image_gray(rgb)
        should_detect = self.face is None or frame_index % self.detect_interval == 0
        reason = ""
        if should_detect:
            reason = self._detect(rgb)
        if self.face is None or self.landmarks is None or self.matrix is None:
            # An absent detection is not evidence that the frame contains no
            # face. Suppress the output while keeping the source panel usable.
            return np.zeros_like(gray), {"status": reason or "no_face", "expression": None, "confidence": None, 'display_status': 'concealed_no_reliable_face'}
        aligned256, fit_error = warp_gray(gray, self.landmarks)
        if fit_error > 25.0:
            self.reset()
            return np.zeros_like(gray), {"status": "landmarks", "expression": None, "confidence": None, 'display_status': 'concealed_no_reliable_face'}
        aligned100 = self.pipeline._resize(aligned256)
        if self.pipeline.ganonymization_backend is None:
            # Keep the donor fixed within this track. Reselecting on every
            # detector pass makes tiny exposure changes alter anonymous texture
            # abruptly, even when the driver's expression has not changed.
            if self.runtime_donor is None:
                self.runtime_donor = self.pipeline._select_component_donor(aligned100)
            self.pipeline._runtime_donor = self.runtime_donor
        # Machine path: local component edit plus the learned residual.  It
        # retains source expression geometry and is the only frame sent to the
        # expression model.
        try:
            machine100 = self.pipeline._machine_runtime_candidate(aligned256, variant=3)
        except (FaceLandmarksUnavailable, AnonymizationUnavailable) as exc:
            self.expression_filter.reset()
            concealed = self.pipeline._composite_component_edit(gray, np.zeros((100, 100), dtype=np.uint8), self.face, self.matrix, mask_mode='conceal')
            return concealed, {'status': 'anonymization_unavailable', 'expression': None, 'confidence': None,
                               'display_status': 'concealed_after_anonymization_failure', 'error': str(exc), 'track_id': self.track_id}
        machine_frame = self.pipeline._composite_component_edit(
            gray, machine100, self.face, self.matrix, mask_mode=getattr(self.pipeline, 'machine_privacy', {}).get('mask_mode', 'features')
        )
        edited100 = self.pipeline._resize(warp_gray(machine_frame, self.landmarks)[0])
        if self.expression_predictor is None:
            expression = self.pipeline.predict_expression(edited100, aligned=True)
        else:
            expression = self.expression_predictor.predict_anonymous(
                edited100, input_source='anonymized_only'
            )
        decision = self.expression_filter.update(expression["probabilities"], timestamp)
        label = decision["label"]
        # Human-facing path: a separate local face edit used only for display.
        # A desktop client can refresh an expensive visual GAN every few frames
        # and re-composite its aligned result onto the current background.  The
        # expression result above is still computed on every frame from the
        # machine-targeted anonymized face.
        display_status = "ok"
        conceal = False
        try:
            if self.display_pipeline is self.pipeline:
                display100 = self.pipeline._human_facing_candidate(aligned256)
            elif (self.display_aligned is None or frame_index - self.display_frame_index >= self.display_interval
                  or timestamp - self.display_timestamp > .5):
                display100 = self.display_pipeline._human_facing_candidate(aligned256)
                self.display_aligned = np.asarray(display100, dtype=np.uint8)
                self.display_timestamp = timestamp
                self.display_frame_index = frame_index
            else:
                display100 = self.display_aligned
        except (FaceLandmarksUnavailable, AnonymizationUnavailable):
            # A temporary visual failure must not stop anonymous-frame
            # classification or expose the original face in the output panel.
            if self.display_aligned is not None and timestamp - self.display_timestamp <= .5:
                display100 = self.display_aligned
                display_status = "cached_after_landmark_loss"
            else:
                self.display_aligned = self.display_timestamp = None
                display100 = np.zeros((100, 100), dtype=np.uint8)
                display_status = "concealed_after_landmark_loss"
                conceal = True
        display_frame = self.pipeline._composite_component_edit(
            gray, display100, self.face, self.matrix,
            mask_mode="conceal" if conceal else "face" if self.display_pipeline.ganonymization_backend is not None else "features",
        )
        return display_frame, {
            "status": "ok",
            "track_id": self.track_id,
            "expression": "uncertain" if label is None else EXPRESSION_NAMES[label],
            "expression_zh": "暂不确定" if label is None else EXPRESSION_ZH[label],
            "confidence": decision["confidence"],
            "probabilities": decision["probabilities"],
            "temporal_state": decision["state"],
            "confidence_source": "temporal_anonymized_probabilities",
            "display_status": display_status,
            "raw_expression": expression["name"],
            "raw_confidence": float(expression["confidence"]),
            "raw_probabilities": expression["probabilities"],
            "expression_input_source": "anonymized_only",
            "expression_backend": expression.get('expression_backend', 'compact_single_frame'),
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
    draw.text((width + 12, 8), "ANONYMIZED VIDEO", fill=(91, 245, 211), font=small)
    if info.get("status") == "ok":
        confidence = info.get("confidence")
        label = info['expression'] if confidence is None else f"{info['expression']}  {confidence * 100:.1f}%"
        draw.rectangle((width + 10, height - 42, width * 2 - 10, height - 10), fill=(8, 12, 14))
        draw.text((width + 18, height - 35), label, fill=(91, 245, 211), font=font)
    else:
        draw.text((width + 12, height - 32), info.get("status", "no_face"), fill=(255, 191, 117), font=small)
    draw.text((12, height - 24), f"frame {frame_index}", fill=(190, 205, 205), font=small)
    return np.asarray(canvas)


def _stream_pipelines(device, expression_checkpoint=None, gan_checkpoint=None, gan_repo=None):
    """One default mapping for file/camera and the native realtime page."""
    from expression_deid.app import build_pipeline
    if expression_checkpoint and not Path(expression_checkpoint).is_file():
        raise FileNotFoundError(f"指定的表情检查点不存在: {expression_checkpoint}")
    if gan_checkpoint and not Path(gan_checkpoint).is_file():
        raise FileNotFoundError(f"指定的匿名化检查点不存在: {gan_checkpoint}")
    if gan_repo and not Path(gan_repo).is_dir():
        raise FileNotFoundError(f"指定的匿名化源码不存在: {gan_repo}")
    pipeline = build_pipeline(device, use_gan=False, expression_checkpoint=expression_checkpoint)
    try:
        if gan_checkpoint:
            display = ExpressionDeidentifier(
                ROOT/'models/anonymizer_adversarial_v1.pt',
                ROOT/'models/expression_machine_selector_texture.pt', device=str(pipeline.device),
                fast_mode=True, privacy_selection=False, ganonymization_checkpoint=gan_checkpoint,
                ganonymization_repo=gan_repo or ROOT/'external_ganonymization')
        else:
            display = build_pipeline(device, use_gan=True)
    except Exception:
        pipeline.close()
        raise
    return pipeline, display


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
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"视频不存在: {input_path}")
    output_path = Path(output_path)
    telemetry_path = Path(telemetry_path) if telemetry_path else output_path.with_suffix(".jsonl")
    if len({input_path.resolve(), output_path.resolve(), telemetry_path.resolve()}) != 3:
        raise ValueError("输入视频、输出视频和遥测文件必须使用不同路径")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline, display = _stream_pipelines(device, expression_checkpoint, ganonymization_checkpoint, ganonymization_repo)
    device = str(pipeline.device)
    warmup_started = time.perf_counter()
    # Warm CUDA kernels before the first camera frame so the displayed stream
    # does not contain a one-off multi-second expression delay.
    try:
        pipeline.predict_expression(np.zeros((100, 100), dtype=np.uint8), aligned=True)
        warmup_seconds = time.perf_counter() - warmup_started
        runner = FastVideoRunner(pipeline, detect_interval=detect_interval, display_pipeline=display, display_interval=15)
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pipeline.close()
        if display is not pipeline: display.close()
        raise
    try:
        reader = av.open(str(input_path))
    except Exception:
        pipeline.close()
        if display is not pipeline: display.close()
        raise
    if not reader.streams.video:
        reader.close(); pipeline.close()
        if display is not pipeline: display.close()
        raise ValueError("输入文件没有视频流")
    input_stream = reader.streams.video[0]
    fps_rate = input_stream.average_rate if input_stream.average_rate else 15
    fps = float(fps_rate)
    writer = None
    try:
        telemetry = telemetry_path.open("w", encoding="utf-8")
    except Exception:
        reader.close(); pipeline.close()
        if display is not pipeline: display.close()
        raise
    started = time.perf_counter()
    processing_seconds = 0.0
    count = successful = 0
    try:
        for frame in reader.decode(input_stream):
            if limit and count >= limit:
                break
            rgb = frame.to_rgb().to_ndarray()
            processing_started = time.perf_counter()
            timestamp = float(frame.time) if frame.time is not None else count / fps
            anonymized, info = runner.process_frame(rgb, count, timestamp=timestamp)
            processing_seconds += time.perf_counter() - processing_started
            successful += info.get("status") == "ok"
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
        try:
            if writer is not None:
                try:
                    if writer.streams.video:
                        for packet in writer.streams.video[0].encode(): writer.mux(packet)
                finally:
                    writer.close()
        finally:
            telemetry.close(); reader.close(); pipeline.close()
            if display is not pipeline: display.close()
    elapsed = time.perf_counter() - started
    result = {"frames": count, "seconds": elapsed, "fps": count / elapsed if elapsed else 0.0, "analysis_seconds": processing_seconds, "successful_analysis_frames": successful, "analysis_fps": successful / processing_seconds if processing_seconds else 0.0, "attempted_fps": count / processing_seconds if processing_seconds else 0.0, "warmup_seconds": warmup_seconds, "device": device, "output": str(output_path), "telemetry": str(telemetry_path)}
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
    pipeline, display = _stream_pipelines(device, expression_checkpoint, ganonymization_checkpoint, ganonymization_repo)
    device = str(pipeline.device)
    try:
        pipeline.predict_expression(np.zeros((100, 100), dtype=np.uint8), aligned=True)
        runner = FastVideoRunner(pipeline, detect_interval=detect_interval, display_pipeline=display, display_interval=15)
    except Exception:
        pipeline.close()
        if display is not pipeline: display.close()
        raise
    capture = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture.release(); pipeline.close()
        if display is not pipeline: display.close()
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
        capture.release(); pipeline.close()
        if display is not pipeline: display.close()
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
    parser.add_argument("--expression-checkpoint", default=None, help="表情模型检查点；默认与桌面实时页一致")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_video(args.input, args.output, telemetry_path=args.telemetry, device=args.device, detect_interval=args.detect_interval, limit=args.limit, ganonymization_checkpoint=args.ganonymization_checkpoint, ganonymization_repo=args.ganonymization_repo, expression_checkpoint=args.expression_checkpoint)


if __name__ == "__main__":
    main()
