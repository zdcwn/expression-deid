"""Local demonstration server; no web framework or cloud service required.

Run ``python -m expression_deid.web``. Uploaded photographs remain in memory.
The static file allowlist never exposes checkpoints, datasets or project files.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from expression_deid.config import KMU_ALIGNED_DIR, ROOT
from expression_deid.labels import EXPRESSION_NAMES, EXPRESSION_ZH

WEB_DIR = ROOT / "frontend"
MAX_BODY = 15 * 1024 * 1024
MAX_IMAGE = 10 * 1024 * 1024
MAX_PIXELS = 24_000_000
ASSETS = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"),
          "/styles.css": ("styles.css", "text/css"), "/favicon.svg": ("favicon.svg", "image/svg+xml")}


def image_data_url(image) -> str:
    stream = io.BytesIO()
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def decode_image(value: str) -> Image.Image:
    if not isinstance(value, str) or not value.startswith("data:image/") or ";base64," not in value:
        raise ValueError("请选择 JPG、PNG 或 WebP 图片")
    try:
        raw = base64.b64decode(value.split(",", 1)[1], validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("图片编码无效，请重新选择") from exc
    if not raw or len(raw) > MAX_IMAGE:
        raise ValueError("图片需小于 10 MB")
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("只支持 JPG、PNG 和 WebP 图片")
            if image.width * image.height > MAX_PIXELS:
                raise ValueError("图片分辨率过大，请缩小到 2400 万像素以内")
            return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("无法读取这张图片，请检查文件是否损坏") from exc


class DemoService:
    def __init__(self, *, device="auto", evaluation=None):
        self.requested_device = device
        self.anonymizer_checkpoint = self._pick_anonymizer_checkpoint()
        self.expression_checkpoint = self._pick_expression_checkpoint()
        self.identity_checkpoint = self._pick_identity_checkpoint()
        self.gan_checkpoint = ROOT / "models/ganonymization_50.ckpt"
        self.gan_repo = ROOT / "external_ganonymization"
        self.evaluation_path = Path(evaluation or ROOT / "outputs/evaluation_machine_runtime_fast_strong_1681/evaluation.json")
        self.pipeline = None
        self.video_runner = None
        self.video_frame_index = 0
        self.lock = threading.Lock()
        self.model_error = None
        self.catalog = {}
        manifest = KMU_ALIGNED_DIR / "manifest.jsonl"
        if manifest.exists():
            for line in manifest.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                path = (KMU_ALIGNED_DIR / record.get("output", "")).resolve()
                if (record.get("status") == "ok" and path.is_file()
                        and path.is_relative_to(KMU_ALIGNED_DIR.resolve())):
                    self.catalog[path.stem] = {"path": path, "record": record}
        self._report = None
        self._report_mtime = None

    @staticmethod
    def _pick_anonymizer_checkpoint() -> Path:
        for candidate in (
            ROOT / "models/anonymizer_adversarial_v1.pt",
            ROOT / "models/anonymizer.pt",
        ):
            if candidate.is_file():
                return candidate
        return ROOT / "models/anonymizer.pt"

    @staticmethod
    def _pick_expression_checkpoint() -> Path:
        """Use the latest validated anonymized-domain model for the web demo."""
        for candidate in (
            ROOT / "models/expression_machine_selector_texture.pt",
            ROOT / "models/expression_gan_strength100.pt",
            ROOT / "models/expression_gan_innerface.pt",
            ROOT / "models/expression_gan_final.pt",
            ROOT / "models/expression_gan_runtime_ppb.pt",
            ROOT / "models/expression_anonymized.pt",
            ROOT / "models/expression.pt",
        ):
            if candidate.is_file():
                return candidate
        return ROOT / "models/expression.pt"

    @staticmethod
    def _pick_identity_checkpoint() -> Path:
        for candidate in (
            ROOT / "models/identity_faceonly.pt",
            ROOT / "models/identity.pt",
        ):
            if candidate.is_file():
                return candidate
        return ROOT / "models/identity_faceonly.pt"

    def status(self) -> dict:
        from expression_deid.models import require_torch
        models = [
            {"key": "anonymizer", "name": "人脸匿名化", "available": self.anonymizer_checkpoint.is_file(), "file": self.anonymizer_checkpoint.name},
            {"key": "expression", "name": "表情识别", "available": self.expression_checkpoint.is_file(), "file": self.expression_checkpoint.name},
            {"key": "identity", "name": "身份评估", "available": self.identity_checkpoint.is_file(), "file": self.identity_checkpoint.name},
        ]
        try:
            torch, _, _ = require_torch()
            cuda = torch.cuda.is_available()
            device = "cuda" if self.requested_device == "auto" and cuda else self.requested_device
            if device == "auto":
                device = "cpu"
            available = all(item["available"] for item in models) and (device != "cuda" or cuda)
            hardware = torch.cuda.get_device_name(0) if device == "cuda" and cuda else "CPU"
        except RuntimeError as exc:
            device, hardware, available = "cpu", "CPU", False
            self.model_error = str(exc)
        if self.pipeline:
            for model in models:
                meta = getattr(self.pipeline, model["key"] + "_meta", None)
                if model["key"] == "expression":
                    meta = self.pipeline.expression_meta
                if meta:
                    model.update(epoch=meta.get("epoch"), metrics=meta.get("metrics", {}))
        return {"ready": available and not self.model_error, "loaded": self.pipeline is not None,
                "device": device, "hardware": hardware, "models": models,
                "error": self.model_error, "local_only": True, "uploads_saved": False}

    def samples(self) -> list[dict]:
        groups = {}
        for sample_id, item in self.catalog.items():
            record = item["record"]
            groups.setdefault(int(record["label"]), []).append((sample_id, record))
        result = []
        for label in range(len(EXPRESSION_ZH)):
            values = groups.get(label, [])
            if not values:
                continue
            chosen = values[len(values) // 2]
            sample_id, record = chosen
            result.append({"id": sample_id, "source": record["source"], "person": record["person"],
                           "label": label, "expression": EXPRESSION_ZH[label],
                           "image": f"/api/sample/{sample_id}.png"})
        return result

    def report(self) -> dict:
        if not self.evaluation_path.is_file():
            return {"available": False, "summary": {}, "rows": [], "updated_at": None}
        stamp = self.evaluation_path.stat().st_mtime
        if stamp != self._report_mtime:
            self._report = json.loads(self.evaluation_path.read_text(encoding="utf-8"))
            self._report_mtime = stamp
        return {"available": True, **self._report,
                "updated_at": datetime.fromtimestamp(stamp).isoformat(timespec="seconds"),
                "scope": "当前 KMU 数据集 / 人脸局部残差匿名化验收；包含匿名化训练样本",
                "labels": list(EXPRESSION_ZH)}

    def process(self, payload: dict) -> dict:
        if not isinstance(payload, dict):
            raise ValueError("请求格式无效")
        realtime = bool(payload.get("realtime", False))
        sample_id = payload.get("sample_id")
        if sample_id:
            if sample_id not in self.catalog:
                raise ValueError("找不到这个样例，请刷新页面")
            item = self.catalog[sample_id]
            image, aligned = item["path"], True
            filename = item["record"]["source"]
            true_label = int(item["record"]["label"])
            known_person = item["record"]["person"]
        else:
            image = decode_image(payload.get("image"))
            aligned = payload.get("aligned", False)
            if not isinstance(aligned, bool):
                raise ValueError("裁切选项格式无效")
            # False means automatic mode. ``None`` lets the pipeline resize
            # already-small faces while still detecting faces in large frames.
            aligned = True if aligned else None
            filename = str(payload.get("filename", "上传图片"))[:160]
            true_label, known_person = None, None
        started = time.perf_counter()
        with self.lock:
            if self.pipeline is None:
                from expression_deid.inference import ExpressionDeidentifier
                status = self.status()
                if not status["ready"]:
                    raise RuntimeError(status["error"] or "模型文件不完整，请先完成本地训练")
                self.pipeline = ExpressionDeidentifier(
                    self.anonymizer_checkpoint, self.expression_checkpoint,
                    # Identity is a diagnostic branch for the UI only.  It
                    # compares source/anonymized identities; the expression
                    # classifier still receives only the anonymized result.
                    identity_checkpoint=(self.identity_checkpoint if self.identity_checkpoint.is_file() else None),
                    device=status["device"], fast_mode=True,
                    ganonymization_checkpoint=(self.gan_checkpoint if self.gan_checkpoint.is_file() else None),
                    ganonymization_repo=(self.gan_repo if self.gan_repo.is_dir() else None),
                    privacy_selection=False,
                )
            if realtime:
                # Reuse face landmarks across adjacent frames.  The runner
                # still performs grayscale -> local anonymization -> expression
                # inference, but avoids repeating detector work on every frame.
                from expression_deid.video_demo import FastVideoRunner
                if self.video_runner is None:
                    self.video_runner = FastVideoRunner(self.pipeline, detect_interval=5)
                if payload.get("realtime_reset"):
                    self.video_runner.reset()
                    self.video_frame_index = 0
                rgb = np.asarray(image, dtype=np.uint8)
                display_frame, telemetry = self.video_runner.process_frame(rgb, self.video_frame_index)
                self.video_frame_index += 1
                expression = {
                    "label": None,
                    "name": telemetry.get("expression", "no_face"),
                    "name_zh": telemetry.get("expression_zh", "未检测到人脸"),
                    "confidence": float(telemetry.get("confidence") or 0.0),
                    "probabilities": [],
                }
                result = {
                    "original": rgb,
                    "anonymized": display_frame,
                    "expression_original": None,
                    "expression_anonymized": expression,
                    "expression_consistent": None,
                    "expression_input_source": "anonymized_only",
                    "expression_privacy_path": telemetry.get("expression_privacy_path", "machine_targeted_local_edit"),
                    "display_privacy_path": telemetry.get("display_privacy_path", "human_facing_local_face_edit"),
                    "face_only_edit": True,
                    "component_edit": True,
                    "background_preserved": True,
                    "realtime_status": telemetry.get("status", "ok"),
                }
            else:
                # Still-image comparison keeps the full identity/SFace/gallery
                # diagnostics.  They are intentionally outside the video path.
                result = self.pipeline.process(image, aligned=aligned, diagnostics=True)
        original = result.pop("original")
        anonymized = result.pop("anonymized")
        for key in ("identity_original", "identity_anonymized"):
            if key in result:
                result[key].pop("embedding", None)
        return {**result, "filename": filename, "sample_id": sample_id,
                "original_image": image_data_url(original), "anonymized_image": image_data_url(anonymized),
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
                "device": str(self.pipeline.device), "true_label": true_label,
                "true_expression": EXPRESSION_ZH[true_label] if true_label is not None else None,
                "known_person": known_person, "labels": list(EXPRESSION_ZH),
                "expression_names": list(EXPRESSION_NAMES),
                "resolution": [int(anonymized.shape[1]), int(anonymized.shape[0])],
                "face_only_edit": bool(result.get("face_only_edit", True)),
                "component_edit": bool(result.get("component_edit", True)),
                "expression_input_source": result.get("expression_input_source", "anonymized_only"),
                "background_preserved": bool(result.get("background_preserved", False))}


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "ExpressionLab/1.0"

    @property
    def service(self):
        return self.server.service

    def _send(self, status: int, data: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type + ("; charset=utf-8" if content_type.startswith(("text/", "application/json")) else ""))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; media-src 'self' blob:; object-src 'none'; frame-ancestors 'none'")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _json(self, data, status=200):
        self._send(status, json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8"), "application/json")

    def do_GET(self):
        path = urlsplit(self.path).path
        try:
            if path in ASSETS:
                name, mime = ASSETS[path]
                return self._send(200, (WEB_DIR / name).read_bytes(), mime)
            if path == "/api/status":
                return self._json(self.service.status())
            if path == "/api/samples":
                return self._json({"samples": self.service.samples()})
            if path == "/api/evaluation":
                return self._json(self.service.report())
            if path.startswith("/api/sample/") and path.endswith(".png"):
                sample_id = path[len("/api/sample/"):-4]
                item = self.service.catalog.get(sample_id)
                if item:
                    return self._send(200, item["path"].read_bytes(), "image/png")
            self._json({"error": "找不到请求的内容"}, 404)
        except (OSError, ValueError, RuntimeError) as exc:
            self._json({"error": str(exc)}, 500)

    def do_POST(self):
        if urlsplit(self.path).path != "/api/process":
            return self._json({"error": "找不到请求的接口"}, 404)
        # Accept only requests issued by the local page, not cross-site forms.
        origin = self.headers.get("Origin")
        if origin and urlsplit(origin).netloc != self.headers.get("Host"):
            return self._json({"error": "不接受跨站请求"}, 403)
        if self.headers.get_content_type() != "application/json":
            return self._json({"error": "请求需使用 JSON 格式"}, 415)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY:
                return self._json({"error": "上传内容过大或为空，单张图片最多 10 MB"}, 413)
            payload = json.loads(self.rfile.read(length))
            self._json(self.service.process(payload))
        except (ValueError, UnicodeDecodeError) as exc:
            self._json({"error": str(exc)}, 400)
        except (RuntimeError, OSError) as exc:
            self._json({"error": str(exc)}, 422)
        except Exception:
            import traceback
            traceback.print_exc()
            self._json({"error": "模型处理失败，请查看本地服务日志后重试"}, 500)


def create_server(host="127.0.0.1", port=7860, *, device="auto", evaluation=None):
    server = ThreadingHTTPServer((host, port), DemoHandler)
    server.service = DemoService(device=device, evaluation=evaluation)
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description="启动座舱表情隐私保护演示界面")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--evaluation", help="指定已有 evaluation.json 的路径")
    args = parser.parse_args(argv)
    server = create_server(args.host, args.port, device=args.device, evaluation=args.evaluation)
    print(f"Expression Lab: http://{args.host}:{server.server_port}  /  device={args.device}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
