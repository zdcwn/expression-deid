"""Optional landmark-only GANonymization backend.

This adapter is deliberately lazy: the realtime component path remains usable
without MediaPipe, OpenCV or the external GAN checkpoint.  When a trained
GANonymization checkpoint is supplied, the generator receives only a rendered
478-point landmark image, so source RGB/gray pixels do not enter that generator.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


class FaceLandmarksUnavailable(RuntimeError):
    """A detected face lacks sufficiently reliable dense geometry."""


class GANonymizationBackend:
    def __init__(self, checkpoint: str | Path, *, device: str = "cuda", repo_root: str | Path | None = None, image_size: int = 512, landmark_model: str | Path | None = None, geometry_strength: float = 0.0, geometry_profile: int = 0):
        try:
            import torch
            import mediapipe as mp
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "GANonymization 后端需要 mediapipe 和 PyTorch；实时局部编辑路径不依赖它们。"
            ) from exc
        self.torch = torch
        self.image_size = int(image_size)
        root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2] / "third_party/ganonymization"
        if not (root / "lib").is_dir():
            raise FileNotFoundError(
                f"找不到 GANonymization 源码目录: {root}; 请先克隆 hcmlab/GANonymization"
            )
        # Reuse only the audited generator architecture from the MIT-licensed
        # official repository.  Importing its training module would also load
        # obsolete Lightning/torchvision dependencies; none are needed for
        # inference.  No import or other top-level statement is executed here.
        tree = ast.parse((root / "lib/models/pix2pix.py").read_text(encoding="utf-8"))
        names = {"UNetDown", "UNetUp", "GeneratorUNet"}
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
        if {node.name for node in classes} != names:
            raise ValueError("GANonymization generator architecture is incomplete")
        namespace = {"nn": torch.nn, "torch": torch, "__name__": __name__}
        module = ast.fix_missing_locations(ast.Module(body=classes, type_ignores=[]))
        exec(compile(module, str(root / "lib/models/pix2pix.py"), "exec"), namespace)
        payload = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        state = payload.get("state_dict", payload)
        generator_state = {key.removeprefix("generator."): value for key, value in state.items() if key.startswith("generator.")}
        if not generator_state:
            raise ValueError("Checkpoint has no generator weights")
        if any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all().item()
               for value in generator_state.values()):
            raise ValueError('GAN 检查点包含无效或非有限的生成器权重')
        self.model = namespace["GeneratorUNet"]()
        self.model.load_state_dict(generator_state, strict=True)
        self.model.eval().to(device)
        self.device = torch.device(device)
        if self.device.type == "cuda":
            self.model.half()
        self.mp = mp
        self.geometry_strength = float(geometry_strength)
        self.geometry_profile = int(geometry_profile)
        model_path = Path(landmark_model) if landmark_model else Path(__file__).resolve().parents[2] / "models/detection/face_landmarker.task"
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing landmark asset: {model_path}; run scripts/assets/download_anonymization_assets.py")
        self.face_mesh = mp.tasks.vision.FaceLandmarker.create_from_options(
            mp.tasks.vision.FaceLandmarkerOptions(
                # Bytes avoid native Windows path encoding and shared C:/Temp
                # permissions. These are the exact same published model bytes.
                base_options=mp.tasks.BaseOptions(model_asset_buffer=model_path.read_bytes()),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=1,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                min_face_detection_confidence=0.5,
            )
        )
        self.last_landmarks = None
        self.last_anonymized_landmarks = None
        self.last_preprocessing = None

    def _landmark_image(self, gray: np.ndarray) -> np.ndarray:
        source = np.asarray(Image.fromarray(gray, mode="L").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR))
        rgb = np.repeat(source[:, :, None], 3, axis=2)
        result = self.face_mesh.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        self.last_preprocessing = 'raw'
        if not result.face_landmarks:
            # IR cabins can have a narrow luminance range. Contrast is changed
            # only for geometry detection; the generator still sees points.
            contrasted = np.asarray(ImageOps.autocontrast(Image.fromarray(source), cutoff=2))
            rgb = np.repeat(contrasted[:, :, None], 3, axis=2)
            result = self.face_mesh.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
            self.last_preprocessing = 'autocontrast'
        if not result.face_landmarks:
            # A blank condition can produce a plausible but unrelated face.
            # Never silently claim that such an image preserves expression.
            self.last_landmarks = self.last_anonymized_landmarks = None
            raise FaceLandmarksUnavailable("GANonymization 没有检测到可信的密集人脸关键点")
        points = result.face_landmarks[0]
        self.last_landmarks = np.asarray([(p.x, p.y) for p in points], dtype=np.float32)
        from expression_deid.preprocessing.align import template_for
        p = self.last_landmarks
        anchors = np.stack(((p[33]+p[133])/2,(p[263]+p[362])/2,p[1],p[61],p[291]))
        if not np.isfinite(p).all() or np.linalg.norm(anchors-template_for(100)/100,axis=1).mean() > .18:
            self.last_landmarks = self.last_anonymized_landmarks = None
            raise FaceLandmarksUnavailable('密集关键点偏离已对齐的人脸位置')
        from expression_deid.preprocessing.feature_geometry import identity_geometry
        try:
            self.last_anonymized_landmarks = identity_geometry(
                self.last_landmarks,
                self.mp.tasks.vision.FaceLandmarksConnections,
                self.geometry_strength,
                self.geometry_profile,
            )
        except (ValueError, RuntimeError) as exc:
            self.last_landmarks = self.last_anonymized_landmarks = None
            raise FaceLandmarksUnavailable(f'密集人脸几何不可用: {exc}') from exc
        canvas = Image.new("L", (self.image_size, self.image_size), 0)
        if points:
            draw = ImageDraw.Draw(canvas)
            # Official MediaPipe drawing_utils draws a radius-one white border
            # around circle_radius=0. Render that same five-pixel cross.
            xy = [(min(int(px * self.image_size), self.image_size - 1), min(int(py * self.image_size), self.image_size - 1)) for px, py in self.last_anonymized_landmarks if 0 <= px <= 1 and 0 <= py <= 1]
            for point in xy:
                draw.ellipse((point[0] - 1, point[1] - 1, point[0] + 1, point[1] + 1), fill=255)
        return np.repeat(np.asarray(canvas)[:, :, None], 3, axis=2)

    def __call__(self, gray: np.ndarray) -> np.ndarray:
        landmark = self._landmark_image(gray)
        values = landmark.astype(np.float32) / 127.5 - 1.0
        x = self.torch.from_numpy(values.transpose(2, 0, 1)[None]).to(self.device)
        if self.device.type == "cuda":
            x = x.half()
        with self.torch.inference_mode():
            output = self.model(x)
        if isinstance(output, (tuple, list)):
            output = output[0]
        output = output.detach().float().cpu().numpy()[0]
        output = np.transpose(output, (1, 2, 0))
        output = np.clip((output + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
        gray_output = np.asarray(Image.fromarray(output, mode="RGB").convert("L").resize((100, 100), Image.Resampling.LANCZOS))
        if not np.any(gray_output):
            raise RuntimeError("GANonymization 没有生成有效人脸")
        return gray_output

    def close(self):
        self.face_mesh.close()
