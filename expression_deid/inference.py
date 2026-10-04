"""Inference wrapper for one image or a KMU aligned frame."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from expression_deid.align import landmarks_of, select_face, similarity_matrix, template_for, warp_gray
from expression_deid.detect import YuNet
from expression_deid.models import load_checkpoint, require_torch
from expression_deid.normalize import iter_raf, load_gray, load_stats
from expression_deid.labels import EXPRESSION_NAMES, EXPRESSION_ZH
from expression_deid.config import KMU_ALIGNED_DIR, ROOT
from expression_deid.prepare import iter_kmu_aligned


class ExpressionDeidentifier:
    # Canonical aligned-face region used only by the identity diagnostics.
    # Hair, cabin background and the outer image corners are excluded from
    # both the source and anonymized comparison.
    FACE_DIAGNOSTIC_BOX = (12, 12, 88, 100)
    DEFAULT_IDENTITY_GALLERY_SIZE = 1680

    def __init__(self, anonymizer_checkpoint, expression_checkpoint, identity_checkpoint=None,
                 device="cpu", auto_align=True, prototype_strength=0.60,
                 reference_strength=1.0, fast_mode=False,
                 ganonymization_checkpoint=None, ganonymization_repo=None,
                 generated_strength=1.0, geometry_strength=1.0,
                 identity_gallery_size=DEFAULT_IDENTITY_GALLERY_SIZE,
                 privacy_candidates=6, privacy_selection=True):
        torch, _, _ = require_torch()
        self.device = torch.device(device)
        self.fast_mode = bool(fast_mode)
        self.anonymizer, self.anonymizer_meta = load_checkpoint(anonymizer_checkpoint, device=self.device)
        self.expression_model, self.expression_meta = load_checkpoint(expression_checkpoint, device=self.device)
        if self.anonymizer_meta.get("kind") != "anonymizer":
            raise ValueError("anonymizer_checkpoint 不是匿名化模型检查点")
        # Only checkpoints trained with the explicit identity-margin loss are
        # applied as a learned residual. Older checkpoints stay regression
        # references and do not silently change the runtime.
        self.learned_anonymizer_enabled = bool(
            self.anonymizer_meta.get("metrics", {}).get("identity_margin") is not None
        )
        if self.expression_meta.get("kind") != "expression":
            raise ValueError("expression_checkpoint 不是表情模型检查点")
        # RAF is more stable on neutral webcam faces while the joint model is
        # stronger on the KMU cabin domain. Average their calibrated
        # probabilities when companion checkpoints are available.
        self.expression_models = [(self.expression_model, self.expression_meta)]
        primary_path = Path(expression_checkpoint).resolve()
        companion_checkpoints = () if self.fast_mode else (ROOT / "models" / "expression_raf.pt", ROOT / "models" / "expression_joint.pt")
        for candidate in companion_checkpoints:
            if not candidate.is_file() or candidate.resolve() == primary_path:
                continue
            try:
                model, metadata = load_checkpoint(candidate, device=self.device)
                if metadata.get("kind") == "expression":
                    self.expression_models.append((model, metadata))
            except (OSError, RuntimeError, ValueError):
                continue
        if self.fast_mode and self.device.type == "cuda":
            # The realtime path has a single classifier.  FP16 reduces CUDA
            # latency and memory; offline/evaluation mode stays FP32.
            for model, _metadata in self.expression_models:
                model.half()
        self.identity_model = None
        self.identity_meta = None
        if identity_checkpoint:
            self.identity_model, self.identity_meta = load_checkpoint(identity_checkpoint, device=self.device)
            if self.identity_meta.get("kind") != "identity":
                raise ValueError("identity_checkpoint 不是身份模型检查点")
            if self.fast_mode and self.device.type == "cuda":
                self.identity_model.half()
        # The old closed-set identity classifier is useful for regression
        # checks, but it is trained on the same people as KMU validation and
        # can memorize hair/background cues.  SFace supplies a face-only
        # verifier for the privacy metric without touching the expression
        # inference path.
        # Realtime/video callers do not load an identity checkpoint and should
        # not pay the startup cost of SFace plus the 1680-person gallery.  The
        # still-image/evaluation path passes identity_checkpoint explicitly and
        # keeps the full diagnostic verifier.
        self.sface_model = None
        if identity_checkpoint:
            try:
                import cv2
                sface_path = ROOT / "models" / "face_recognition_sface_2021dec.onnx"
                if sface_path.is_file():
                    self.sface_model = cv2.FaceRecognizerSF_create(str(sface_path), "")
            except (ImportError, OSError, AttributeError, RuntimeError):
                self.sface_model = None
        self.identity_gallery = None
        self.identity_gallery_size = max(1, int(identity_gallery_size))
        if self.sface_model is not None:
            gallery_root = ROOT / "datasets" / "lfw-gallery"
            if gallery_root.is_dir():
                try:
                    from expression_deid.identity_gallery import SFaceGallery
                    self.identity_gallery = SFaceGallery(
                        gallery_root, self.sface_model,
                        max_identities=self.identity_gallery_size,
                    )
                except (OSError, ValueError, RuntimeError):
                    self.identity_gallery = None
        self.stats = load_stats()
        self.auto_align = bool(auto_align)
        self.detector = None
        self.expression_temperature = float(
            self.expression_meta.get("metrics", {}).get("calibration_temperature", 1.0)
        )
        if not np.isfinite(self.expression_temperature) or self.expression_temperature <= 0:
            self.expression_temperature = 1.0
        self.prototype_strength = float(np.clip(prototype_strength, 0.0, 0.9))
        # Blend only the inner face generated from landmarks.  Keeping this as
        # a runtime parameter lets evaluation select the strongest privacy
        # level that still preserves expression geometry.
        self.generated_strength = float(np.clip(generated_strength, 0.0, 1.0))
        self.privacy_candidates = max(1, int(privacy_candidates))
        self.privacy_selection = bool(privacy_selection)
        self._last_privacy_selection = {
            "used": False, "candidate_count": 1, "selected_index": 0,
            "sface_scores": [],
        }
        # Reference-face replacement is intentionally disabled.  It changes
        # the whole identity image and can also change the background when a
        # caller uploads a full frame.  Anonymization below is a residual edit
        # of the detected face, then composited back onto the untouched frame.
        self.reference_strength = 0.0
        self.reference_bank = self._load_reference_bank()
        self.component_bank = [
            (int(person), np.asarray(image, dtype=np.float32))
            for people in self.reference_bank.values()
            for person, images in people.items()
            for image in images
        ]
        self.component_people = np.asarray([person for person, _image in self.component_bank], dtype=np.int64)
        self.component_vectors = (
            np.stack([self._component_descriptor(image) for _person, image in self.component_bank])
            if self.component_bank else np.zeros((0, 1), dtype=np.float32)
        )
        self.ganonymization_backend = None
        if ganonymization_checkpoint:
            from expression_deid.ganonymization_backend import GANonymizationBackend
            self.ganonymization_backend = GANonymizationBackend(
                ganonymization_checkpoint, device=str(self.device), repo_root=ganonymization_repo,
                geometry_strength=geometry_strength,
            )
        self.expression_prototypes = {}
        self._last_anonymized_aligned = None
        self._last_source_aligned256 = None
        self._last_raw_gray = None
        self._last_face = None
        self._last_matrix = None
        self._last_landmarks = None
        self._last_was_raw = False
        self._last_display_aligned = None

    def _load_reference_bank(self):
        """Load aligned feature donors without using an expression prediction.

        Donors are used only for local eye/nose/mouth patches.  They are not
        selected by an expression label, so a runtime source frame never has
        to enter the expression classifier before anonymization.
        """
        bank = {}
        if (KMU_ALIGNED_DIR / "manifest.jsonl").exists():
            try:
                for path, label, _record in iter_kmu_aligned(KMU_ALIGNED_DIR, include_review=False):
                    if label < 0:
                        continue
                    person = int(_record["person"])
                    with Image.open(path) as handle:
                        gray = np.asarray(handle.convert("L").resize((100, 100), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
                    bank.setdefault(int(label), {}).setdefault(person, []).append(gray)
            except (OSError, ValueError, KeyError, TypeError):
                pass
        # KMU has no neutral class. Add a bounded RAF neutral donor pool as a
        # second camera-domain source. No classifier probe is performed here.
        try:
            neutral_paths = [path for path, label in iter_raf("train") if int(label) == 6][:300]
            for index, path in enumerate(neutral_paths):
                with Image.open(path) as handle:
                    gray = np.asarray(handle.convert("L").resize((100, 100), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
                bank.setdefault(6, {}).setdefault(-(index + 1), []).append(gray)
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return bank

    @staticmethod
    def _build_expression_prototypes(reference_bank):
        """Keep a mean-image fallback for installations without a reference match."""
        prototypes = {}
        for label, people in reference_bank.items():
            images = [image for records in people.values() for image in records]
            if images:
                prototypes[int(label)] = np.mean(images, axis=0).astype(np.float32)
        return prototypes

    @staticmethod
    def _reference_face(reference_bank, expression_label, identity_label, gray):
        """Pick a deterministic reference from another person.

        Deterministic selection prevents a video stream from flickering between
        random references while still distributing samples across the bank.
        """
        people = reference_bank.get(int(expression_label), {})
        if not people:
            return None
        excluded = None if identity_label is None else int(identity_label) + 1
        candidates = sorted(person for person in people if person != excluded)
        if not candidates:
            candidates = sorted(people)
        seed = int(np.rint(float(gray.mean()) * 1000.0 + float(gray.std()) * 100.0))
        person = candidates[seed % len(candidates)]
        images = people[person]
        image = images[seed % len(images)]
        return np.asarray(image, dtype=np.float32)

    @staticmethod
    @lru_cache(maxsize=4)
    def _feature_masks(size=100):
        """Soft masks for local facial features; hair and face outline stay intact."""
        from PIL import ImageDraw, ImageFilter
        masks = []
        for box, strength in (
            ((4, 14, 54, 62), 0.99),  # left eye/brow/glasses region
            ((46, 14, 96, 62), 0.99),  # right eye/brow/glasses region
            ((33, 33, 67, 70), 0.92),  # nose bridge and nose tip
            ((22, 60, 78, 88), 0.72),  # mouth, still softer to retain expression geometry
        ):
            mask = Image.new("L", (size, size), 0)
            ImageDraw.Draw(mask).ellipse(box, fill=255)
            mask = np.asarray(mask.filter(ImageFilter.GaussianBlur(2.2)), dtype=np.float32) / 255.0
            masks.append((mask * strength, box))
        return masks

    @classmethod
    def _component_descriptor(cls, image):
        """Build a compact normalized descriptor for vectorized donor search."""
        image = np.asarray(image, dtype=np.float32)
        parts = []
        for _mask, (x0, y0, x1, y1) in cls._feature_masks(image.shape[0]):
            patch = image[y0:y1, x0:x1]
            parts.append(((patch - float(patch.mean())) / (float(patch.std()) + 1e-4)).reshape(-1))
        return np.concatenate(parts).astype(np.float32)

    @staticmethod
    @lru_cache(maxsize=4)
    def _identity_skin_masks(size=100):
        """Masks for identity-bearing skin, away from expression apertures."""
        from PIL import Image, ImageDraw, ImageFilter
        masks = []
        for box, strength in (
            ((8, 45, 38, 82), 0.92),   # cheeks
            ((62, 45, 92, 82), 0.92),
            ((26, 8, 74, 22), 0.82),   # forehead below hairline
            ((30, 84, 70, 98), 0.88),  # chin
        ):
            mask = Image.new("L", (size, size), 0)
            ImageDraw.Draw(mask).ellipse(box, fill=255)
            mask = np.asarray(mask.filter(ImageFilter.GaussianBlur(1.8)), dtype=np.float32) / 255.0
            masks.append((mask * strength, box))
        return masks

    def _select_component_donor(self, source, identity_label=None):
        """Pick one aligned donor by feature appearance, never by expression label."""
        if not self.component_bank:
            return None
        source = np.asarray(source, dtype=np.float32)
        source_vector = self._component_descriptor(source)
        excluded = None if identity_label is None else int(identity_label) + 1
        scores = np.mean(np.abs(self.component_vectors - source_vector[None, :]), axis=1)
        valid = np.ones(len(self.component_bank), dtype=bool)
        if excluded is not None:
            valid &= self.component_people != excluded
        indices = np.flatnonzero(valid)
        if len(indices) == 0:
            indices = np.arange(len(self.component_bank))
        indices = indices[np.argsort(scores[indices], kind="stable")]
        # Select among the closest donors so expression-like geometry is kept,
        # while the identity source is still changed by local feature patches.
        pool = indices[: min(128, len(indices))]
        seed = int(np.rint(float(source.mean()) * 10000.0 + float(source.std()) * 1000.0))
        # Use a middle/far donor from the appearance-near pool.  The nearest
        # donor often leaves identity unchanged; a moderate distance changes
        # the eyes/nose visibly while keeping local expression geometry.
        midpoint = len(pool) // 2
        chosen = pool[midpoint + (seed % max(1, len(pool) - midpoint))]
        return np.asarray(self.component_bank[int(chosen)][1], dtype=np.float32)

    def _component_anonymize(self, gray, identity_label=None, donor_override=None):
        """Edit only local feature texture while keeping source expression geometry.

        Copying a donor patch pixel-for-pixel also copies its eyelid, brow and
        mouth geometry, which is exactly the information the expression model
        needs.  We therefore keep the source patch's low-frequency structure
        and transfer only donor high-frequency appearance.  A small amount of
        donor low-frequency tone makes the result visibly different without
        turning the operation into a whole-face replacement.
        """
        source = self._resize(np.asarray(gray, dtype=np.uint8)).astype(np.float32) / 255.0
        donor = donor_override
        if donor is None:
            donor = self._select_component_donor(source, identity_label=identity_label)
        if donor is None:
            return np.clip(np.rint(source * 255.0), 0, 255).astype(np.uint8)
        output = source.copy()
        # The eye and mouth geometry is expression-critical.  The donor is
        # used more strongly for texture than for its low-frequency shape.
        low_frequency_strength = (0.18, 0.18, 0.24, 0.12)
        texture_strength = (1.15, 1.15, 1.05, 0.90)
        for index, (mask, (x0, y0, x1, y1)) in enumerate(self._feature_masks(source.shape[0])):
            source_patch = source[y0:y1, x0:x1]
            donor_patch = donor[y0:y1, x0:x1]
            # Match local luminance/contrast before separating geometry from
            # texture.  A small box blur is sufficient at the 100x100 model
            # resolution and avoids a per-frame PIL filter allocation.
            donor_matched = (donor_patch - float(donor_patch.mean())) * (
                float(source_patch.std()) + 1e-4
            ) / (float(donor_patch.std()) + 1e-4) + float(source_patch.mean())
            source_low = self._box_blur(source_patch, radius=2)
            donor_low = self._box_blur(donor_matched, radius=2)
            donor_high = donor_matched - donor_low
            replacement_patch = (
                source_patch * (1.0 - low_frequency_strength[index])
                + donor_low * low_frequency_strength[index]
                + donor_high * texture_strength[index]
            )
            replacement = source.copy()
            replacement[y0:y1, x0:x1] = np.clip(replacement_patch, 0.0, 1.0)
            output = output * (1.0 - mask) + replacement * mask
        return np.clip(np.rint(output * 255.0), 0, 255).astype(np.uint8)

    @staticmethod
    def _box_blur(array: np.ndarray, radius: int = 2) -> np.ndarray:
        """Small edge-preserving-enough blur for local texture transfer."""
        radius = max(1, int(radius))
        padded = np.pad(np.asarray(array, dtype=np.float32), radius, mode="edge")
        height, width = array.shape
        total = np.zeros((height, width), dtype=np.float32)
        for dy in range(2 * radius + 1):
            for dx in range(2 * radius + 1):
                total += padded[dy:dy + height, dx:dx + width]
        side = float((2 * radius + 1) ** 2)
        return total / side

    def _generated_component_edit(self, aligned_gray):
        """Generate an inner-face residual while preserving the face boundary.

        The landmark-only generator receives geometry, never source pixels or a
        donor identity.  Its output is blended into the inner face ellipse so
        eyes, nose and mouth change visibly while hair, outline and background
        remain on the source canvas.
        """
        if self.ganonymization_backend is None:
            raise RuntimeError("No landmark-conditioned generator configured")
        generated = self.ganonymization_backend(np.asarray(aligned_gray, dtype=np.uint8)).astype(np.float32)
        from expression_deid.feature_geometry import warp_features
        source_frame = self._resize(np.asarray(aligned_gray, dtype=np.uint8))
        source = warp_features(source_frame, self.ganonymization_backend.last_landmarks, self.ganonymization_backend.last_anonymized_landmarks).astype(np.float32)
        # The released generator was trained on normalized RGB portraits and
        # therefore has a different grayscale exposure from near-IR cabin
        # frames.  Match only the generated face's mean/contrast before the
        # local composite; this is illumination transfer, not source pixels
        # entering the generator.
        yy, xx = np.ogrid[:100, :100]
        face_region = (((xx - 50.0) / 34.0) ** 2 + ((yy - 54.0) / 41.5) ** 2) <= 1.0
        source_mean = float(source[face_region].mean())
        source_std = float(source[face_region].std())
        generated_mean = float(generated[face_region].mean())
        generated_std = float(generated[face_region].std())
        generated = (generated - generated_mean) * ((source_std + 1.0) / (generated_std + 1.0)) + source_mean
        generated = np.clip(generated, 0.0, 255.0)
        # Blend only the inner face.  The surrounding hairline, jaw boundary
        # and all pixels outside the face mask are kept from the source.
        from PIL import ImageDraw, ImageFilter
        inner = Image.new("L", (100, 100), 0)
        ImageDraw.Draw(inner).ellipse((13, 16, 87, 97), fill=255)
        inner_mask = np.asarray(inner.filter(ImageFilter.GaussianBlur(1.8)), dtype=np.float32) / 255.0
        generated_strength = self.generated_strength
        output = source * (1.0 - generated_strength * inner_mask) + generated * (generated_strength * inner_mask)
        return np.clip(np.rint(output), 0, 255).astype(np.uint8)

    def _composite_component_edit(self, raw_gray, output, face, matrix, *, mask_mode="features"):
        """Composite a local aligned edit back into a raw grayscale frame."""
        from PIL import ImageDraw, ImageFilter
        height, width = raw_gray.shape
        edited256 = np.asarray(
            Image.fromarray(output, mode="L").resize((256, 256), Image.Resampling.BILINEAR)
        )
        feature_mask = np.zeros((100, 100), dtype=np.float32)
        if mask_mode == "features":
            for mask, _box in self._feature_masks(100):
                feature_mask = np.maximum(feature_mask, mask)
        elif mask_mode == "face":
            face_shape = Image.new("L", (100, 100), 0)
            ImageDraw.Draw(face_shape).ellipse((16, 13, 84, 96), fill=235)
            feature_mask = np.asarray(face_shape, dtype=np.float32) / 255.0
        else:
            raise ValueError(f"未知的合成掩膜模式: {mask_mode}")
        feature_mask = np.asarray(
            Image.fromarray(np.clip(feature_mask * 255.0, 0, 255).astype(np.uint8), mode="L")
            .resize((256, 256), Image.Resampling.BILINEAR)
            .filter(ImageFilter.GaussianBlur(2.0)),
            dtype=np.float32,
        ) / 255.0
        face_mask = Image.fromarray(
            np.clip(feature_mask * 255.0, 0, 255).astype(np.uint8), mode="L"
        )
        # PIL samples the input (aligned template) at each output (raw-frame)
        # pixel.  The aligned-to-raw compositing therefore uses the forward
        # raw->template similarity matrix; warp_gray uses its inverse in the
        # opposite direction.
        affine = tuple(float(v) for v in matrix.reshape(-1))
        edited_frame = np.asarray(
            Image.fromarray(edited256, mode="L").transform(
                (width, height), Image.Transform.AFFINE, affine,
                resample=Image.Resampling.BILINEAR, fillcolor=0,
            ), dtype=np.float32,
        )
        alpha = np.asarray(
            face_mask.transform(
                (width, height), Image.Transform.AFFINE, affine,
                resample=Image.Resampling.BILINEAR, fillcolor=0,
            ), dtype=np.float32,
        ) / 255.0
        x, y, box_w, box_h = [int(round(float(v))) for v in face[:4]]
        rect = np.zeros((height, width), dtype=np.float32)
        rect[max(0, y):min(height, y + box_h), max(0, x):min(width, x + box_w)] = 1.0
        alpha *= rect
        base_gray = np.asarray(Image.fromarray(raw_gray, mode="L"), dtype=np.float32)
        result = base_gray * (1.0 - alpha) + edited_frame * alpha
        return np.clip(np.rint(result), 0, 255).astype(np.uint8)

    @staticmethod
    def _image_gray(image) -> np.ndarray:
        if isinstance(image, (str, Path)):
            return load_gray(image)
        if isinstance(image, Image.Image):
            return np.asarray(image.convert("L"), dtype=np.uint8)
        array = np.asarray(image)
        if array.ndim == 3:
            array = np.asarray(Image.fromarray(array.astype(np.uint8)).convert("L"))
        if array.ndim != 2:
            raise ValueError(f"输入图像应为二维灰度或三维 RGB，实际为 {array.shape}")
        return np.clip(array, 0, 255).astype(np.uint8)

    @staticmethod
    def _image_rgb(image) -> np.ndarray:
        if isinstance(image, (str, Path)):
            with Image.open(image) as handle:
                return np.asarray(handle.convert("RGB"))
        if isinstance(image, Image.Image):
            return np.asarray(image.convert("RGB"))
        array = np.asarray(image)
        if array.ndim == 2:
            return np.repeat(array[:, :, None], 3, axis=2).astype(np.uint8)
        return np.asarray(Image.fromarray(array.astype(np.uint8)).convert("RGB"))

    def _prepare_gray(self, image, *, aligned: bool | None = None) -> np.ndarray:
        """Return an aligned face; raw large frames are detected automatically."""
        gray = self._image_gray(image)
        already_aligned = gray.shape in {(100, 100), (256, 256)}
        if aligned is True or (aligned is None and (not self.auto_align or already_aligned)):
            return self._resize(gray)
        rgb = self._image_rgb(image)
        if self.detector is None:
            self.detector = YuNet()
        faces = self.detector.detect(rgb)
        face, reason = select_face(faces, rgb.shape[:2])
        if face is None:
            raise RuntimeError(f"原始画面没有找到可信人脸: {reason}")
        aligned, fit_error = warp_gray(gray, landmarks_of(face))
        if fit_error > 25.0:
            raise RuntimeError(f"人脸关键点拟合误差过大: {fit_error:.2f}px")
        return self._resize(aligned)

    @staticmethod
    def _resize(gray: np.ndarray) -> np.ndarray:
        if gray.shape == (100, 100):
            return gray
        return np.asarray(Image.fromarray(gray, mode="L").resize((100, 100), Image.Resampling.BILINEAR))

    def _tensor(self, gray: np.ndarray, normalized: bool):
        torch, _, _ = require_torch()
        gray = self._resize(gray)
        values = gray.astype(np.float32) / 255.0
        if normalized:
            values = (values - float(self.stats.mean)) / float(self.stats.std)
        tensor = torch.from_numpy(values[None, None]).to(self.device)
        if self.fast_mode and self.device.type == "cuda":
            tensor = tensor.half()
        return tensor

    def _privacy_candidate_specs(self):
        """Return bounded geometry/coverage variants for identity-aware selection."""
        count = max(1, int(self.privacy_candidates))
        base_geometry = float(self.ganonymization_backend.geometry_strength) if self.ganonymization_backend is not None else 0.0
        base_generated = float(self.generated_strength)
        if count == 1 or self.ganonymization_backend is None:
            return [(base_geometry, base_generated, 0)]
        # Keep the normal candidate first for expression quality, then add
        # deliberately stronger landmark-only identity edits.  The selected
        # frame is still scored by SFace before expression inference, so a
        # stronger candidate is used only when it lowers identity similarity.
        geometry = [
            base_geometry,
            max(base_geometry, 1.20),
            max(base_geometry, 1.40),
            max(base_geometry, 1.60),
            max(base_geometry, 1.80),
            max(base_geometry, 2.00),
        ]
        generated = [base_generated] + [max(0.92, base_generated)] * 5
        profiles = [0, 1, 2, 3, 4, 5]
        return list(zip(geometry[:count], generated[:count], profiles[:count]))

    def _learned_component_edit(self, image):
        """Apply the identity-adversarial residual inside the face oval."""
        if not self.learned_anonymizer_enabled:
            return np.asarray(image, dtype=np.uint8)
        torch, _, _ = require_torch()
        source = self._resize(np.asarray(image, dtype=np.uint8)).astype(np.float32) / 255.0
        tensor = torch.from_numpy(source[None, None]).to(self.device)
        anonymizer_dtype = next(self.anonymizer.parameters()).dtype
        if self.fast_mode and self.device.type == "cuda" and anonymizer_dtype == torch.float16:
            tensor = tensor.half()
        with self._inference_mode():
            generated = self.anonymizer(tensor).float().detach().cpu().numpy()[0, 0]
        from PIL import Image, ImageDraw, ImageFilter
        mask = Image.new("L", (100, 100), 0)
        ImageDraw.Draw(mask).ellipse((13, 16, 87, 97), fill=255)
        alpha = np.asarray(mask.filter(ImageFilter.GaussianBlur(1.8)), dtype=np.float32) / 255.0
        edited = source * (1.0 - alpha) + np.clip(generated, 0.0, 1.0) * alpha
        return np.clip(np.rint(edited * 255.0), 0, 255).astype(np.uint8)

    @classmethod
    def _machine_texture_residual(cls, image, variant=0):
        """Add a bounded, local skin-texture residual for machine privacy.

        The residual is restricted to cheeks, forehead and chin.  Eye/mouth
        apertures and all geometry remain source-derived, so this is a texture
        perturbation rather than a replacement face or an expression label
        lookup.  Different deterministic variants give the privacy selector
        additional black-box candidates for the SFace score.
        """
        image = cls._resize(np.asarray(image, dtype=np.uint8)).astype(np.float32)
        yy, xx = np.mgrid[0:100, 0:100]
        seed = int(9173 + int(variant) * 7919)
        rng = np.random.default_rng(seed)
        phase_a, phase_b = rng.uniform(0.0, 2.0 * np.pi, size=2)
        freq_a = 0.07 + 0.012 * (int(variant) % 5)
        freq_b = 0.11 + 0.009 * ((int(variant) + 2) % 5)
        pattern = (
            np.sin(freq_a * xx + phase_a) +
            np.cos(freq_b * yy + phase_b) +
            0.5 * np.sin(0.05 * (xx + yy) + phase_a - phase_b)
        ) / 2.5
        mask = np.zeros((100, 100), dtype=np.float32)
        for skin_mask, _box in cls._identity_skin_masks(100):
            mask = np.maximum(mask, skin_mask)
        # Keep the first candidate unperturbed; later candidates are bounded
        # enough to look like illumination/texture variation to a viewer.
        amplitude = min(0.14, 0.035 + 0.021 * max(0, int(variant)))
        output = image + 255.0 * amplitude * pattern * mask
        return np.clip(np.rint(output), 0, 255).astype(np.uint8)

    def _machine_runtime_candidate(self, aligned_gray, variant=3):
        """Fast machine-private candidate without per-frame SFace screening."""
        if self.ganonymization_backend is None:
            return self._learned_component_edit(
                self._component_anonymize(
                    self._resize(aligned_gray),
                    donor_override=getattr(self, "_runtime_donor", None),
                )
            )
        backend_geometry = float(self.ganonymization_backend.geometry_strength)
        backend_profile = int(self.ganonymization_backend.geometry_profile)
        source_generated = float(self.generated_strength)
        try:
            self.ganonymization_backend.geometry_strength = max(backend_geometry, 2.00)
            self.ganonymization_backend.geometry_profile = 5
            candidate = self._generated_component_edit(aligned_gray)
            candidate = self._learned_component_edit(candidate)
            return self._machine_texture_residual(candidate, max(5, int(variant)))
        finally:
            self.ganonymization_backend.geometry_strength = backend_geometry
            self.ganonymization_backend.geometry_profile = backend_profile
            self.generated_strength = source_generated

    def select_anonymized_candidate(self, aligned_gray, reference_gray=None):
        """Generate local anonymized candidates and keep the least similar face.

        ``reference_gray`` is used only by the identity diagnostic (SFace).  The
        expression classifier is called later by the caller, exactly once, on
        the selected anonymized result.
        """
        if self.ganonymization_backend is None:
            selected = self._learned_component_edit(self._component_anonymize(aligned_gray))
            self._last_privacy_selection = {
                "used": False, "candidate_count": 1, "selected_index": 0,
                "sface_scores": [],
            }
            self._last_anonymized_aligned = np.asarray(selected, dtype=np.uint8)
            return selected
        backend_geometry = float(self.ganonymization_backend.geometry_strength)
        backend_profile = int(self.ganonymization_backend.geometry_profile)
        source_generated = float(self.generated_strength)
        candidates = []
        scores = []
        try:
            for geometry_strength, generated_strength, geometry_profile in self._privacy_candidate_specs():
                self.ganonymization_backend.geometry_strength = float(geometry_strength)
                self.ganonymization_backend.geometry_profile = int(geometry_profile)
                self.generated_strength = float(generated_strength)
                candidate = self._generated_component_edit(aligned_gray)
                candidate = self._learned_component_edit(candidate)
                candidate = self._machine_texture_residual(candidate, len(candidates))
                candidates.append(np.asarray(candidate, dtype=np.uint8))
                score = None
                if reference_gray is not None and self.sface_model is not None:
                    score = self.predict_sface_similarity(reference_gray, candidate)
                scores.append(score)
        finally:
            self.ganonymization_backend.geometry_strength = backend_geometry
            self.ganonymization_backend.geometry_profile = backend_profile
            self.generated_strength = source_generated
        valid_scores = [float(value) if value is not None and np.isfinite(value) else np.inf for value in scores]
        selected_index = int(np.argmin(valid_scores)) if candidates else 0
        selected = candidates[selected_index]
        self._last_privacy_selection = {
            "used": bool(any(np.isfinite(value) for value in valid_scores)),
            "candidate_count": len(candidates),
            "selected_index": selected_index,
            "sface_scores": [None if not np.isfinite(value) else float(value) for value in valid_scores],
        }
        self._last_anonymized_aligned = selected
        return selected

    def _human_facing_candidate(self, aligned_gray):
        """Create the separate visual output without using an expression label.

        This path is only for the image shown outside the classifier.  It uses
        the landmark-conditioned generator on the inner face and leaves the
        source face oval boundary, hair and background untouched.  The
        classifier path never consumes this image.
        """
        if self.ganonymization_backend is None:
            return self._component_anonymize(
                self._resize(aligned_gray),
                donor_override=getattr(self, "_runtime_donor", None),
            )
        backend_geometry = float(self.ganonymization_backend.geometry_strength)
        backend_profile = int(self.ganonymization_backend.geometry_profile)
        source_generated = float(self.generated_strength)
        try:
            # A stronger horizontal identity profile makes the displayed face
            # visually anonymous while preserving vertical expression cues.
            self.ganonymization_backend.geometry_strength = max(backend_geometry, 1.75)
            self.ganonymization_backend.geometry_profile = 2
            self.generated_strength = source_generated
            return self._generated_component_edit(aligned_gray)
        finally:
            self.ganonymization_backend.geometry_strength = backend_geometry
            self.ganonymization_backend.geometry_profile = backend_profile
            self.generated_strength = source_generated

    def anonymize(self, image, *, expression_label=None, identity_label=None,
                  aligned=None, selection_reference=None, purpose="display"):
        raw_gray = self._image_gray(image)
        is_raw = aligned is not True and self.auto_align and raw_gray.shape not in {(100, 100), (256, 256)}
        matrix = None
        original_rgb = None
        face = None
        if is_raw:
            original_rgb = self._image_rgb(image)
            if self.detector is None:
                self.detector = YuNet()
            faces = self.detector.detect(original_rgb)
            face, reason = select_face(faces, original_rgb.shape[:2])
            if face is None:
                raise RuntimeError(f"原始画面没有找到可信人脸: {reason}")
            landmarks = landmarks_of(face)
            aligned256, fit_error = warp_gray(raw_gray, landmarks)
            if fit_error > 25.0:
                raise RuntimeError(f"人脸关键点拟合误差过大: {fit_error:.2f}px")
            gray = self._resize(aligned256)
            matrix = similarity_matrix(landmarks, template_for(256))
        else:
            gray = self._prepare_gray(image, aligned=aligned)
            aligned256 = np.asarray(gray, dtype=np.uint8)
            landmarks = None
        self._last_source_aligned256 = np.asarray(aligned256, dtype=np.uint8)
        self._last_raw_gray = np.asarray(raw_gray, dtype=np.uint8) if is_raw else None
        self._last_face = face
        self._last_matrix = matrix
        self._last_landmarks = landmarks
        self._last_was_raw = bool(is_raw)
        # The default runtime is deliberately component based.  It never needs
        # an expression label from the source frame and never replaces the
        # complete face.  An explicitly supplied GANonymization checkpoint is
        # an offline/high-quality alternative: its generator receives only
        # landmarks and is composited inside the inner face ellipse.
        if (purpose == "machine" and self.privacy_selection
                and self.ganonymization_backend is not None and selection_reference is not None):
            # Offline privacy evaluation may select among landmark-only
            # candidates using SFace.  No expression label or source pixels
            # enter the generator, and the selected result is still composited
            # only inside the face region.
            output = self.select_anonymized_candidate(
                aligned256 if is_raw else raw_gray,
                reference_gray=selection_reference,
            )
        elif purpose == "machine" and self.ganonymization_backend is not None:
            output = self._machine_runtime_candidate(aligned256 if is_raw else raw_gray)
        elif purpose == "machine":
            # Realtime fallback: local component edit plus learned residual.
            # It does not copy a donor face or use an expression label.
            output = self._learned_component_edit(
                self._component_anonymize(gray, identity_label=identity_label)
            )
        elif self.ganonymization_backend is not None:
            output = self.select_anonymized_candidate(
                aligned256 if is_raw else raw_gray,
                reference_gray=selection_reference,
            )
        else:
            output = self._learned_component_edit(
                self._component_anonymize(gray, identity_label=identity_label)
            )
        # Keep the aligned anonymized face for process(); this avoids running
        # the detector a second time on the already anonymized raw frame.
        self._last_anonymized_aligned = np.asarray(output, dtype=np.uint8)
        if not is_raw:
            return output
        # Only the selected inner-face mask is composited back; hair and
        # background remain exact copies of the grayscale source frame.
        composited = self._composite_component_edit(
            raw_gray, output, face, matrix,
            mask_mode="face" if self.ganonymization_backend is not None else "features",
        )
        # The classifier must see the pixels actually displayed after pasting
        # the edits into the video, including boundary blending/resampling.
        # Reuse source landmarks for alignment; do not re-detect a source face.
        self._last_anonymized_aligned = self._resize(warp_gray(composited, landmarks)[0])
        return composited

    def _display_from_last_source(self):
        """Render the human-facing result from the source cached by anonymize."""
        if self._last_source_aligned256 is None:
            raise RuntimeError("没有可用于显示匿名化的源脸")
        aligned = self._human_facing_candidate(self._last_source_aligned256)
        if not self._last_was_raw:
            self._last_display_aligned = np.asarray(aligned, dtype=np.uint8)
            return self._last_display_aligned
        displayed = self._composite_component_edit(
            self._last_raw_gray, aligned, self._last_face, self._last_matrix,
            mask_mode="face" if self.ganonymization_backend is not None else "features",
        )
        self._last_display_aligned = self._resize(
            warp_gray(displayed, self._last_landmarks)[0]
        )
        return displayed

    @staticmethod
    def _blend_aligned_face(source, generated):
        """Blend a generated aligned face inside an inner ellipse, keeping hair."""
        from PIL import ImageDraw, ImageFilter
        source = ExpressionDeidentifier._resize(np.asarray(source, dtype=np.uint8)).astype(np.float32)
        generated = ExpressionDeidentifier._resize(np.asarray(generated, dtype=np.uint8)).astype(np.float32)
        mask = Image.new("L", (100, 100), 0)
        ImageDraw.Draw(mask).ellipse((16, 13, 84, 96), fill=235)
        alpha = np.asarray(mask.filter(ImageFilter.GaussianBlur(2.0)), dtype=np.float32) / 255.0
        return np.clip(np.rint(source * (1.0 - alpha) + generated * alpha), 0, 255).astype(np.uint8)

    def process(self, image, *, aligned: bool | None = None, diagnostics: bool = True):
        """Run the privacy-first runtime chain.

        The source frame is converted to grayscale and anonymized first.  The
        expression classifier is called exactly once, on the anonymized face.
        No source-frame expression prediction or expression-conditioned donor
        selection is performed.  ``diagnostics=False`` keeps identity/SFace and
        gallery scoring out of realtime video frames; those checks remain
        available for the image comparison and offline evaluation paths.
        """
        raw_input = self._image_rgb(image)
        was_raw = aligned is not True and self.auto_align and tuple(raw_input.shape[:2]) not in {(100, 100), (256, 256)}
        identity_original = None
        identity_face = None
        if diagnostics and self.identity_model is not None:
            # Identity evaluation is optional diagnostics only.  It never feeds
            # the expression classifier or the component donor selection.
            original_face = self._prepare_gray(image, aligned=aligned)
            identity_face = original_face
            identity_original = self.predict_identity(original_face, aligned=True)

        # Machine path: local residual/component edit, used by expression and
        # the internal privacy diagnostic.  It never sees the display output.
        machine_anonymized = self.anonymize(
            image,
            identity_label=identity_original["label"] if identity_original else None,
            aligned=aligned,
            selection_reference=identity_face,
            purpose="machine",
        )
        expression_input = (
            self._last_anonymized_aligned
            if self._last_anonymized_aligned is not None
            else self._prepare_gray(machine_anonymized, aligned=True)
        )
        expression_anonymized = self.predict_expression(expression_input, aligned=True)
        raw_anon_confidence = float(expression_anonymized["confidence"])
        expression_anonymized["raw_confidence"] = raw_anon_confidence
        # Human-facing branch is generated independently after the machine
        # branch.  It is never passed to the expression model.
        anonymized = self._display_from_last_source()
        display_aligned = self._last_display_aligned
        result = {
            "original": raw_input if was_raw else self._image_gray(image),
            "expression_original": None,
            "expression_anonymized": expression_anonymized,
            "expression_consistent": None,
            "expression_input_source": "anonymized_only",
            "expression_privacy_path": "machine_targeted_local_edit",
            "display_privacy_path": "human_facing_local_face_edit",
            "reference_swap": False,
            "reference_strength": self.reference_strength,
            "expression_confidence_change_raw": None,
            "expression_confidence_change_adjusted": None,
            "prototype_blend": False,
            "prototype_strength": self.prototype_strength,
            "face_only_edit": True,
            "component_edit": True,
            "background_preserved": bool(was_raw),
            "anonymized": anonymized,
            "privacy_selection": dict(self._last_privacy_selection),
            "learned_anonymizer": bool(self.learned_anonymizer_enabled),
        }
        if identity_original is not None:
            identity_anonymized = self.predict_identity(expression_input, aligned=True)
            identity_display = self.predict_identity(display_aligned, aligned=True)
            a = np.asarray(identity_original["embedding"], dtype=np.float64)
            b = np.asarray(identity_anonymized["embedding"], dtype=np.float64)
            sface_cosine = self.predict_sface_similarity(
                original_face,
                self._last_anonymized_aligned if self._last_anonymized_aligned is not None else anonymized,
            )
            display_sface_cosine = self.predict_sface_similarity(original_face, display_aligned)
            result.update({
                "identity_original": identity_original,
                "identity_anonymized": identity_anonymized,
                "identity_cosine": float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)),
                "sface_cosine": sface_cosine,
                "identity_region": "face_only_ellipse",
                "identity_prediction_changed": identity_original["label"] != identity_anonymized["label"],
                "identity_confidence_drop": float(identity_original["confidence"] - identity_anonymized["confidence"]),
                "identity_display": identity_display,
                "display_sface_cosine": display_sface_cosine,
            })
        if diagnostics and self.identity_gallery is not None:
            if identity_face is None:
                identity_face = self._prepare_gray(image, aligned=aligned)
            result.update(self.identity_gallery.compare(
                identity_face,
                self._last_anonymized_aligned if self._last_anonymized_aligned is not None else anonymized,
            ))
            display_gallery = self.identity_gallery.compare(identity_face, display_aligned)
            result.update({f"display_{key}": value for key, value in display_gallery.items()})
        return result

    def predict_expression(self, image, *, aligned=None):
        return self._predict_expression(image, aligned=aligned)

    @staticmethod
    def _shared_confidence(original_support: float, anonymized_confidence: float) -> float:
        """Confidence supported by both the source and edited face views."""
        return float(min(max(0.0, float(original_support)), max(0.0, float(anonymized_confidence))))

    def _predict_expression(self, image, *, aligned=None):
        gray = self._prepare_gray(image, aligned=aligned)
        x = self._tensor(gray, normalized=True)
        model_probabilities = []
        temperatures = []
        with self._inference_mode():
            for model, metadata in self.expression_models:
                temperature = float(metadata.get("metrics", {}).get("calibration_temperature", 1.0))
                if not np.isfinite(temperature) or temperature <= 0:
                    temperature = 1.0
                logits = model(x)
                model_probabilities.append(
                    self._softmax(logits / temperature).detach().cpu().numpy()[0]
                )
                temperatures.append(temperature)
        probabilities = np.mean(np.stack(model_probabilities, axis=0), axis=0)
        label = int(probabilities.argmax())
        return {"label": label, "name": EXPRESSION_NAMES[label], "name_zh": EXPRESSION_ZH[label], "confidence": float(probabilities[label]), "probabilities": probabilities.tolist(), "temperature": self.expression_temperature, "ensemble_size": len(model_probabilities), "model_temperatures": temperatures}

    def predict_identity(self, image, *, aligned=None):
        if self.identity_model is None:
            raise RuntimeError("没有加载 identity_checkpoint")
        gray = self._face_only_view(self._prepare_gray(image, aligned=aligned))
        x = self._tensor(gray, normalized=True)
        with self._inference_mode():
            logits = self.identity_model(x)
            embedding = self.identity_model.embedding(x).detach().cpu().numpy()[0]
            probabilities = self._softmax(logits).detach().cpu().numpy()[0]
        label = int(probabilities.argmax())
        return {"label": label, "confidence": float(probabilities[label]), "embedding": embedding.tolist(), "probabilities": probabilities.tolist()}

    def predict_sface_similarity(self, original, anonymized):
        """Return a face-recognition cosine for the privacy diagnostic only."""
        if self.sface_model is None:
            return None
        import cv2
        def aligned_bgr(image):
            gray = self._face_only_view(image)
            return cv2.resize(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), (112, 112), interpolation=cv2.INTER_LINEAR)
        try:
            first = self.sface_model.feature(aligned_bgr(original))
            second = self.sface_model.feature(aligned_bgr(anonymized))
            return float(self.sface_model.match(first, second, cv2.FaceRecognizerSF_FR_COSINE))
        except (cv2.error, ValueError, TypeError):
            return None

    @classmethod
    def _face_only_view(cls, image):
        """Crop a canonical face oval and fill outside it with face luminance."""
        from PIL import ImageDraw, ImageFilter
        gray = cls._resize(cls._image_gray(image)).astype(np.float32)
        x0, y0, x1, y1 = cls.FACE_DIAGNOSTIC_BOX
        crop = gray[y0:y1, x0:x1]
        mask = Image.new("L", (x1 - x0, y1 - y0), 0)
        ImageDraw.Draw(mask).ellipse((2, 0, x1 - x0 - 2, y1 - y0), fill=255)
        mask = np.asarray(mask.filter(ImageFilter.GaussianBlur(1.0)), dtype=np.float32) / 255.0
        inside = crop[mask > 0.5]
        fill = float(np.median(inside)) if inside.size else float(np.median(crop))
        crop = crop * mask + fill * (1.0 - mask)
        return np.asarray(Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8)).resize((100, 100), Image.Resampling.BILINEAR))

    def _inference_mode(self):
        torch, _, _ = require_torch()
        return torch.inference_mode()

    @staticmethod
    def _softmax(logits):
        torch, _, _ = require_torch()
        return torch.softmax(logits, dim=1)
