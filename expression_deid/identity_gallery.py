"""Face-only SFace gallery diagnostics for open-set re-identification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


class SFaceGallery:
    """Compare an anonymized face with one enrolled source and LFW distractors.

    The source face is enrolled per query, so the result answers the concrete
    question: does the anonymized face still retrieve its original identity
    when a larger gallery is present?  The reported confidence is a relative
    gallery softmax score, not a calibrated real-world probability.
    """

    def __init__(self, root, recognizer, *, max_identities: int = 1680, temperature: float = 0.08):
        self.root = Path(root)
        self.recognizer = recognizer
        self.temperature = float(max(0.01, temperature))
        manifest = self.root / "manifest.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(f"身份干扰库清单不存在: {manifest}")
        records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        max_identities = max(1, int(max_identities))
        records = records[:max_identities]
        if len(records) < max_identities:
            raise ValueError(f"身份干扰库只有 {len(records)} 个身份，需要至少 {max_identities} 个")
        self.records = records
        if len({record['identity'] for record in records}) != len(records):
            raise ValueError("身份干扰库存在重复身份")
        embeddings = np.stack([self._feature(self._read(self._resolve_path(record["path"]))) for record in records]).astype(np.float32)
        if not np.isfinite(embeddings).all() or (np.linalg.norm(embeddings, axis=1) <= 1e-12).any():
            raise ValueError("身份图库嵌入包含无效数值")
        self.embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)

    def _resolve_path(self, value):
        path = Path(str(value).replace('\\', '/'))
        if path.is_absolute():
            relocated = self.root / path.name
            if relocated.is_file(): return relocated
            raise FileNotFoundError(f'当前身份图库缺少图片: {relocated}')
        if not path.is_absolute():
            path = self.root / path
        if not path.resolve().is_relative_to(self.root.resolve()):
            raise ValueError('身份图库图片路径超出当前图库目录')
        if path.is_file():
            return path
        # Old handoff archives used absolute Windows paths. The images live
        # beside their manifest, so a relocated copy can resolve by filename.
        relocated = self.root / Path(str(value).replace('\\', '/')).name
        if relocated.is_file():
            return relocated
        raise FileNotFoundError(f"身份图库图片不存在: {path}")

    @staticmethod
    def _read(path):
        with Image.open(path) as handle:
            return np.asarray(handle.convert("L"))

    @staticmethod
    def _face_view(image):
        # Import lazily to keep this utility usable without loading torch at
        # module import time.
        from expression_deid.inference import ExpressionDeidentifier
        return ExpressionDeidentifier._face_only_view(image)

    def _feature(self, image):
        import cv2
        gray = self._face_view(image)
        bgr = cv2.resize(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), (112, 112), interpolation=cv2.INTER_LINEAR)
        return np.asarray(self.recognizer.feature(bgr), dtype=np.float32).reshape(-1)

    def compare(self, original, anonymized):
        original_feature = self._feature(original)
        anonymized_feature = self._feature(anonymized)
        if (not np.isfinite(original_feature).all() or not np.isfinite(anonymized_feature).all()
                or np.linalg.norm(original_feature) <= 1e-12 or np.linalg.norm(anonymized_feature) <= 1e-12):
            raise ValueError("身份诊断收到无效嵌入")
        original_feature /= np.linalg.norm(original_feature) + 1e-12
        anonymized_feature /= np.linalg.norm(anonymized_feature) + 1e-12
        scores = np.concatenate((
            np.asarray([float(np.dot(original_feature, anonymized_feature))], dtype=np.float32),
            self.embeddings @ anonymized_feature,
        ))
        order = np.argsort(-scores, kind="stable")
        source_rank = int(np.flatnonzero(order == 0)[0]) + 1
        distractor_scores = scores[1:]
        best_distractor = float(np.max(distractor_scores))
        margin = float(scores[0] - best_distractor)
        logits = (scores - float(np.max(scores))) / self.temperature
        probabilities = np.exp(logits)
        probabilities /= float(np.sum(probabilities))
        top = int(order[0])
        return {
            "gallery_size": int(scores.size),
            "gallery_distractor_count": int(len(self.records)),
            "gallery_source_rank": source_rank,
            "gallery_match_original": bool(top == 0),
            "gallery_top_identity": "original" if top == 0 else self.records[top - 1]["identity"],
            "gallery_confidence": float(probabilities[0]),
            "gallery_source_cosine": float(scores[0]),
            "gallery_best_distractor_cosine": best_distractor,
            "gallery_margin": margin,
        }
