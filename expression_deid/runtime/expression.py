"""Accuracy-oriented expression recognition of already anonymous face pixels.

This backend deliberately has no access to a source frame, face detector,
identity gallery, donor selector or expression-labelled replacement images.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from expression_deid.config import ROOT
from expression_deid.labels import EXPRESSION_NAMES, EXPRESSION_ZH
from expression_deid.models import load_checkpoint, require_torch


def anonymous_tensor(gray: np.ndarray, *, device: str):
    """Use the exact float resize used during anonymous-domain fine tuning."""
    torch, _, F = require_torch()
    gray = np.asarray(gray)
    if gray.dtype != np.uint8 or gray.shape != (100, 100):
        raise ValueError('准确模式只接受匿名化后的 100×100 uint8 灰度人脸')
    x = torch.from_numpy(gray.copy()).to(device=device, dtype=torch.float32)[None, None] / 255
    views = torch.cat((x, x.flip(-1)), dim=0)
    resized = F.interpolate(views, size=(260, 260), mode='bilinear', align_corners=False)
    mean = torch.tensor([.485, .456, .406], device=device)[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device=device)[None, :, None, None]
    return views, (resized.expand(-1, 3, -1, -1) - mean) / std


class QualityExpressionPredictor:
    """Two complementary classifiers, calibrated on anonymous validation data."""

    def __init__(self, device='cpu', *, manifest: Path | None = None):
        torch, _, _ = require_torch()
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if str(device) == 'auto' else str(device)
        if self.device.startswith('cuda'):
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
        path = Path(manifest or ROOT / 'models/expression/expression_video_quality.json')
        self.metadata = json.loads(path.read_text(encoding='utf-8'))
        m = self.metadata
        if (m.get('schema') != 1 or m.get('expression_input_source') != 'anonymized_only'
                or m.get('class_names') != list(EXPRESSION_NAMES)
                or m.get('preprocessing') != 'float_bilinear_260_imagenet_gray_v1'):
            raise ValueError('准确模式模型的输入契约或类别顺序不匹配')
        self.weight = float(m['quality_weight'])
        self.quality_temperature = float(m['quality_temperature'])
        self.final_temperature = float(m.get('final_temperature', 1.0))
        self.compact_temperature = float(m['compact_temperature'])
        self.mean, self.std = float(m['compact_mean']), float(m['compact_std'])
        values = (self.quality_temperature, self.final_temperature, self.compact_temperature, self.std)
        if (not np.isfinite([self.weight, self.mean, *values]).all()
                or not 0 <= self.weight <= 1 or min(values) <= 0):
            raise ValueError('准确模式模型的校准参数无效')
        assets = {}
        for key in ('quality_model', 'compact_model'):
            asset = m[key]
            file = path.parent / asset['file']
            if not file.is_file():
                raise FileNotFoundError(f'缺少准确模式权重：{file}，请复制交接模型包')
            with file.open('rb') as handle:
                digest = hashlib.file_digest(handle, 'sha256').hexdigest()
            if digest != asset['sha256']:
                raise ValueError(f'准确模式权重校验失败：{file.name}')
            assets[key] = file
        self.quality = torch.jit.load(str(assets['quality_model']), map_location=self.device).eval()
        self.compact, _ = load_checkpoint(assets['compact_model'], device=self.device)
        self.compact.float().eval()

    def validate_pipeline(self, pipeline):
        if (pipeline.ganonymization_backend is not None
                or getattr(pipeline, 'machine_privacy', {}) != self.metadata.get('machine_privacy', {})):
            raise ValueError('准确模式权重与匿名化输入域不匹配，需要重新验证对应模型')

    def predict_anonymous(self, gray: np.ndarray, *, input_source: str) -> dict:
        probabilities = self.predict_anonymous_batch(np.asarray(gray)[None], input_source=input_source)[0]
        label = int(probabilities.argmax())
        return dict(label=label, name=EXPRESSION_NAMES[label], name_zh=EXPRESSION_ZH[label],
                    confidence=float(probabilities[label]), probabilities=probabilities.tolist(),
                    expression_input_source='anonymized_only', expression_backend='quality_tta_fp32')

    def predict_anonymous_batch(self, gray: np.ndarray, *, input_source: str) -> np.ndarray:
        """Return calibrated probabilities for a batch of anonymous faces."""
        if input_source != 'anonymized_only':
            raise ValueError('原始图像不能进入表情分类器')
        torch, _, _ = require_torch()
        gray = np.asarray(gray)
        if gray.ndim != 3 or gray.shape[1:] != (100, 100) or gray.dtype != np.uint8:
            raise ValueError('准确模式只接受匿名化后的 N×100×100 uint8 灰度人脸')
        with torch.inference_mode():
            images = torch.from_numpy(gray.copy()).to(device=self.device, dtype=torch.float32)[:, None] / 255
            views = torch.cat((images, images.flip(-1)), dim=0)
            resized = torch.nn.functional.interpolate(views, size=(260, 260), mode='bilinear', align_corners=False)
            mean = torch.tensor([.485, .456, .406], device=self.device)[None, :, None, None]
            std = torch.tensor([.229, .224, .225], device=self.device)[None, :, None, None]
            quality_input = (resized.expand(-1, 3, -1, -1) - mean) / std
            a = self.quality(quality_input).float()
            b = self.compact((views - self.mean) / self.std).float()
            if a.shape != (2 * len(gray), 7) or b.shape != (2 * len(gray), 7) or not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise RuntimeError('表情模型输出无效，停止分析当前帧')
            # `views` is laid out as all original faces followed by all
            # mirrored faces.  Pair by face before averaging; reshaping the
            # flat 2N rows as (N, 2, ...) would incorrectly pair face i with
            # face i+1 for every batch larger than one.
            qa = (a / self.quality_temperature).softmax(-1)
            qb = (b / self.compact_temperature).softmax(-1)
            p = self.weight * (qa[:len(gray)] + qa[len(gray):]) / 2
            p += (1 - self.weight) * (qb[:len(gray)] + qb[len(gray):]) / 2
            p = (p.clamp_min(1e-12).log() / self.final_temperature).softmax(-1)
            return p.cpu().numpy()
