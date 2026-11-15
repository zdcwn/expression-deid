"""Checkpoint expression diagnostics on explicitly anonymous pixels."""
from pathlib import Path

import numpy as np

from expression_deid.config import ROOT
from expression_deid.labels import EXPRESSION_NAMES, EXPRESSION_ZH
from expression_deid.models import load_checkpoint, require_torch
from expression_deid.preprocessing.normalize import load_stats


class MainExpressionPredictor:
    def __init__(self, device='cpu', *, checkpoint=None):
        torch, _, _ = require_torch()
        self.device = str(device)
        self.path = Path(checkpoint or ROOT/'models/expression/expression_current.pt')
        self.model, self.metadata = load_checkpoint(self.path, device=device)
        if self.metadata.get('kind') != 'expression' or self.model.classifier.out_features != 7:
            raise ValueError('Use a seven-class expression checkpoint')
        stats = load_stats()
        spec = self.metadata.get('metrics', {})
        self.mean = float(spec.get('normalization', {}).get('mean', stats.mean))
        self.std = float(spec.get('normalization', {}).get('std', stats.std))
        self.temperature = float(spec.get('calibration_temperature', 1.0))
        if not np.isfinite([self.mean, self.std, self.temperature]).all() or min(self.std, self.temperature) <= 0:
            raise ValueError('Invalid main expression input normalization or temperature')

    def predict_anonymous_batch(self, gray, *, input_source):
        if input_source != 'anonymized_only':
            raise ValueError('原始图像不能进入表情分类器')
        gray = np.asarray(gray)
        if gray.dtype != np.uint8 or gray.ndim != 3 or gray.shape[1:] != (100, 100) or not len(gray):
            raise ValueError('Expression input must be N x 100 x 100 uint8 anonymous faces')
        torch, _, _ = require_torch()
        with torch.inference_mode():
            x = torch.from_numpy(gray.copy()).to(self.device, dtype=torch.float32)[:, None]/255
            logits = self.model((x-self.mean)/self.std).float()/self.temperature
            if logits.shape != (len(gray), 7) or not torch.isfinite(logits).all():
                raise RuntimeError('Invalid expression output')
            return logits.softmax(1).cpu().numpy()

    def predict_anonymous(self, gray, *, input_source):
        probability = self.predict_anonymous_batch(np.asarray(gray)[None], input_source=input_source)[0]
        label = int(probability.argmax())
        return dict(label=label, name=EXPRESSION_NAMES[label], name_zh=EXPRESSION_ZH[label],
                    confidence=float(probability[label]), probabilities=probability.tolist(),
                    expression_input_source='anonymized_only', expression_backend='fer_primary_cnn')

    def validate_pipeline(self, pipeline):
        # This is an explicit new combination. Historical accuracy attached to
        # the main weights is not reused as an evaluation of this anonymizer.
        if getattr(pipeline, 'required_backend', None) is None:
            raise ValueError('The restored expression adapter requires the dataset pipeline')
