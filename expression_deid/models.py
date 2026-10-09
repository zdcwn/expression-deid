"""Small PyTorch models used by the three project branches.

PyTorch is deliberately imported lazily.  The preprocessing and alignment
utilities stay usable in a light environment, while training and inference
fail with a clear message when the optional training dependency is missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def require_torch():
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
        import torch.nn.functional as F  # type: ignore
    except (ImportError, OSError) as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "模型训练需要 PyTorch，请先安装 requirements-train.txt 中的依赖。"
        ) from exc
    return torch, nn, F


def _build_models():
    torch, nn, F = require_torch()

    class ConvBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.block(x)

    class FaceBackbone(nn.Module):
        """Compact grayscale face encoder for 100x100 input."""

        def __init__(self, embedding_dim: int = 128):
            super().__init__()
            self.features = nn.Sequential(
                ConvBlock(1, 32), nn.MaxPool2d(2),
                ConvBlock(32, 64), nn.MaxPool2d(2),
                ConvBlock(64, 128), nn.MaxPool2d(2),
                ConvBlock(128, 192), nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.projection = nn.Sequential(
                nn.Flatten(), nn.Linear(192, embedding_dim), nn.ReLU(inplace=True),
                nn.Dropout(0.2),
            )

        def forward(self, x):
            return self.projection(self.features(x))

    class ExpressionNet(nn.Module):
        """Expression classifier with a reusable face embedding."""

        def __init__(self, num_classes: int = 7, embedding_dim: int = 128):
            super().__init__()
            self.backbone = FaceBackbone(embedding_dim)
            self.classifier = nn.Linear(embedding_dim, num_classes)

        def embedding(self, x):
            return F.normalize(self.backbone(x), dim=1)

        def forward(self, x):
            embedding = self.backbone(x)
            return self.classifier(embedding)

    class IdentityNet(nn.Module):
        """Identity verifier/classifier used only for privacy evaluation."""

        def __init__(self, num_identities: int, embedding_dim: int = 128):
            super().__init__()
            if num_identities < 2:
                raise ValueError("身份模型至少需要两个身份类别")
            self.backbone = FaceBackbone(embedding_dim)
            self.classifier = nn.Linear(embedding_dim, num_identities)

        def embedding(self, x):
            return F.normalize(self.backbone(x), dim=1)

        def forward(self, x):
            embedding = self.backbone(x)
            return self.classifier(embedding)

    class AnonymizerUNet(nn.Module):
        """Face anonymizer that retains a one-channel 100x100 interface.

        The residual output makes an untrained model well behaved.  Training
        supplies the identity suppression and expression preservation losses.
        """

        def __init__(self):
            super().__init__()
            self.enc1 = ConvBlock(1, 32)
            self.enc2 = ConvBlock(32, 64)
            self.enc3 = ConvBlock(64, 128)
            self.pool = nn.AvgPool2d(2)
            self.bottleneck = ConvBlock(128, 192)
            self.up3 = nn.ConvTranspose2d(192, 128, 2, stride=2)
            self.dec3 = ConvBlock(256, 128)
            self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.dec2 = ConvBlock(128, 64)
            self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.dec1 = ConvBlock(64, 32)
            self.output = nn.Conv2d(32, 1, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            b = self.bottleneck(self.pool(e3))
            d3 = self.up3(b)
            d3 = F.interpolate(d3, size=e3.shape[-2:], mode="bilinear", align_corners=False)
            d3 = F.avg_pool2d(d3, kernel_size=3, stride=1, padding=1)
            d3 = self.dec3(torch.cat([d3, e3], dim=1))
            d2 = self.up2(d3)
            d2 = F.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
            d2 = F.avg_pool2d(d2, kernel_size=3, stride=1, padding=1)
            d2 = self.dec2(torch.cat([d2, e2], dim=1))
            d1 = self.up1(d2)
            d1 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
            d1 = F.avg_pool2d(d1, kernel_size=3, stride=1, padding=1)
            d1 = self.dec1(torch.cat([d1, e1], dim=1))
            # Add a bounded residual instead of applying sigmoid to the input.
            # This keeps the original luminance range and avoids the bright,
            # banded output produced by sigmoid(x + residual).
            residual = 0.25 * torch.tanh(self.output(d1))
            return torch.clamp(x + residual, 0.0, 1.0)

    return {
        "FaceBackbone": FaceBackbone,
        "ExpressionNet": ExpressionNet,
        "IdentityNet": IdentityNet,
        "AnonymizerUNet": AnonymizerUNet,
    }


def model_class(name: str):
    models = _build_models()
    try:
        return models[name]
    except KeyError as exc:
        raise ValueError(f"未知模型类型: {name}") from exc


def __getattr__(name: str):
    """Expose model classes without importing PyTorch for preprocessing users."""
    if name in {"FaceBackbone", "ExpressionNet", "IdentityNet", "AnonymizerUNet"}:
        return model_class(name)
    raise AttributeError(name)


def checkpoint_payload(model, *, kind: str, epoch: int = 0, metrics: dict[str, Any] | None = None):
    """Return a portable checkpoint dictionary."""
    torch, _, _ = require_torch()
    return {
        "format": 1,
        "kind": kind,
        "epoch": int(epoch),
        "metrics": metrics or {},
        "model_config": _model_config(model, kind),
        "state_dict": model.state_dict(),
        "torch_version": str(torch.__version__),
    }


def _model_config(model, kind: str) -> dict[str, Any]:
    config: dict[str, Any] = {"kind": kind}
    if hasattr(model, "classifier"):
        config["num_classes"] = int(model.classifier.out_features)
        config["embedding_dim"] = int(model.classifier.in_features)
    return config


def save_checkpoint(model, path, *, kind: str, epoch: int = 0, metrics: dict[str, Any] | None = None):
    torch, _, _ = require_torch()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, kind=kind, epoch=epoch, metrics=metrics), output)


def load_checkpoint(path, *, device="cpu"):
    """Load a checkpoint and construct the correct model from its metadata."""
    torch, _, _ = require_torch()
    # Older local checkpoints stored TorchVersion (a str subclass). Allow that
    # type explicitly while retaining the restricted weights-only loader.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        payload = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get('state_dict'), dict):
        raise ValueError('模型检查点缺少 state_dict 字典')
    if any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all().item()
           for value in payload['state_dict'].values()):
        raise ValueError('模型检查点包含无效或非有限权重')
    kind = payload.get("kind") or payload.get("model_config", {}).get("kind")
    config = payload.get("model_config", {})
    if kind == "expression":
        model = model_class("ExpressionNet")(
            num_classes=int(config.get("num_classes", 7)),
            embedding_dim=int(config.get("embedding_dim", 128)),
        )
    elif kind == "identity":
        model = model_class("IdentityNet")(
            num_identities=int(config["num_classes"]),
            embedding_dim=int(config.get("embedding_dim", 128)),
        )
    elif kind == "anonymizer":
        model = model_class("AnonymizerUNet")()
    else:
        raise ValueError(f"检查点没有支持的 kind: {kind!r}")
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload


def tensor_from_input(array: np.ndarray, device="cpu"):
    torch, _, _ = require_torch()
    values = np.asarray(array, dtype=np.float32)
    if values.ndim == 2:
        values = values[None, None]
    elif values.ndim == 3:
        values = values[:, None]
    if values.ndim != 4 or values.shape[1] != 1:
        raise ValueError(f"模型输入应为 (N,1,H,W)，实际为 {values.shape}")
    return torch.from_numpy(values).to(device)
