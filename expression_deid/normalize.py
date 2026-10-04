"""读图时的灰度转换和像素归一化。不写回原图。

均值和方差只从 RAF 训练集算一次。验证集和 KMU 用同一组数。
不要按单张图自己的均值方差归一，皱眉和侧窗的明暗会被抹掉，连续帧还会闪。
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from PIL import Image

from expression_deid.config import (
    AUG_BRIGHTNESS,
    AUG_ROTATION_DEG,
    EXPR_SIZE,
    RAF_DIR,
    STATS_PATH,
)


@dataclass(frozen=True)
class NormStats:
    mean: float
    std: float
    channels: int = 1

    def to_json(self) -> dict:
        return {
            "channels": self.channels,
            "space": "gray_0_1",
            "mean": self.mean,
            "std": self.std,
            "source": "datasets/raf-db/train",
            "reducer": "all pixels, population std",
        }


def to_gray(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)


def load_gray(path) -> np.ndarray:
    with Image.open(path) as image:
        return to_gray(image)


def load_rgb(path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def iter_raf(split: str):
    """产出 (路径, 类别 id)。split 是 train 或 valid。"""
    root = RAF_DIR / split
    if not root.is_dir():
        raise FileNotFoundError(f"找不到 RAF 划分: {root}")
    for label in range(7):
        folder = root / str(label)
        for path in sorted(folder.glob("*.jpg")):
            yield path, label


def compute_raf_train_stats() -> tuple[NormStats, int]:
    total = 0.0
    total_sq = 0.0
    pixels = 0
    images = 0
    for path, _label in iter_raf("train"):
        values = load_gray(path).astype(np.float64).ravel() / 255.0
        total += float(values.sum())
        total_sq += float(np.dot(values, values))
        pixels += int(values.size)
        images += 1
    if pixels == 0:
        raise RuntimeError("RAF 训练集是空的，没法算均值和方差")
    mean = total / pixels
    variance = total_sq / pixels - mean * mean
    stats = NormStats(mean=round(mean, 6), std=round(float(np.sqrt(max(variance, 0.0))), 6))
    return stats, images


def save_stats(stats: NormStats, images: int, path=STATS_PATH) -> None:
    payload = stats.to_json()
    payload["images"] = images
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_stats(path=STATS_PATH) -> NormStats:
    if not path.exists():
        raise FileNotFoundError(f"还没有归一化参数，先运行 python -m expression_deid.prepare。缺少 {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("channels") != 1 or payload.get("space") != "gray_0_1":
        raise ValueError(f"归一化参数不是单通道灰度: {path}")
    return NormStats(mean=float(payload["mean"]), std=float(payload["std"]), channels=1)


def normalize(gray: np.ndarray, stats: NormStats) -> np.ndarray:
    values = gray.astype(np.float32) / 255.0
    return (values - stats.mean) / stats.std


def expression_input(
    gray: np.ndarray,
    stats: NormStats,
    augment: bool = False,
    component_augment: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """表情模型的输入，形状 (1, 100, 100)。

    augment 只给 RAF 训练集。几何变换在缩小之前做，亮度在除以 255 之后、减均值之前。
    """
    image = gray
    generator = rng
    if augment or component_augment:
        generator = rng if rng is not None else np.random.default_rng()
    if augment:
        image = _geometric_augment(image, generator)
    if image.shape[0] != EXPR_SIZE or image.shape[1] != EXPR_SIZE:
        image = np.asarray(
            Image.fromarray(image, mode="L").resize(
                (EXPR_SIZE, EXPR_SIZE), Image.Resampling.BILINEAR
            )
        )
    if component_augment:
        assert generator is not None
        image = _component_augment(image, generator)
    values = image.astype(np.float32) / 255.0
    if augment:
        assert generator is not None
        delta = float(generator.uniform(-AUG_BRIGHTNESS, AUG_BRIGHTNESS))
        values = np.clip(values + delta, 0.0, 1.0)
    values = (values - np.float32(stats.mean)) / np.float32(stats.std)
    return values[None, ...]


def _component_augment(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate local eye/nose/mouth edits while keeping expression labels.

    The live anonymizer replaces only these regions.  Training with bounded
    local donor-like perturbations makes the classifier rely on expression
    geometry that remains outside any one patch instead of overfitting to the
    identity texture in the original face.
    """
    image = np.asarray(gray, dtype=np.float32).copy()
    height, width = image.shape
    boxes = (
        (12, 20, 48, 47, 0.80),
        (52, 20, 88, 47, 0.80),
        (33, 33, 67, 70, 0.70),
        (22, 60, 78, 88, 0.55),
    )
    for x0, y0, x1, y1, strength in boxes:
        if rng.random() > 0.72:
            continue
        patch = image[y0:y1, x0:x1].copy()
        if rng.random() < 0.5:
            patch = np.fliplr(patch)
        contrast = float(rng.uniform(0.75, 1.25))
        shift = float(rng.uniform(-0.08, 0.08) * 255.0)
        patch = np.clip((patch - patch.mean()) * contrast + patch.mean() + shift, 0.0, 255.0)
        if rng.random() < 0.45:
            # A small 3x3 box blur is enough to mimic the softened donor seam
            # and is much faster than constructing a PIL Gaussian filter for
            # every sample in a multi-domain training epoch.
            padded = np.pad(patch, 1, mode="edge")
            patch = (
                padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
                padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
                padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
            ) / 9.0
        yy, xx = np.ogrid[:height, :width]
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        rx, ry = max(1.0, (x1 - x0) / 2.0), max(1.0, (y1 - y0) / 2.0)
        distance = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
        mask = np.clip((1.0 - distance) * 4.0, 0.0, 1.0) * float(strength)
        replacement = image.copy()
        replacement[y0:y1, x0:x1] = patch
        image = image * (1.0 - mask) + replacement * mask
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def _geometric_augment(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    image = gray
    if rng.random() < 0.5:
        image = np.ascontiguousarray(np.fliplr(image))
    angle = float(rng.uniform(-AUG_ROTATION_DEG, AUG_ROTATION_DEG))
    rotated = Image.fromarray(image, mode="L").rotate(
        angle, resample=Image.Resampling.BILINEAR, fillcolor=0
    )
    return np.asarray(rotated)
