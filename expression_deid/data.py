"""Dataset adapters shared by training, inference and evaluation."""

from __future__ import annotations

import csv
import json
import hashlib
import pickle
from io import BytesIO
from pathlib import Path

import numpy as np

from expression_deid.config import EXPR_SIZE, FERPLUS_DIR, KMU_ALIGNED_DIR, PPB_ALIGNED_DIR
from expression_deid.labels import EXPRESSION_NAMES
from expression_deid.normalize import (
    expression_input,
    iter_raf,
    load_gray,
    load_stats,
)
from expression_deid.prepare import iter_kmu_aligned


def require_torch():
    try:
        import torch  # type: ignore
        from torch.utils.data import Dataset  # type: ignore
    except (ImportError, OSError) as exc:  # pragma: no cover
        raise RuntimeError("数据加载需要 PyTorch，请先安装训练依赖。") from exc
    return torch, Dataset


def _to_01(gray: np.ndarray) -> np.ndarray:
    """Resize an image to 100x100 and return a float32 [0, 1] array."""
    from PIL import Image

    if gray.shape != (EXPR_SIZE, EXPR_SIZE):
        gray = np.asarray(
            Image.fromarray(gray, mode="L").resize(
                (EXPR_SIZE, EXPR_SIZE), Image.Resampling.BILINEAR
            )
        )
    return gray.astype(np.float32) / 255.0


class RAFExpressionDataset:
    """Lazy RAF-DB dataset.

    The class implements the small subset of the Dataset protocol needed by
    PyTorch without importing torch at module import time.
    """

    def __init__(self, split: str = "train", augment: bool = False, component_augment: bool = False, stats=None):
        _, Dataset = require_torch()
        self._dataset_base = Dataset
        self.items = list(iter_raf(split))
        if not self.items:
            raise RuntimeError(f"RAF {split} 划分为空")
        self.augment = bool(augment)
        self.component_augment = bool(component_augment)
        self.stats = stats or load_stats()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]
        gray = load_gray(path)
        values = expression_input(gray, self.stats, augment=self.augment, component_augment=self.component_augment)
        return values.astype(np.float32), int(label), str(path)


class CachedAnonymizedExpressionDataset:
    """Actual runtime outputs with an explicit source-preserving split."""

    def __init__(self, root, split="train", augment=False, stats=None, domain=None):
        self.root = Path(root)
        manifest = self.root / "manifest.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing anonymized manifest: {manifest}")
        records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.records = [r for r in records if r.get("split") == split and r.get("status") == "ok" and (domain is None or r.get("domain") == domain)]
        if not self.records:
            raise ValueError(f"No anonymized samples: split={split}, domain={domain}")
        partitions = {}
        for record in records:
            key = (record.get("domain"), record.get("source"))
            partitions.setdefault(key, set()).add(record.get("split"))
        if any(len(splits) > 1 for splits in partitions.values()):
            raise ValueError("A source image appears in multiple anonymized partitions")
        self.stats = stats or load_stats()
        self.augment = bool(augment)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        path = self.root / record["output"]
        values = expression_input(load_gray(path), self.stats, augment=self.augment)
        return values.astype(np.float32), int(record["label"]), str(path)


class KMUIdentityDataset:
    """Aligned KMU dataset with person ids and expression labels."""

    def __init__(self, root: Path = KMU_ALIGNED_DIR, include_review: bool = False):
        require_torch()
        self.items = [item for item in iter_kmu_aligned(root, include_review=include_review)]
        self.items = [item for item in self.items if item[1] >= 0 and item[2].get("person")]
        persons = sorted({record["person"] for _, _, record in self.items})
        if len(persons) < 2:
            raise RuntimeError("KMU 可用身份少于两个，无法训练身份模型")
        self.person_to_id = {person: idx for idx, person in enumerate(persons)}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, expression_label, record = self.items[index]
        values = _to_01(load_gray(path))[None, ...]
        identity = self.person_to_id[record["person"]]
        return values.astype(np.float32), int(identity), int(expression_label), str(path)


class KMUExpressionDataset:
    """KMU expression samples with the same normalized input contract as RAF."""

    def __init__(self, root: Path = KMU_ALIGNED_DIR, include_review: bool = False,
                 augment: bool = False, component_augment: bool = False, stats=None):
        require_torch()
        self.items = [item for item in iter_kmu_aligned(root, include_review=include_review)
                      if item[1] >= 0 and item[2].get("person")]
        if not self.items:
            raise RuntimeError("KMU 可用表情样本为空")
        self.augment = bool(augment)
        self.component_augment = bool(component_augment)
        self.stats = stats or load_stats()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, expression_label, _record = self.items[index]
        values = expression_input(load_gray(path), self.stats, augment=self.augment, component_augment=self.component_augment)
        return values.astype(np.float32), int(expression_label), str(path)


class PPBNeutralExpressionDataset:
    """Participant-level split friendly PPB-Emo central-IR neutral frames."""

    def __init__(self, root: Path = PPB_ALIGNED_DIR, include_review: bool = False,
                 augment: bool = False, component_augment: bool = False, stats=None):
        require_torch()
        self.root = Path(root)
        manifest = self.root / "manifest.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(f"还没有 PPB 对齐清单: {manifest}; 先运行 prepare-ppb")
        allowed = {"ok", "weak", "fit"} if include_review else {"ok"}
        self.items = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") in allowed and (self.root / record["output"]).is_file():
                self.items.append((self.root / record["output"], str(record["person"]), record))
        if not self.items:
            raise RuntimeError("PPB 对齐清单中没有可用中性帧")
        self.augment = bool(augment)
        self.component_augment = bool(component_augment)
        self.stats = stats or load_stats()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, _person, _record = self.items[index]
        values = expression_input(load_gray(path), self.stats, augment=self.augment, component_augment=self.component_augment)
        return values.astype(np.float32), 6, str(path)


def split_ppb_people(dataset: PPBNeutralExpressionDataset, validation_fraction: float = 0.2,
                     test_fraction: float = 0.2, seed: int = 0):
    """Split PPB by participant, keeping every frame of a person together."""
    if not 0 < validation_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("PPB 划分比例无效")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("PPB 验证集和测试集比例之和必须小于 1")
    people = sorted({str(record["person"]) for _path, _person, record in dataset.items})
    if len(people) < 3:
        raise ValueError("PPB 至少需要三个参与者才能划分")
    rng = np.random.default_rng(seed)
    people = [str(value) for value in np.asarray(people)[rng.permutation(len(people))]]
    n_test = max(1, int(round(len(people) * test_fraction))) if test_fraction else 0
    n_valid = max(1, int(round(len(people) * validation_fraction)))
    n_valid = min(n_valid, len(people) - n_test - 1)
    test_people = set(people[:n_test])
    valid_people = set(people[n_test:n_test + n_valid])
    train_people = set(people[n_test + n_valid:])
    train = [i for i, (_p, _person, record) in enumerate(dataset.items) if str(record["person"]) in train_people]
    valid = [i for i, (_p, _person, record) in enumerate(dataset.items) if str(record["person"]) in valid_people]
    test = [i for i, (_p, _person, record) in enumerate(dataset.items) if str(record["person"]) in test_people]
    return train, valid, test, {"train": sorted(train_people), "valid": sorted(valid_people), "test": sorted(test_people)}


class FERPlusExpressionDataset:
    """FER2013 images paired with Microsoft's high-agreement FER+ labels.

    The image files come from the public FER2013 mirror; the official FER+
    annotation CSV supplies crowd votes. Samples with fewer than
    ``min_votes`` votes for their winning basic expression are skipped to keep
    noisy/ambiguous labels out of domain adaptation.
    """

    _cache = {}
    _label_map = {
        "neutral": 6, "happiness": 3, "surprise": 0, "sadness": 4,
        "anger": 5, "disgust": 2, "fear": 1,
    }

    def __init__(self, split: str = "train", root: Path = FERPLUS_DIR,
                 min_votes: int = 5, augment: bool = False, component_augment: bool = False, stats=None):
        require_torch()
        if split not in {"train", "valid"}:
            raise ValueError("FER+ split 必须是 train 或 valid")
        self.root = Path(root)
        self.records = self._load_records(split, self.root, int(min_votes))
        if not self.records:
            raise RuntimeError(f"FER+ {split} 没有达到一致性阈值的样本")
        self.augment = bool(augment)
        self.component_augment = bool(component_augment)
        self.stats = stats or load_stats()

    @classmethod
    def _load_records(cls, split: str, root: Path, min_votes: int):
        key = (str(root.resolve()), split, int(min_votes))
        if key in cls._cache:
            return cls._cache[key]
        image_csv = root / "original" / "fer2013.csv"
        if not image_csv.is_file():
            image_csv = root / "fer2013.csv"
        label_file = root / "fer2013new.csv"
        if not image_csv.is_file() or not label_file.is_file():
            raise FileNotFoundError(
                f"缺少 FER+ 原始像素或标注: {image_csv} 或 {label_file}"
            )
        with label_file.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        with image_csv.open(encoding="utf-8", newline="") as handle:
            image_rows = list(csv.DictReader(handle))
        if len(image_rows) != len(rows):
            raise ValueError(f"FER+ 原始像素和标注数量不一致: {len(image_rows)} != {len(rows)}")
        if any(image.get('Usage') != row.get('Usage') for image,row in zip(image_rows,rows)):
            raise ValueError('FER+ 像素和标签的 Usage 划分顺序不一致')
        def pixel_key(value):
            pixels=np.fromstring(value,sep=' ',dtype=np.int16)
            if pixels.size != 2304 or (pixels<0).any() or (pixels>255).any():
                raise ValueError('FER2013 像素需要 2304 个 [0,255] 整数')
            return hashlib.sha256(pixels.astype(np.uint8).tobytes()).digest()
        heldout_pixels = {pixel_key(image['pixels']) for image in image_rows
                          if image.get('Usage') in ('PublicTest','PrivateTest')} if split == 'train' else set()
        usage = "Training" if split == "train" else "PublicTest"
        columns = tuple(cls._label_map)
        records = []
        for index, (image, row) in enumerate(zip(image_rows, rows)):
            if image.get("Usage") != usage or row.get("Usage") != usage:
                continue
            votes = {name: int(row.get(name, 0) or 0) for name in columns}
            name = max(columns, key=votes.get)
            total = sum(int(row.get(key,0) or 0) for key in (*columns,'contempt','unknown','NF'))
            if votes[name] < min_votes or total <= 0 or votes[name] <= total*.5:
                continue
            if heldout_pixels and pixel_key(image['pixels']) in heldout_pixels:
                continue
            records.append({
                "pixels": image["pixels"],
                "label": cls._label_map[name],
                "agreement": float(votes[name] / total),
                "source": f"fer{index:07d}.png",
            })
        cls._cache[key] = records
        return records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        from PIL import Image

        gray = np.fromstring(record["pixels"], sep=" ", dtype=np.uint8).reshape(48, 48)
        values = expression_input(gray, self.stats, augment=self.augment, component_augment=self.component_augment)
        return values.astype(np.float32), int(record["label"]), record["source"]


class FEREnhancedExpressionDataset:
    """Quality-filtered FER2013 mirror with explicit class and pixel columns.

    This loader is intentionally separate from the compact pickle mirror: the
    enhanced CSV keeps the image pixels and labels in the same row, so there is
    no split-order assumption when pairing annotations with images.
    """

    _cache = {}

    def __init__(self, split: str = "train", root: Path = FERPLUS_DIR,
                 quality_threshold: float = 0.55, augment: bool = False, component_augment: bool = False, stats=None):
        require_torch()
        if split not in {"train", "valid"}:
            raise ValueError("FER 增强集 split 必须是 train 或 valid")
        self.root = Path(root)
        self.records = self._load_records(split, self.root, float(quality_threshold))
        if not self.records:
            raise RuntimeError(f"FER 增强集 {split} 没有满足质量阈值的样本")
        self.augment = bool(augment)
        self.component_augment = bool(component_augment)
        self.stats = stats or load_stats()

    @classmethod
    def _load_records(cls, split: str, root: Path, quality_threshold: float):
        key = (str(root.resolve()), split, round(float(quality_threshold), 4))
        if key in cls._cache:
            return cls._cache[key]
        name = "ferplus_enhanced_train.csv" if split == "train" else "ferplus_enhanced_validation.csv"
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"缺少 FER 增强集文件: {path}")
        records = []
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if float(row.get("quality_score", 0.0) or 0.0) < quality_threshold:
                    continue
                # This mirror uses FER2013 order (angry=0, surprise=5),
                # whereas runtime uses RAF order (surprise=0, anger=5).
                fer_to_raf = {0: 5, 1: 2, 2: 1, 3: 3, 4: 4, 5: 0, 6: 6}
                source_label = int(row['emotion'])
                if source_label not in fer_to_raf:
                    continue
                label = fer_to_raf[source_label]
                pixels = row.get("pixels", "").strip()
                if len(pixels.split()) != 48 * 48:
                    continue
                records.append({
                    "pixels": pixels, "label": label,
                    "source": row.get("sample_id", "fer-enhanced"),
                    "quality": float(row.get("quality_score", 0.0) or 0.0),
                })
        cls._cache[key] = records
        return records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        from PIL import Image

        gray = np.asarray(
            Image.fromarray(np.fromstring(record["pixels"], sep=" ", dtype=np.uint8).reshape(48, 48), mode="L")
        )
        values = expression_input(gray, self.stats, augment=self.augment, component_augment=self.component_augment)
        return values.astype(np.float32), int(record["label"]), record["source"]


def split_indices(length: int, validation_fraction: float = 0.2, seed: int = 0):
    """Deterministic random image split; does not isolate people or sequences."""
    if length < 2:
        raise ValueError("数据量至少需要两张")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction 必须在 0 和 1 之间")
    rng = np.random.default_rng(seed)
    indices = np.arange(length)
    rng.shuffle(indices)
    cut = min(max(1, int(round(length * (1.0 - validation_fraction)))), length - 1)
    return indices[:cut].tolist(), indices[cut:].tolist()


def split_kmu_sequence_tails(items, validation_fraction=0.2):
    """Shared source-image boundary for training and the local donor bank.

    People and sequences still overlap; this is a tail-frame regression split.
    """
    if not 0 < validation_fraction < 1: raise ValueError('KMU 划分比例必须在 0 和 1 之间')
    groups={}
    for index, (_path, _label, record) in enumerate(items):
        groups.setdefault((record['person'],record['expression']),[]).append((int(record['frame']),index))
    train,valid=[],[]
    for values in groups.values():
        values.sort()
        count=min(max(1,int(round(len(values)*validation_fraction))),max(0,len(values)-1))
        if count:
            valid.extend(index for _,index in values[-count:])
            train.extend(index for _,index in values[:-count])
        else: train.extend(index for _,index in values)
    return train,valid
