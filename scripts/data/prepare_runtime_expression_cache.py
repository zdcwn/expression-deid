"""Prepare anonymous faces from the exact desktop machine-expression path.

The ordinary prepared cache is the output of the anonymizer checkpoint.  The
desktop stream additionally composites the local feature edit back into the
aligned face before classification.  This script records that final
anonymous 100x100 face, without invoking expression inference, so a candidate
classifier can be fine-tuned on the same pixel domain it receives at runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from expression_deid.models import load_checkpoint  # noqa: E402
from expression_deid.runtime.pipeline import ExpressionDeidentifier  # noqa: E402


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(dataset: str, *, anonymizer: Path, expression: Path, device: str,
            output_name: str, force: bool = False, data_root: Path | None = None) -> dict:
    data_root = Path(data_root or ROOT / "datasets")
    prepared = data_root / (dataset if dataset == "fer2013" else f"{dataset}-auxiliary") / "prepared"
    if not prepared.is_dir():
        raise FileNotFoundError(f"缺少已整理的 {dataset} 辅助数据: {prepared}")
    source = prepared / "images.npy"
    labels = prepared / "labels.npy"
    splits = prepared / "splits.npy"
    if not all(path.is_file() for path in (source, labels, splits)):
        raise FileNotFoundError(f"{dataset} prepared 缓存不完整")
    output = prepared / output_name
    manifest = output.with_suffix(".json")
    if output.exists() and manifest.exists() and not force:
        record = json.loads(manifest.read_text(encoding="utf-8"))
        expected = {"source_images_sha256": digest(source), "labels_sha256": digest(labels),
                    "splits_sha256": digest(splits), "output_sha256": digest(output),
                    "anonymizer_sha256": digest(anonymizer)}
        if all(record.get(k) == v for k,v in expected.items()) and record.get("expression_input_source") == "anonymized_only":
            return record
        raise ValueError("Anonymous cache has changed; rebuild it explicitly")

    images = np.load(source, mmap_mode="r", allow_pickle=False)
    source_labels = np.load(labels, mmap_mode="r", allow_pickle=False)
    source_splits = np.load(splits, mmap_mode="r", allow_pickle=False)
    if images.dtype != np.uint8 or images.ndim != 3 or images.shape[1:] != (100, 100):
        raise ValueError(f"{dataset} 必须是 100x100 灰度对齐图")
    if not (len(images) == len(source_labels) == len(source_splits)):
        raise ValueError(f"{dataset} 的图像、标签和划分长度不一致")

    # Loading the expression checkpoint is required by the runtime contract,
    # but no expression prediction is called while preparing this cache.
    _model, expression_meta = load_checkpoint(expression, device=device)
    if expression_meta.get("kind") != "expression":
        raise ValueError("expression 必须是表情检查点")
    if expression_meta.get("metrics", {}).get("expression_input_source") != "anonymized_only":
        raise ValueError("表情检查点必须声明 anonymized_only")
    expected_anonymizer = expression_meta.get("metrics", {}).get("anonymizer_sha256")
    actual_anonymizer = digest(anonymizer)
    if expected_anonymizer and expected_anonymizer != actual_anonymizer:
        raise ValueError(
            "表情检查点与匿名化检查点不匹配: "
            f"expected {expected_anonymizer}, got {actual_anonymizer}"
        )
    pipe = ExpressionDeidentifier(anonymizer, expression, device=device, fast_mode=False)
    target = np.lib.format.open_memmap(
        output.with_suffix(".partial.npy"), mode="w+", dtype=np.uint8, shape=images.shape
    )
    try:
        for index in range(len(images)):
            # The input is already aligned.  anonymize() only creates the
            # local anonymous composite; it never calls the expression model.
            pipe.anonymize(np.asarray(images[index]), aligned=True, purpose="machine")
            anonymous = np.asarray(pipe._last_anonymized_aligned, dtype=np.uint8)
            if anonymous.shape != (100, 100):
                raise ValueError("运行时匿名缓存输出尺寸不是 100x100")
            target[index] = anonymous
            if (index + 1) % 500 == 0:
                print(json.dumps({"dataset": dataset, "prepared": index + 1}), flush=True)
        target.flush()
    finally:
        del target
        pipe.close()
    output.with_suffix(".partial.npy").replace(output)
    record = {
        "schema": 1,
        "dataset": dataset,
        "samples": int(len(images)),
        "shape": [100, 100],
        "expression_input_source": "anonymized_only",
        "producer": "ExpressionDeidentifier.anonymize(purpose=machine)",
        "anonymizer_sha256": actual_anonymizer,
        "expression_checkpoint_sha256": digest(expression),
        "source_images_sha256": digest(source),
        "labels_sha256": digest(labels),
        "splits_sha256": digest(splits),
        "output_sha256": digest(output),
    }
    manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("fer2013", "raf", "kmu"), nargs="+", default=("fer2013", "kmu"))
    parser.add_argument("--anonymizer", default="models/anonymization/anonymizer_celebahq.pt")
    parser.add_argument("--expression", default="models/expression/expression_current.pt")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-name", default="anonymous-runtime.npy")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--data-root", type=Path, default=ROOT/"datasets")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("指定 CUDA 但当前没有可用 GPU")
    for dataset in args.dataset:
        result = prepare(dataset, anonymizer=Path(args.anonymizer), expression=Path(args.expression),
                         device=args.device, output_name=args.output_name, force=args.force, data_root=args.data_root)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
