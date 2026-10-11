"""Joint evaluation of privacy and expression preservation."""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from expression_deid.config import KMU_ALIGNED_DIR
from expression_deid.inference import ExpressionDeidentifier
from expression_deid.prepare import iter_kmu_aligned


def evaluate_joint(
    *, anonymizer_checkpoint: str, expression_checkpoint: str,
    identity_checkpoint: str | None = None, root=KMU_ALIGNED_DIR,
    output_dir: str | None = None, limit: int = 0, device: str = "cpu",
    include_review: bool = False, prototype_strength: float = 0.60,
    reference_strength: float = 1.0, fast_model_only: bool = True,
    ganonymization_checkpoint: str | None = None,
    ganonymization_repo: str | None = None, generated_strength: float = 1.0,
    geometry_strength: float = 1.0,
) -> dict:
    pipeline = ExpressionDeidentifier(
        anonymizer_checkpoint, expression_checkpoint, identity_checkpoint,
        device=device, prototype_strength=prototype_strength,
        reference_strength=reference_strength,
        fast_mode=fast_model_only,
        ganonymization_checkpoint=ganonymization_checkpoint,
        ganonymization_repo=ganonymization_repo,
        generated_strength=generated_strength,
        geometry_strength=geometry_strength,
    )
    try:
        person_to_id = {}
        if pipeline.identity_meta:
            person_to_id = pipeline.identity_meta.get("metrics", {}).get("person_to_id", {})
        rows = []
        items = iter_kmu_aligned(Path(root), include_review=include_review)
        for index, (path, true_label, record) in enumerate(items, start=1):
            if limit and index > limit:
                break
            result = pipeline.process(path)
            anonymized = result.pop("anonymized")
            result.pop("original", None)
            expression_anonymized = result["expression_anonymized"]
            row = {
                "source": record["source"],
                "person": record["person"],
                "true_expression": int(true_label),
                "original_expression": None,
                "anonymized_expression": int(expression_anonymized["label"]),
                "original_confidence": None,
                "anonymized_confidence": float(expression_anonymized["confidence"]),
                "anonymized_raw_confidence": float(expression_anonymized.get("raw_confidence", expression_anonymized["confidence"])),
                "expression_confidence_change_raw": None,
                "expression_confidence_change_adjusted": None,
                "expression_original_correct": None,
                "expression_anonymized_correct": int(expression_anonymized["label"]) == int(true_label),
                "expression_consistent": None,
                "expression_input_source": result.get("expression_input_source", "anonymized_only"),
                "component_edit": bool(result.get("component_edit", False)),
                "background_preserved": bool(result.get("background_preserved", False)),
                "generated_strength": float(generated_strength),
                "geometry_strength": float(geometry_strength),
                "learned_anonymizer": bool(result.get("learned_anonymizer", False)),
            }
            if "identity_cosine" in result:
                row.update({
                    "identity_cosine": float(result["identity_cosine"]),
                    "identity_prediction_changed": bool(result["identity_prediction_changed"]),
                    "identity_region": result.get("identity_region", "face_only_ellipse"),
                    "identity_original_confidence": float(result["identity_original"]["confidence"]),
                    "identity_anonymized_confidence": float(result["identity_anonymized"]["confidence"]),
                    "identity_confidence_drop": float(result["identity_original"]["confidence"] - result["identity_anonymized"]["confidence"]),
                })
                if record["person"] in person_to_id:
                    expected_id = int(person_to_id[record["person"]])
                    row["identity_original_correct"] = int(result["identity_original"]["label"]) == expected_id
                    row["identity_anonymized_correct"] = int(result["identity_anonymized"]["label"]) == expected_id
            if output_dir:
                target = Path(output_dir)
                target.mkdir(parents=True, exist_ok=True)
                Image.fromarray(anonymized, mode="L").save(target / f"{Path(record['source']).stem}.png")
            rows.append(row)
            if index % 50 == 0:
                print(f"评估 {index}", flush=True)
        summary = summarize(rows)
        def asset(path):
            if not path: return None
            path = Path(path).resolve()
            with path.open('rb') as handle:
                digest = hashlib.file_digest(handle, 'sha256').hexdigest()
            return {'path': str(path), 'sha256': digest}
        result = {"summary": summary, "rows": rows, "provenance": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "anonymizer": asset(anonymizer_checkpoint), "expression": asset(expression_checkpoint),
            "identity": asset(identity_checkpoint), "gan": asset(ganonymization_checkpoint),
            "expression_ensemble_size": len(pipeline.expression_models), "fast_model_only": fast_model_only,
            "machine_privacy": pipeline.machine_privacy,
            "generated_strength": generated_strength, "geometry_strength": geometry_strength,
            "diagnostic_errors": pipeline.diagnostic_errors, "device": str(pipeline.device),
            "dataset_root": str(Path(root).resolve()),
            "scope": "Selected manifest samples; may include training identities/images. Not a held-out video or vehicle benchmark.",
        }}
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            (Path(output_dir) / "evaluation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
    finally:
        pipeline.close()


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("没有可评估样本")
    summary = {
        "samples": len(rows),
        "expression_anonymized_accuracy": float(np.mean([r["expression_anonymized_correct"] for r in rows])),
        "expression_input_source": "anonymized_only",
    }
    labels = sorted({int(row["true_expression"]) for row in rows if row.get("true_expression") is not None})
    per_class = {}
    confusion = {}
    for label in labels:
        subset = [row for row in rows if row.get("true_expression") is not None and int(row["true_expression"]) == label]
        predictions = Counter(int(row["anonymized_expression"]) for row in subset)
        per_class[str(label)] = {
            "samples": len(subset),
            "accuracy": float(np.mean([row["expression_anonymized_correct"] for row in subset])),
            "predicted": {str(key): int(value) for key, value in sorted(predictions.items())},
        }
        confusion[str(label)] = [int(predictions.get(index, 0)) for index in range(7)]
    summary["expression_anonymized_by_class"] = per_class
    summary["expression_confusion_matrix"] = confusion
    # Keep legacy metrics when evaluating an old result file, but never invent
    # source/anonymized agreement for the privacy-first runtime.
    legacy_rows = [r for r in rows if r.get("expression_original_correct") is not None]
    if legacy_rows:
        summary.update({
            "expression_original_accuracy": float(np.mean([r["expression_original_correct"] for r in legacy_rows])),
        })
        consistency = [r['expression_consistent'] for r in legacy_rows if r.get('expression_consistent') is not None]
        if consistency:
            summary['expression_consistency'] = float(np.mean(consistency))
    raw_changes = [r.get("expression_confidence_change_raw") for r in rows if r.get("expression_confidence_change_raw") is not None]
    adjusted_changes = [r.get("expression_confidence_change_adjusted") for r in rows if r.get("expression_confidence_change_adjusted") is not None]
    if raw_changes:
        summary["mean_expression_confidence_change_raw"] = float(np.mean(raw_changes))
    if adjusted_changes:
        summary["mean_expression_confidence_change_adjusted"] = float(np.mean(adjusted_changes))
    identity_rows = [row for row in rows if "identity_cosine" in row]
    if identity_rows:
        summary.update({
            "identity_pairs": len(identity_rows),
            "identity_mean_cosine": float(np.mean([r["identity_cosine"] for r in identity_rows])),
            "identity_median_cosine": float(np.median([r["identity_cosine"] for r in identity_rows])),
            "identity_prediction_change_rate": float(np.mean([r["identity_prediction_changed"] for r in identity_rows])),
        })
        if all("identity_confidence_drop" in row for row in identity_rows):
            summary["identity_mean_confidence_drop"] = float(np.mean([r["identity_confidence_drop"] for r in identity_rows]))
        if all("identity_original_correct" in row for row in identity_rows):
            summary.update({
                "identity_original_accuracy": float(np.mean([row["identity_original_correct"] for row in identity_rows])),
                "identity_anonymized_accuracy": float(np.mean([row["identity_anonymized_correct"] for row in identity_rows])),
            })
    return summary


def build_parser(subparsers):
    p = subparsers.add_parser("evaluate", help="联合评估身份保护和表情保持")
    p.add_argument("--anonymizer-checkpoint", required=True)
    p.add_argument("--expression-checkpoint", required=True)
    p.add_argument("--identity-checkpoint")
    p.add_argument("--root", default=str(KMU_ALIGNED_DIR))
    p.add_argument("--output-dir", default="outputs/evaluation")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--prototype-strength", type=float, default=0.60,
                   help="兼容旧配置；当前运行时采用局部特征编辑")
    p.add_argument("--reference-strength", type=float, default=1.0,
                   help="兼容旧配置；当前运行时不做整脸替换")
    p.add_argument("--include-review", action="store_true")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--fast-model-only", action="store_true", default=True,
                   help="只使用 expression-checkpoint，不混入旧 RAF/KMU companion 模型")
    group.add_argument('--expression-ensemble',dest='fast_model_only',action='store_false',
                       help='显式加入旧 RAF/joint 辅助模型；默认只用指定检查点')
    p.add_argument("--ganonymization-checkpoint", help="可选：用 landmark-only GAN 做匿名化评估")
    p.add_argument("--ganonymization-repo", help="GANonymization 源码目录")
    p.add_argument("--generated-strength", type=float, default=1.0,
                   help="landmark-only 生成内脸的覆盖比例，越高匿名越强")
    p.add_argument("--geometry-strength", type=float, default=1.0,
                   help="只改变水平身份比例的关键点变形强度")
    p.set_defaults(func=lambda a: evaluate_joint(anonymizer_checkpoint=a.anonymizer_checkpoint, expression_checkpoint=a.expression_checkpoint, identity_checkpoint=a.identity_checkpoint, root=a.root, output_dir=a.output_dir, limit=a.limit, device=a.device, include_review=a.include_review, prototype_strength=a.prototype_strength, reference_strength=a.reference_strength, fast_model_only=a.fast_model_only, ganonymization_checkpoint=a.ganonymization_checkpoint, ganonymization_repo=a.ganonymization_repo, generated_strength=a.generated_strength, geometry_strength=a.geometry_strength))
