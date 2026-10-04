"""Build an offline GAN-anonymized KMU expression set for domain adaptation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from expression_deid.config import KMU_ALIGNED_DIR
from expression_deid.inference import ExpressionDeidentifier
from expression_deid.prepare import iter_kmu_aligned


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "datasets" / "kmu-gan-aligned"))
    parser.add_argument("--checkpoint", default=str(ROOT / "models" / "ganonymization_50.ckpt"))
    parser.add_argument("--repo", default=str(ROOT / "external_ganonymization"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--anonymizer-checkpoint", default=str(ROOT / "models" / "anonymizer_adversarial_v1.pt"))
    parser.add_argument("--privacy-selected", action="store_true",
                        help="生成与运行时相同的身份筛选+局部纹理残差结果")
    parser.add_argument("--generated-strength", type=float, default=1.0,
                        help="landmark-only 生成内脸的覆盖比例")
    parser.add_argument("--geometry-strength", type=float, default=1.0,
                        help="只改变水平身份比例的关键点变形强度")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pipeline = ExpressionDeidentifier(
        args.anonymizer_checkpoint, ROOT / "models/expression_gan_strength100.pt",
        device=args.device, fast_mode=True,
        ganonymization_checkpoint=args.checkpoint, ganonymization_repo=args.repo,
        generated_strength=args.generated_strength,
        geometry_strength=args.geometry_strength,
    )
    rows = []
    items = list(iter_kmu_aligned(KMU_ALIGNED_DIR, include_review=False))
    people = sorted({str(record["person"]) for _path, _label, record in items})
    train_people = set(people[: max(1, int(round(len(people) * 0.8)))])
    for index, (path, label, record) in enumerate(items, start=1):
        if args.limit and index > args.limit:
            break
        with Image.open(path) as handle:
            gray = handle.convert("L")
            # Build exactly the aligned pixels that the runtime sends to the
            # expression classifier. No classifier call is made here.
            gray_array = np.asarray(gray)
            if args.privacy_selected:
                image = pipeline.select_anonymized_candidate(
                    gray_array, reference_gray=gray_array,
                )
            else:
                image = pipeline._generated_component_edit(gray_array)
        name = f"{Path(record['source']).stem}.png"
        target = output / name
        Image.fromarray(image, mode="L").save(target)
        rows.append({
            "source": record["source"], "output": name, "label": int(label),
            "person": record["person"], "expression": record["expression"],
            "status": "ok", "backend": (
                "GANonymization_50_machine_selector" if args.privacy_selected
                else "GANonymization_50"
            ),
            "domain": "kmu", "split": "train" if str(record["person"]) in train_people else "valid",
        })
        if index % 50 == 0:
            print(f"生成 {index}", flush=True)
    (output / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    (output / "meta.json").write_text(json.dumps({"count": len(rows), "backend": (
        "GANonymization_50_machine_selector" if args.privacy_selected else "GANonymization_50"
    ), "privacy_selected": bool(args.privacy_selected), "generated_strength": args.generated_strength, "geometry_strength": args.geometry_strength, "train_people": sorted(train_people), "valid_people": sorted(set(people) - train_people)}, indent=2), encoding="utf-8")
    print(json.dumps({"samples": len(rows), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
