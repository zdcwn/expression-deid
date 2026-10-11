"""Build an offline GAN-anonymized KMU expression set for domain adaptation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

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
    parser.add_argument("--generated-strength", type=float, default=1.0,
                        help="landmark-only 生成内脸的覆盖比例")
    parser.add_argument("--geometry-strength", type=float, default=1.0,
                        help="只改变水平身份比例的关键点变形强度")
    args = parser.parse_args()
    if args.limit < 0: parser.error("--limit 不能为负数")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pipeline = ExpressionDeidentifier(
        args.anonymizer_checkpoint, ROOT / "models/expression_machine_selector_texture.pt",
        device=args.device, fast_mode=True,
        ganonymization_checkpoint=args.checkpoint, ganonymization_repo=args.repo,
        generated_strength=args.generated_strength,
        geometry_strength=args.geometry_strength,
    )
    try:
        rows = []
        items = list(iter_kmu_aligned(KMU_ALIGNED_DIR, include_review=False))
        from expression_deid.data import split_kmu_sequence_tails
        train_indices,valid_indices=split_kmu_sequence_tails(items)
        train_indices=set(train_indices)
        train_people={str(items[i][2]["person"]) for i in train_indices}
        for index, (path, label, record) in enumerate(items, start=1):
            if args.limit and index > args.limit:
                break
            # No expression-model call. Cache the actual machine composite after
            # re-alignment, rather than the generator's uncomposited pixels.
            gray_array = pipeline._image_gray(path)
            pipeline.anonymize(gray_array, aligned=True, purpose="machine")
            image = pipeline._last_anonymized_aligned.copy()
            name = f"{Path(record['source']).stem}.png"
            target = output / name
            Image.fromarray(image, mode="L").save(target)
            rows.append({
                "source": record["source"], "output": name, "label": int(label),
                "person": record["person"], "expression": record["expression"],
                "status": "ok", "backend": "GANonymization_50",
                "domain": "kmu", "split": "train" if index-1 in train_indices else "valid",
                "expression_input_source": "anonymized_only", "composited_and_realigned": True,
            })
            if index % 50 == 0:
                print(f"生成 {index}", flush=True)
        (output / "manifest.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
        (output / "meta.json").write_text(json.dumps({"count": len(rows), "backend": "GANonymization_50", "generated_strength": args.generated_strength, "geometry_strength": args.geometry_strength, "split_protocol": "source_sequence_tail_frames", "train_people": sorted(train_people), "valid_people": sorted({str(items[i][2]["person"]) for i in valid_indices})}, indent=2), encoding="utf-8")
        print(json.dumps({"samples": len(rows), "output": str(output)}, ensure_ascii=False))
    finally:
        pipeline.close()



if __name__ == "__main__":
    main()
