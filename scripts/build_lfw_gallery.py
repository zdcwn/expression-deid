"""Build the LFW identity gallery for face-only diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="datasets/lfw-sklearn/lfw_home/lfw_funneled")
    parser.add_argument("--output", default="datasets/lfw-gallery")
    parser.add_argument("--identities", type=int, default=1680,
                        help="身份数；LFW 当前可用上限为 1680")
    args = parser.parse_args()
    root = Path(args.root)
    output = Path(args.output)
    candidates = []
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        images = sorted(directory.glob("*.jpg"))
        if len(images) >= 2:
            candidates.append((len(images), directory.name, images[0]))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    chosen = candidates[: int(args.identities)]
    if len(chosen) < int(args.identities):
        raise RuntimeError(f"可用身份只有 {len(chosen)}，少于要求的 {args.identities}")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for _count, identity, source in chosen:
        target = output / f"{identity}.jpg"
        target.write_bytes(source.read_bytes())
        records.append({"identity": identity, "path": str(target.resolve()), "source": str(source.resolve())})
    (output / "manifest.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8"
    )
    (output / "meta.json").write_text(json.dumps({"identities": len(records), "source": "LFW", "min_images_per_identity": 2}, indent=2), encoding="utf-8")
    print(json.dumps({"identities": len(records), "output": str(output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
