"""Create a small, portable handoff package for local datasets and models.

The project keeps large datasets and checkpoints out of Git. This script
records repository-relative paths, the current machine's absolute paths,
file counts, sizes and the role of each asset, then zips only those manifests
and the handoff instructions. It never copies or deletes the actual data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DATASETS = [
    ("kmu-fed", "datasets/kmu-fed", "source", "KMU near-infrared cabin faces; 1,100 tracked source images", "academic-only; do not redistribute"),
    ("kmu-fed-aligned", "datasets/kmu-fed-aligned", "derived", "256x256 grayscale aligned KMU images used by the desktop examples", "rebuild with python -m expression_deid.prepare"),
    ("raf-db", "datasets/raf-db", "training", "RAF-DB aligned expression training images", "non-commercial; follow the dataset license"),
    ("fer2013-hf", "datasets/fer2013-hf", "training", "FER2013 + FER+ high-consensus expression labels", "follow source dataset and mirror terms"),
    ("ppb-emo", "datasets/ppb-emo", "optional-domain", "PPB-Emo central-IR neutral videos and extracted frames", "CC BY 4.0/source terms; do not redistribute blindly"),
    ("lfw-sklearn", "datasets/lfw-sklearn", "identity-source", "LFW source used to build the 1,680-person gallery", "follow LFW terms"),
    ("lfw-gallery", "datasets/lfw-gallery", "identity-gallery", "derived 1,680-person identity diagnostic gallery", "derived from LFW; diagnostic only"),
    ("kmu-machine-selected", "datasets/kmu-machine-selected", "derived-history", "machine-privacy selected KMU candidates and manifest", "generated evaluation artifact"),
    ("kmu-gan-aligned", "datasets/kmu-gan-aligned", "derived-history", "older GAN aligned candidates", "generated evaluation artifact"),
    ("kmu-gan-strength100", "datasets/kmu-gan-strength100", "derived-history", "older GAN strength sweep candidates", "generated evaluation artifact"),
]

MODELS = [
    ("anonymizer_adversarial_v1.pt", "models/anonymizer_adversarial_v1.pt", "required", "fast machine-targeted local anonymization"),
    ("expression_realtime_stable.pt", "models/expression_realtime_stable.pt", "required-for-current-video", "2026-10-09 audit-validated anonymous-domain and consistency fine-tuned realtime expression classifier"),
    ("expression_video_quality.ts.pt", "models/expression_video_quality.ts.pt", "required-for-accuracy-video", "FP32 TorchScript quality-priority video expression classifier; anonymous input only"),
    ("expression_video_quality.json", "models/expression_video_quality.json", "required-for-accuracy-video", "quality model contract, calibration, provenance and SHA-256"),
    ("expression_machine_selector_texture.pt", "models/expression_machine_selector_texture.pt", "required", "expression classifier used after anonymization"),
    ("ganonymization_50.ckpt", "models/ganonymization_50.ckpt", "required-for-visual-output", "landmark-only human-facing display anonymizer"),
    ("face_landmarker.task", "models/face_landmarker.task", "required-for-visual-output", "MediaPipe dense landmark asset"),
    ("face_detection_yunet_2023mar.onnx", "models/face_detection_yunet_2023mar.onnx", "required", "YuNet face detector"),
    ("identity_faceonly.pt", "models/identity_faceonly.pt", "diagnostic", "face-only identity diagnostic; never enters expression inference"),
    ("face_recognition_sface_2021dec.onnx", "models/face_recognition_sface_2021dec.onnx", "diagnostic", "SFace similarity diagnostic"),
    ("anonymizer.pt", "models/anonymizer.pt", "legacy-fallback", "older anonymizer checkpoint"),
    ("expression.pt", "models/expression.pt", "legacy-fallback", "older expression checkpoint"),
    ("identity.pt", "models/identity.pt", "legacy-fallback", "older identity checkpoint"),
]

OUTPUTS = [
    ("full_audit_20261008", "outputs/full_audit_20261008", "current-audit", "2026-10-08/09 audit evidence; contains experimental candidates, not all deployed"),
    ("evaluation_machine_runtime_fast_strong_1681", "outputs/evaluation_machine_runtime_fast_strong_1681", "historical-report", "older GAN configuration; not the current realtime result"),
    ("example_kmu_expressions.mp4", "outputs/example_kmu_expressions.mp4", "example", "desktop example video"),
    ("example_anger.mp4", "outputs/example_anger.mp4", "example", "desktop example video"),
    ("example_surprise.mp4", "outputs/example_surprise.mp4", "example", "desktop example video"),
    ("example_mixed.mp4", "outputs/example_mixed.mp4", "example", "desktop example video"),
]


def stats(path: Path) -> dict:
    if not path.exists():
        return {"exists": False, "files": 0, "bytes": 0, "sample_files": []}
    if path.is_file():
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        return {"exists": True, "files": 1, "bytes": path.stat().st_size, "sha256": digest, "sample_files": [path.name]}
    files = [item for item in path.rglob("*") if item.is_file()]
    return {
        "exists": True,
        "files": len(files),
        "bytes": sum(item.stat().st_size for item in files),
        "sample_files": [str(item.relative_to(path)) for item in sorted(files)[:5]],
    }


def record(name: str, relative_path: str, role: str, description: str, note: str = "") -> dict:
    path = ROOT / relative_path
    return {
        "name": name,
        "relative_path": relative_path,
        "absolute_path_on_source_machine": str(path.resolve()),
        "role": role,
        "description": description,
        "note": note,
        **stats(path),
    }


def write_text_manifest(path: Path, payload: dict) -> None:
    rows = []
    for group in ("datasets", "models", "outputs"):
        for item in payload[group]:
            rows.append({
                "group": group,
                "name": item["name"],
                "relative_path": item["relative_path"],
                "absolute_path_on_source_machine": item["absolute_path_on_source_machine"],
                "exists": item["exists"],
                "files": item["files"],
                "bytes": item["bytes"],
                "sha256": item.get("sha256", ""),
                "role": item["role"],
                "description": item["description"],
                "note": item["note"],
            })
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="写出项目数据、模型和输出的交接路径包")
    parser.add_argument("--output", default="handoff", help="交接文件输出目录")
    args = parser.parse_args(argv)
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_machine_root": str(ROOT.resolve()),
        "warning": "This package records paths and inventory only. Large datasets and checkpoints remain local and are not copied into Git.",
        "datasets": [record(*spec) for spec in DATASETS],
        "models": [record(*spec) for spec in MODELS],
        "outputs": [record(*spec) for spec in OUTPUTS],
    }
    json_path = output / "asset_manifest.json"
    csv_path = output / "asset_manifest.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_text_manifest(csv_path, payload)
    zip_path = output / "dataset-paths.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(json_path, "asset_manifest.json")
        archive.write(csv_path, "asset_manifest.csv")
        readme = ROOT / "handoff" / "README.md"
        if readme.is_file():
            archive.write(readme, "README.md")
        audit_report = ROOT / 'docs/full-audit-20261009.md'
        if audit_report.is_file(): archive.write(audit_report, 'full-audit-20261009.md')
        audit_result = ROOT / 'docs/audit-results-20261009.json'
        if audit_result.is_file(): archive.write(audit_result, 'audit-results-20261009.json')
    print(json.dumps({"manifest": str(json_path), "csv": str(csv_path), "zip": str(zip_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
