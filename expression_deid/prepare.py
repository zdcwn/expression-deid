"""对齐 KMU，并写出 RAF 训练集的灰度均值和方差。

原图不动。对齐图写到 datasets/kmu-fed-aligned，不进仓库。
检不出、关键点不可信、帧间跳动都记在 manifest.jsonl 里。

    python -m expression_deid.prepare
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from expression_deid.align import (
    fit_is_extreme,
    landmarks_of,
    select_face,
    warp_gray,
)
from expression_deid.config import (
    ALIGN_SIZE,
    DETECT_SCORE,
    JITTER_PX,
    KMU_ALIGNED_DIR,
    KMU_DIR,
    STATS_PATH,
)
from expression_deid.detect import YuNet
from expression_deid.labels import parse_kmu_stem
from expression_deid.normalize import compute_raf_train_stats, load_gray, load_rgb, save_stats


def align_kmu(
    source_dir: Path = KMU_DIR,
    output_dir: Path = KMU_ALIGNED_DIR,
    limit: int = 0,
    detector: YuNet | None = None,
) -> list[dict]:
    sources = sorted(source_dir.glob("*.jpg"))
    if limit:
        sources = sources[:limit]
    if not sources:
        raise FileNotFoundError(f"没有找到 KMU 图片: {source_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "_partial.jsonl"
    done = _load_partial(partial_path, output_dir)
    pending = [path for path in sources if path.name not in done]
    print(f"已有 {len(done)} 张，待处理 {len(pending)} 张", flush=True)
    model = detector or YuNet() if pending else None
    records = []
    partial = partial_path.open("a", encoding="utf-8")
    try:
        for index, path in enumerate(sources, start=1):
            record = done.get(path.name)
            if record is None:
                record = _align_one(path, output_dir, model)
                partial.write(json.dumps(record, ensure_ascii=False) + "\n")
                partial.flush()
            records.append(record)
            if index % 50 == 0 or index == len(sources):
                print(f"对齐 {index}/{len(sources)}", flush=True)
    finally:
        partial.close()
    _mark_jitter(records)
    _write_reports(records, output_dir)
    partial_path.unlink(missing_ok=True)
    return records


def _load_partial(path: Path, output_dir: Path) -> dict[str, dict]:
    """中断后续跑。只认清单里、输出文件还在的记录。最后一行写了一半就丢掉。"""
    done: dict[str, dict] = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            break
        output = output_dir / record["output"]
        if record.get("status") == "fail" or output.exists():
            done[record["source"]] = record
    return done


def _align_one(path: Path, output_dir: Path, detector: YuNet) -> dict:
    meta = parse_kmu_stem(path.stem)
    output_path = output_dir / f"{path.stem}.png"
    record = {
        "source": path.name,
        "output": output_path.name,
        "person": meta["person"],
        "expression": meta["expression"],
        "name": meta["name"],
        "frame": meta["frame"],
        "label": meta["label"],
        "score": 0.0,
        "n_faces": 0,
        "box": [],
        "landmarks": [],
        "fit_error": None,
        "eye_shift": None,
        "status": "fail",
        "reason": "",
    }
    try:
        rgb = load_rgb(path)
    except OSError as exc:
        record["reason"] = f"read_error: {exc}"
        _drop_output(output_path)
        return record

    faces = detector.detect(rgb, DETECT_SCORE)
    record["n_faces"] = int(len(faces))
    face, reason = select_face(faces, rgb.shape[:2])
    if face is None:
        record["reason"] = reason
        _drop_output(output_path)
        return record

    gray = load_gray(path)
    aligned, fit_error = warp_gray(gray, landmarks_of(face), ALIGN_SIZE)
    Image.fromarray(aligned, mode="L").save(output_path)
    record.update(
        {
            "score": round(float(face[14]), 4),
            "box": [round(float(value), 2) for value in face[:4]],
            "landmarks": [[round(float(x), 2), round(float(y), 2)] for x, y in landmarks_of(face)],
            "fit_error": round(fit_error, 3),
            "status": "weak" if reason else "ok",
            "reason": reason,
        }
    )
    if fit_is_extreme(fit_error):
        record["status"] = "fit"
        record["reason"] = "fit_error"
    return record


def _drop_output(path: Path) -> None:
    if path.exists():
        path.unlink()


def _eye_center(record: dict) -> np.ndarray:
    points = np.asarray(record["landmarks"], dtype=np.float64)
    return (points[0] + points[1]) / 2


def _mark_jitter(records: list[dict]) -> None:
    """眼位相对该段中位数写在 eye_shift 里，只作记录。

    状态改成 jitter 的唯一情况：某一帧相对前后两帧都跳开，前后两帧彼此仍靠近。
    这是检测点抖了一帧。人在镜头里挪过去就停住，不标。
    """
    groups: dict[tuple, list[dict]] = {}
    for record in records:
        if record["status"] == "fail" or not record["landmarks"]:
            continue
        groups.setdefault((record["person"], record["expression"]), []).append(record)
    for group in groups.values():
        group.sort(key=lambda record: record["frame"])
        centers = [_eye_center(record) for record in group]
        median = np.median(np.stack(centers), axis=0)
        for record, center in zip(group, centers):
            record["eye_shift"] = round(float(np.linalg.norm(center - median)), 2)
        for index in range(1, len(group) - 1):
            jump_in = float(np.linalg.norm(centers[index] - centers[index - 1]))
            jump_out = float(np.linalg.norm(centers[index + 1] - centers[index]))
            span = float(np.linalg.norm(centers[index + 1] - centers[index - 1]))
            if jump_in > JITTER_PX and jump_out > JITTER_PX and span < JITTER_PX * 0.5:
                record = group[index]
                if record["status"] in {"ok", "weak"}:
                    record["status"] = "jitter"
                    record["reason"] = "eye_spike"


def _write_reports(records: list[dict], output_dir: Path) -> None:
    manifest = output_dir / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    counts = Counter(record["status"] for record in records)
    meta = {
        "align_size": ALIGN_SIZE,
        "gray": "rec601 luminance, png",
        "pixel_norm": "not baked in; expression_deid/norm_stats.json",
        "counts": dict(counts),
        "failures": [record["source"] for record in records if record["status"] == "fail"],
        "review": [
            record["source"]
            for record in records
            if record["status"] in {"weak", "jitter", "fit"}
        ],
    }
    (output_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "结果 "
        + ", ".join(f"{name}={counts[name]}" for name in ("ok", "weak", "jitter", "fit", "fail"))
    )
    if meta["failures"]:
        print("失败: " + ", ".join(meta["failures"]))


def load_manifest(output_dir: Path = KMU_ALIGNED_DIR) -> list[dict]:
    path = output_dir / "manifest.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"还没有对齐清单，先运行 python -m expression_deid.prepare。缺少 {path}")
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def iter_kmu_aligned(output_dir: Path = KMU_ALIGNED_DIR, include_review: bool = True):
    """产出 (对齐图路径, 类别 id, 记录)。失败的跳过。

    include_review 为假时，分数偏低、跳动、拟合差的也不产出。
    """
    allowed = {"ok"} if not include_review else {"ok", "weak", "jitter", "fit"}
    for record in load_manifest(output_dir):
        if record["status"] not in allowed:
            continue
        yield output_dir / record["output"], int(record["label"]), record


def ensure_stats(recompute: bool = False):
    if STATS_PATH.exists() and not recompute:
        print(f"沿用已有归一化参数 {STATS_PATH}")
        return
    print("计算 RAF 训练集灰度均值和方差")
    stats, images = compute_raf_train_stats()
    save_stats(stats, images)
    print(f"mean={stats.mean:.6f} std={stats.std:.6f} images={images}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="对齐 KMU，并计算 RAF 训练集的灰度均值和方差")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 张，用来试跑")
    parser.add_argument("--recompute-stats", action="store_true")
    parser.add_argument("--stats-only", action="store_true")
    args = parser.parse_args(argv)
    ensure_stats(recompute=args.recompute_stats or args.stats_only)
    if args.stats_only:
        return
    align_kmu(limit=args.limit)


if __name__ == "__main__":
    main()
