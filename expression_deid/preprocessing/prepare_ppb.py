"""Extract and align neutral central-infrared PPB-Emo frames.

The downloaded PPB-Emo archive stores short videos. This command samples a
small number of well-spaced frames from each ``*-ND.mp4`` video, aligns the
face to the same 256x256 grayscale format as KMU, and writes a participant
manifest so splits can be made by person rather than frame.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from expression_deid.preprocessing.align import fit_is_extreme, landmarks_of, select_face, warp_gray
from expression_deid.config import ALIGN_SIZE, PPB_ALIGNED_DIR, PPB_DIR, PPB_FIT_ERROR_PX
from expression_deid.preprocessing.detect import YuNet


def _frame_count(container, stream):
    if stream.frames: return int(stream.frames)
    duration = (float(stream.duration * stream.time_base) if stream.duration is not None
                else float(container.duration) / 1_000_000 if container.duration is not None else None)
    if duration is not None:
        count = round(duration * float(stream.average_rate or 30))
        if count > 0: return count
    # Local files with neither duration nor frame-count metadata can still be
    # decoded. Count them once and rewind, rather than silently sampling frame 1.
    count = sum(1 for _ in container.decode(stream))
    container.seek(0)
    if count <= 0: raise ValueError('视频中没有可解码帧')
    return count


def align_ppb_neutral(
    source_dir: Path = PPB_DIR / "raw-cir-neutral",
    output_dir: Path = PPB_ALIGNED_DIR,
    frames_per_video: int = 5,
) -> list[dict]:
    try:
        import av  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PPB-Emo 抽帧需要 PyAV，请先安装训练依赖。") from exc
    videos = sorted(source_dir.glob("*-ND.mp4"))
    if not videos:
        raise FileNotFoundError(f"没有找到 PPB-Emo 中央红外中性视频: {source_dir}")
    frames_per_video = max(1, min(15, int(frames_per_video)))
    output_dir.mkdir(parents=True, exist_ok=True)
    detector = YuNet()
    records: list[dict] = []
    for video_index, path in enumerate(videos, start=1):
        match = re.search(r"-P(\d+)-ND\.mp4$", path.name, re.IGNORECASE)
        person = match.group(1) if match else path.stem
        container = av.open(str(path))
        try:
            if not container.streams.video:
                raise ValueError(f"输入文件没有视频流: {path}")
            stream = container.streams.video[0]
            frame_count = _frame_count(container, stream)
            targets = np.linspace(int(frame_count * .08), min(frame_count-1, int(frame_count * .92)), frames_per_video, dtype=int)
            target_set = set(int(x) for x in targets)
            for index, frame in enumerate(container.decode(stream)):
                if index not in target_set:
                    continue
                rgb = frame.to_ndarray(format="rgb24")
                output_name = f"ppb_{person}_nd_{index:04d}.png"
                output_path = output_dir / output_name
                record = {
                    "source": path.name, "person": str(person), "frame": int(index),
                    "label": 6, "output": output_name, "status": "fail",
                    "score": 0.0, "fit_error": None, "reason": "",
                }
                faces = detector.detect(rgb)
                face, reason = select_face(faces, rgb.shape[:2])
                if face is None:
                    record["reason"] = reason
                    output_path.unlink(missing_ok=True)
                else:
                    gray = np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.uint8)
                    aligned, fit_error = warp_gray(gray, landmarks_of(face), ALIGN_SIZE)
                    Image.fromarray(aligned, mode="L").save(output_path)
                    quality_review = float(fit_error) > PPB_FIT_ERROR_PX
                    record.update({
                        "status": "fit" if quality_review else ("weak" if reason else "ok"),
                        "score": round(float(face[14]), 4),
                        "fit_error": round(float(fit_error), 3),
                        "reason": "fit_error" if quality_review else reason,
                    })
                records.append(record)
        finally:
            container.close()
        print(f"PPB 对齐 {video_index}/{len(videos)}: {path.name}", flush=True)
    manifest = output_dir / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    meta = {
        "source": "PPB-Emo facial_expression_data / video-face-CIR / *-ND.mp4",
        "align_size": ALIGN_SIZE, "frames_per_video": frames_per_video,
        "gray": "rec601 luminance, png", "records": len(records),
        "fit_error_threshold_px": PPB_FIT_ERROR_PX,
        "ok": sum(r["status"] == "ok" for r in records),
        "review": sum(r["status"] in {"weak", "fit"} for r in records),
        "fail": sum(r["status"] == "fail" for r in records),
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False), flush=True)
    return records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="抽取并对齐 PPB-Emo 中央红外中性帧")
    parser.add_argument("--source-dir", default=str(PPB_DIR / "raw-cir-neutral"))
    parser.add_argument("--output-dir", default=str(PPB_ALIGNED_DIR))
    parser.add_argument("--frames-per-video", type=int, default=5)
    args = parser.parse_args(argv)
    align_ppb_neutral(Path(args.source_dir), Path(args.output_dir), args.frames_per_video)


if __name__ == "__main__":
    main()
