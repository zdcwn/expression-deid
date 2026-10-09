"""Download the optional visual GAN and MediaPipe landmark assets."""
import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    "ganonymization_50.ckpt": "https://mediastore.rz.uni-augsburg.de/get/Sfle_etB1D/",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
}


def main():
    ledger = {}
    for name, url in ASSETS.items():
        target = ROOT / "models" / name
        temporary = target.with_suffix(target.suffix + ".part")
        if not target.is_file():
            start = time.monotonic()
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
                expected = int(response.headers.get("Content-Length", 0))
                count = 0
                next_report = 32 * 1024 * 1024
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    count += len(chunk)
                    if count >= next_report:
                        print(f"{name}: {count / 1e6:.1f}/{expected / 1e6:.1f} MB, {time.monotonic() - start:.1f}s", flush=True)
                        next_report += 32 * 1024 * 1024
                if expected and count != expected:
                    raise RuntimeError(f"Incomplete download: {count} != {expected}")
            temporary.replace(target)
        digest = hashlib.file_digest(target.open("rb"), "sha256").hexdigest()
        ledger[name] = {"url": url, "bytes": target.stat().st_size, "sha256": digest}
        print(f"{name}: complete", flush=True)
    (ROOT / "models" / "anonymization_assets.json").write_text(json.dumps(ledger, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
