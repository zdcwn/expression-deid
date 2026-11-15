"""Download the two user-specified Kaggle datasets and record their origin."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    'celebahq': 'badasstechie/celebahq-resized-256x256',
    'pins': 'hereisburak/pins-face-recognition',
}


def download(name: str, root: Path):
    import requests
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / 'download.zip'
    partial = folder / 'download.zip.part'
    url = 'https://www.kaggle.com/api/v1/datasets/download/' + SOURCES[name]
    if not archive.is_file():
        for attempt in range(3):
            try:
                with requests.get(url, stream=True, timeout=(20, 90)) as response:
                    response.raise_for_status()
                    if 'zip' not in response.headers.get('Content-Type', ''):
                        raise RuntimeError('Kaggle did not return a ZIP; download it manually from the recorded source page.')
                    total = int(response.headers.get('Content-Length', 0))
                    copied = 0
                    reported = time.monotonic()
                    with partial.open('wb') as output:
                        for chunk in response.iter_content(1024 * 1024):
                            output.write(chunk); copied += len(chunk)
                            if time.monotonic() - reported > 10:
                                print(json.dumps({'dataset': name, 'downloaded': copied, 'expected': total}), flush=True)
                                reported = time.monotonic()
                    if total and copied != total:
                        raise RuntimeError(f'Incomplete archive: {copied}/{total}')
                if not zipfile.is_zipfile(partial):
                    raise RuntimeError('The downloaded file is not a ZIP')
                partial.replace(archive)
                break
            except (requests.RequestException, RuntimeError):
                if attempt == 2: raise
                time.sleep(2)
    destination = folder / 'raw'
    destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        bad = source.testzip()
        if bad: raise ValueError(f'Archive CRC failed: {bad}')
        for entry in source.infolist():
            relative = Path(entry.filename)
            if relative.is_absolute() or relative.drive or '..' in relative.parts:
                raise ValueError('Archive contains a path outside its data directory')
            if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Dataset archives must not contain symbolic links')
            source.extract(entry, destination)
    with archive.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    images = [p for p in destination.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'} and not p.name.startswith('.')]
    record = {'dataset': name, 'source': 'https://www.kaggle.com/datasets/' + SOURCES[name],
              'archive_bytes': archive.stat().st_size, 'archive_sha256': digest,
              'images': len(images), 'crc': 'pass'}
    if not images: raise ValueError('The archive contains no images')
    (folder / 'source.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=[*SOURCES, 'all'], default='all')
    parser.add_argument('--root', type=Path, default=ROOT / 'datasets')
    args = parser.parse_args()
    for name in SOURCES if args.dataset == 'all' else [args.dataset]:
        download(name, args.root.resolve())


if __name__ == '__main__':
    main()
