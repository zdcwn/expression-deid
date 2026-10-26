"""Build a self-contained Windows desktop folder and a transferable ZIP."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def export_generator(source: Path, target: Path) -> dict:
    """Drop optimizer/discriminator state; preserve every inference tensor bit."""
    import torch
    payload = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
    state = payload.get('state_dict', payload)
    generator = {key: value for key, value in state.items() if key.startswith('generator.')}
    if not generator or any(not isinstance(value, torch.Tensor) for value in generator.values()):
        raise ValueError('Checkpoint has no valid generator state')
    torch.save({'state_dict': generator}, target)
    exported = torch.load(target, map_location='cpu', weights_only=True, mmap=True)['state_dict']
    if set(exported) != set(generator) or any(
            exported[key].dtype != value.dtype or not torch.equal(exported[key], value)
            for key, value in generator.items()):
        raise ValueError('Inference export changed a generator tensor')
    return {'inference_only': True, 'source_bytes': source.stat().st_size,
            'generator_tensors': len(generator), 'tensors_identical': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/distribution')
    parser.add_argument('--skip-build', action='store_true')
    parser.add_argument('--skip-zip', action='store_true')
    args = parser.parse_args()
    if sys.platform != 'win32' or sys.maxsize <= 2**32:
        raise SystemExit('Build this release using 64-bit Windows Python.')
    output = args.output.resolve()
    stage = output / 'staging'
    stage.mkdir(parents=True, exist_ok=True)
    resources = [
        'models/detection/face_detection_yunet_2023mar.onnx',
        'models/detection/face_landmarker.task',
        'models/anonymization/anonymizer_adversarial_v1.pt',
        'models/anonymization/ganonymization_50.ckpt',
        'models/expression/expression_realtime_stable.pt',
        'models/expression/expression_machine_selector_texture.pt',
        'models/expression/expression_video_quality.ts.pt',
        'models/expression/expression_video_quality.json',
        'expression_deid/preprocessing/norm_stats.json',
        'docs/model-metrics.json',
        'third_party/ganonymization/lib/models/pix2pix.py',
        'third_party/ganonymization/LICENSE',
    ]
    for folder in ['datasets/kmu-fed-aligned', 'examples/videos', 'third_party/licenses', 'expression_deid/ui/assets']:
        resources.extend(p.relative_to(ROOT).as_posix() for p in (ROOT / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files = []
    for relative in sorted(set(resources)):
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f'Missing release resource: {relative}')
        if source.suffix in {'.pt', '.ckpt', '.onnx', '.task'} and source.stat().st_size < 1000:
            raise ValueError(f'Replace Git LFS pointer with actual asset: {relative}')
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        export_info = {}
        if relative == 'models/anonymization/ganonymization_50.ckpt':
            export_info = export_generator(source, target)
        else:
            shutil.copy2(source, target)
        with target.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        files.append({'path': relative, 'bytes': target.stat().st_size, 'sha256': digest, **export_info})
    from PIL import Image, ImageDraw
    icon = Image.new('RGBA', (256, 256), '#173c33')
    pen = ImageDraw.Draw(icon)
    pen.rounded_rectangle((65, 35, 191, 222), radius=57, outline='#b9e6ca', width=10)
    pen.line((95, 112, 110, 112), fill='#b9e6ca', width=9)
    pen.line((146, 112, 161, 112), fill='#b9e6ca', width=9)
    pen.arc((93, 135, 163, 184), 10, 170, fill='#b9e6ca', width=8)
    icon.save(stage / 'expression_deid/ui/assets/app.ico', sizes=[(32,32),(48,48),(64,64),(128,128),(256,256)])
    manifest = {'schema': 1, 'platform': 'windows-x64', 'source_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                'source_dirty': bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),
                'files': files, 'versions': {name: importlib.metadata.version(name) for name in ['torch','PySide6','mediapipe','PyInstaller']}}
    (stage / 'package_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    dist = output / 'dist'
    if not args.skip_build:
        env = os.environ.copy()
        env['EXPRESSION_RELEASE_STAGE'] = str(stage)
        env['PYTHONIOENCODING'] = 'utf-8'
        subprocess.run([sys.executable, '-m', 'PyInstaller', '--noconfirm', '--distpath', str(dist),
                        '--workpath', str(output / 'build'), str(ROOT / 'scripts/distribution/ExpressionLab.spec')],
                       cwd=ROOT, env=env, check=True)
    folder = dist / 'ExpressionLab'
    if not (folder / 'ExpressionLab.exe').is_file():
        raise FileNotFoundError('Executable was not built')
    if args.skip_build:
        # Repack an existing frozen runtime when only release assets change.
        # Python code changes still require the normal freeze/build step.
        for row in files:
            target = folder / '_internal' / row['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(stage / row['path'], target)
        shutil.copy2(stage / 'package_manifest.json', folder / '_internal/package_manifest.json')
    shutil.copy2(ROOT / 'docs/distribution/使用说明.txt', folder / '使用说明.txt')
    if not args.skip_zip:
        archive = output / 'ExpressionLab-Windows-x64.zip'
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as zip:
            for path in sorted(folder.rglob('*')):
                if path.is_file():
                    zip.write(path, Path('ExpressionLab') / path.relative_to(folder))
        with archive.open('rb') as stream:
            checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
        (output / 'SHA256SUMS.txt').write_text(f'{checksum}  {archive.name}\n', encoding='ascii')
        print(archive, archive.stat().st_size, flush=True)
    print('RELEASE_FOLDER=' + str(folder), flush=True)


if __name__ == '__main__':
    main()
