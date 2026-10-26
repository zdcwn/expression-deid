"""Split a verified executable into small release assets without changing it."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chunk-mib', type=int, default=900)
    args = parser.parse_args()
    if not 1 <= args.chunk_mib <= 1024:
        raise ValueError('Choose a chunk size between 1 and 1024 MiB')
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    parts, original_hash = [], hashlib.sha256()
    chunk_size = args.chunk_mib * 1024**2
    with source.open('rb') as stream:
        index = 1
        while stream.tell() < source.stat().st_size:
            target = output / f'ExpressionLab-GPU.exe.part{index}'
            part_hash, size = hashlib.sha256(), 0
            with target.open('wb') as part:
                while size < chunk_size:
                    block = stream.read(min(8 * 1024**2, chunk_size - size))
                    if not block:
                        break
                    part.write(block)
                    part_hash.update(block)
                    original_hash.update(block)
                    size += len(block)
            parts.append({'name': target.name, 'bytes': size, 'sha256': part_hash.hexdigest()})
            index += 1
    digest = original_hash.hexdigest()
    # Re-read every part in order: the release must reconstruct the exact file.
    reconstructed = hashlib.sha256()
    for part in parts:
        with (output / part['name']).open('rb') as stream:
            while block := stream.read(8 * 1024**2):
                reconstructed.update(block)
    if reconstructed.hexdigest() != digest:
        raise ValueError('Split executable failed byte-for-byte reconstruction')
    build = output / 'build'
    build.mkdir(exist_ok=True)
    names = ','.join(json.dumps(part['name']) for part in parts)
    sizes = ','.join(str(part['bytes']) + 'L' for part in parts)
    config = build / 'PackageInfo.cs'
    config.write_text(
        f'internal static class PackageInfo {{ public const long Size={source.stat().st_size}L; '
        f'public const string Hash="{digest}"; '
        f'public static readonly string[] Parts=new string[]{{{names}}}; '
        f'public static readonly long[] PartSizes=new long[]{{{sizes}}}; }}', encoding='utf-8')
    launcher = output / 'ExpressionLab-GPU-Start.exe'
    compiler = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
    subprocess.run([str(compiler), '/nologo', '/target:winexe', '/platform:x64',
                    '/out:' + str(launcher), '/win32manifest:' + str(ROOT / 'scripts/distribution/launcher.manifest'),
                    '/reference:System.Windows.Forms.dll', '/reference:System.Drawing.dll',
                    str(ROOT / 'scripts/distribution/ReassembleLauncher.cs'), str(config)], check=True)
    with launcher.open('rb') as stream:
        launcher_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    record = {'original_bytes': source.stat().st_size, 'original_sha256': digest,
              'parts': parts, 'launcher': {'name': launcher.name, 'bytes': launcher.stat().st_size,
                                         'sha256': launcher_hash}, 'reconstruction': 'identical'}
    (output / 'parts-check.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
