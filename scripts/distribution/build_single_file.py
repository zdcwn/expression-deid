"""Wrap the verified portable ZIP in a local, one-time-extracting Windows launcher."""
from pathlib import Path
import argparse
import hashlib
import os
import shutil
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/distribution')
    args = parser.parse_args()
    out = args.output.resolve()
    archive = out / 'ExpressionLab-Windows-x64.zip'
    if not archive.is_file():
        raise FileNotFoundError('Build the verified portable ZIP first')
    compiler = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
    stub = out / 'build' / 'PortableLauncher.exe'
    stub.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(compiler), '/nologo', '/target:winexe', '/platform:x64',
                    '/out:' + str(stub), '/win32manifest:' + str(ROOT / 'scripts/distribution/launcher.manifest'),
                    '/win32icon:' + str(out / 'staging/expression_deid/ui/assets/app.ico'),
                    '/reference:System.Windows.Forms.dll', '/reference:System.Drawing.dll',
                    '/reference:System.IO.Compression.dll', str(ROOT / 'scripts/distribution/PortableLauncher.cs')], check=True)
    with archive.open('rb') as handle:
        archive_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
    destination = out / 'ExpressionLab-Windows-x64.exe'
    with destination.open('wb') as output:
        with stub.open('rb') as handle:
            shutil.copyfileobj(handle, output)
        start = output.tell()
        with archive.open('rb') as handle:
            shutil.copyfileobj(handle, output)
        output.write(struct.pack('<qq', start, archive.stat().st_size))
        output.write(archive_hash.encode('ascii'))
        output.write(b'EXPRESSIONLABPKG1')
    with destination.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    with (out / 'SHA256SUMS.txt').open('a', encoding='ascii') as handle:
        handle.write(f'{digest}  {destination.name}\n')
    print(destination, destination.stat().st_size)


if __name__ == '__main__':
    main()
