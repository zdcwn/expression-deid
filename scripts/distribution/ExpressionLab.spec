# -*- mode: python ; coding: utf-8 -*-
"""Windows x64 offline distribution. Build with the project inference environment."""
import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, copy_metadata

root = Path(SPECPATH).resolve().parents[1]
stage = Path(os.environ['EXPRESSION_RELEASE_STAGE'])
datas = [(str(stage), '.')]
datas += collect_data_files('mediapipe')
datas += collect_data_files('av')
for package in ['torch', 'mediapipe', 'PySide6', 'opencv-contrib-python', 'onnxruntime', 'av', 'numpy', 'pillow', 'scipy', 'matplotlib']:
    datas += copy_metadata(package)
binaries = collect_dynamic_libs('mediapipe')
binaries += collect_dynamic_libs('onnxruntime')

a = Analysis(
    [str(root / 'scripts/distribution/desktop_entry.py')],
    pathex=[str(root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=['expression_deid.distribution_check', 'PySide6.QtSvg',
                   'scipy.spatial.transform', 'scipy.spatial._qhull', 'av'],
    hooksconfig={'matplotlib': {'backends': ['Agg']}},
    excludes=['PyQt5', 'PyQt6', 'PySide2', 'tkinter', 'IPython', 'notebook',
              'jupyter', 'tensorflow', 'jax', 'jaxlib', 'transformers',
              'torchvision', 'torchaudio', 'pytorch_lightning', 'pandas', 'pytest'],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True, name='ExpressionLab',
    debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
    console=False, disable_windowed_traceback=False,
    icon=str(stage / 'expression_deid/ui/assets/app.ico'),
    version=str(root / 'scripts/distribution/version_info.txt'),
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='ExpressionLab')
