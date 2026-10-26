"""Portable desktop startup: local resources, writable state and device choice."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import sys


def resource_root() -> Path:
    return Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parents[1]))


def user_data_dir() -> Path:
    override = os.environ.get('EXPRESSION_DEID_DATA_HOME')
    if override:
        folder = Path(override).expanduser()
    elif getattr(sys, 'frozen', False):
        folder = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData/Local')) / 'ExpressionLab'
    else:
        folder = resource_root() / 'outputs' / 'desktop'
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def resolve_device(requested: str = 'auto') -> tuple[str, str]:
    """Exercise a CUDA kernel, including cuDNN, before selecting acceleration."""
    import torch

    torch.set_num_threads(min(8, max(1, (os.cpu_count() or 2) // 2)))
    if requested == 'cpu':
        return 'cpu', 'CPU · 本地计算'
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('当前没有可用的 NVIDIA 加速设备')
        with torch.inference_mode():
            x = torch.ones((1, 1, 8, 8), device='cuda')
            kernel = torch.ones((1, 1, 3, 3), device='cuda')
            torch.nn.functional.conv2d(x, kernel).sum().item()
            torch.cuda.synchronize()
        name = torch.cuda.get_device_name(0)
        logging.info('CUDA ready: %s', name)
        return 'cuda', f'GPU 加速 · {name}'
    except (RuntimeError, OSError, AssertionError) as exc:
        logging.warning('CUDA unavailable; using CPU: %s', exc)
        if requested == 'cuda':
            raise RuntimeError('该电脑当前无法使用 NVIDIA 加速，请使用自动选择或 CPU。') from exc
        return 'cpu', 'CPU · 自动适配'


def verify_assets(*, deep: bool = False) -> dict:
    """The release manifest rejects missing resources and Git LFS pointers."""
    root = resource_root()
    manifest = root / 'package_manifest.json'
    if not manifest.is_file():
        if getattr(sys, 'frozen', False):
            raise FileNotFoundError('软件资源清单缺失，请重新解压完整的软件包。')
        return {}
    data = json.loads(manifest.read_text(encoding='utf-8'))
    for row in data['files']:
        relative = Path(row['path'])
        # Check the logical path before Windows opens it. MSIX can redirect
        # individual files into LocalCache while leaving the parent unchanged;
        # resolve() would then incorrectly reject legitimate bundled assets.
        if relative.is_absolute() or relative.drive or '..' in relative.parts:
            raise ValueError('软件资源清单包含无效路径')
        path = root / relative
        if not path.is_file() or path.stat().st_size != row['bytes']:
            raise FileNotFoundError(f"软件资源不完整：{row['path']}，请重新解压完整的软件包。")
        if deep:
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if digest != row['sha256']:
                raise ValueError(f"软件资源校验失败：{row['path']}")
    return data
