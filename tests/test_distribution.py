"""Release assets remain checked under Windows per-file path redirection."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from expression_deid.distribution import verify_assets


class DistributionAssetTests(unittest.TestCase):
    def test_redirected_asset_keeps_logical_containment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'models').mkdir()
            (root / 'models/model.pt').write_bytes(b'original asset bytes')
            manifest = {'files': [{'path': 'models/model.pt', 'bytes': 20,
                                  'sha256': hashlib.sha256(b'original asset bytes').hexdigest()}]}
            (root / 'package_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            # MSIX reports a different physical path for the file and parent.
            def redirected(path, *args, **kwargs):
                return Path('C:/MSIX/redirected/model.pt') if path.name == 'model.pt' else path
            with patch('expression_deid.distribution.resource_root', return_value=root), patch.object(Path, 'resolve', redirected):
                self.assertEqual(verify_assets(deep=True), manifest)

    def test_traversal_and_absolute_manifest_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in ['../outside', 'models/../../outside', str(root / 'absolute')]:
                with self.subTest(relative=relative):
                    (root / 'package_manifest.json').write_text(json.dumps({'files': [{'path': relative}]}), encoding='utf-8')
                    with patch('expression_deid.distribution.resource_root', return_value=root):
                        with self.assertRaisesRegex(ValueError, '无效路径'):
                            verify_assets(deep=True)

    def test_incorrect_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'model.pt').write_bytes(b'wrong')
            (root / 'package_manifest.json').write_text(json.dumps({'files': [{'path': 'model.pt', 'bytes': 5, 'sha256': '0' * 64}]}), encoding='utf-8')
            with patch('expression_deid.distribution.resource_root', return_value=root):
                with self.assertRaisesRegex(ValueError, '校验失败'):
                    verify_assets(deep=True)
