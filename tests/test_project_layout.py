"""Check the resources shared by the reorganized desktop and training entries."""
import os
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


class ProjectLayoutTests(unittest.TestCase):
    def test_desktop_examples_resolve_outside_generated_outputs(self):
        from PySide6.QtWidgets import QApplication
        from expression_deid.config import EXAMPLE_VIDEO_DIR
        from expression_deid.ui.desktop import VideoPage
        app = QApplication.instance() or QApplication([])
        page = VideoPage('cpu')
        try:
            self.assertEqual(page.examples.count(), 4)
            for index in range(page.examples.count()):
                page.examples.setCurrentIndex(index)
                with patch.object(page, '_start') as start:
                    page.play_example()
                source = Path(start.call_args.args[0][1])
                self.assertTrue(source.is_file())
                self.assertEqual(source.parent, EXAMPLE_VIDEO_DIR)
        finally:
            page.close()
            page.deleteLater()
            app.processEvents()

    def test_candidate_training_outputs_cannot_overwrite_runtime_weights(self):
        from expression_deid.cli import build_parser
        parser = build_parser()
        for command in ['train-expression', 'train-identity', 'train-expression-kmu', 'train-expression-joint']:
            args = parser.parse_args([command])
            self.assertEqual(Path(args.output).parent, Path('models/training'))

    def test_quality_contract_keeps_its_weight_bundle_together(self):
        import hashlib
        import json
        from expression_deid.config import ROOT
        manifest = ROOT/'models/expression/expression_video_quality.json'
        spec = json.loads(manifest.read_text(encoding='utf8'))
        for key in ['compact_model', 'quality_model']:
            target = manifest.parent/spec[key]['file']
            self.assertTrue(target.is_file())
            with target.open('rb') as stream:
                self.assertEqual(hashlib.file_digest(stream, 'sha256').hexdigest(), spec[key]['sha256'])
