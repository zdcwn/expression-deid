"""The desktop must expose every image, including late-class samples."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from expression_deid.ui import desktop


class DesktopGalleryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        rows = []
        for label in [0, 3]:
            for index in range(550):
                filename = f'{label}_{index:02d}.png'
                Image.new('L', (32, 32), 120).save(self.folder / filename)
                rows.append(dict(status='ok', label=label, output=filename, person='01'))
        rows += [dict(status='ok',label=0,output='../outside.png'),
                 dict(status='failed',label=0,output='0_00.png'),
                 dict(status='ok',label=99,output='0_00.png')]
        (self.folder / 'manifest.jsonl').write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
        self.paths = patch.object(desktop, 'KMU_ALIGNED_DIR', self.folder)
        self.paths.start()
        self.page = desktop.ExperimentPage('cpu')
        self.page.resize(1100, 760); self.page.show(); self.app.processEvents()

    def tearDown(self):
        self.page.thumbnail_timer.stop(); self.page.close(); self.page.deleteLater()
        self.app.processEvents(); self.paths.stop(); self.temporary.cleanup()

    def test_gallery_has_no_four_per_class_cap(self):
        self.assertEqual(len(desktop.samples()), 1100)
        self.assertEqual(self.page.sample_list.count(), 1100)
        self.assertLess(len(self.page.loaded_thumbnails), 1100)

    def test_per_file_windows_redirection_does_not_hide_gallery(self):
        def redirected(path, *args, **kwargs):
            return Path('C:/MSIX/redirected') / path.name if path.suffix == '.png' else path
        with patch.object(Path, 'resolve', redirected):
            self.assertEqual(len(desktop.samples()), 1100)

    def test_filter_and_last_sample_remain_accessible(self):
        self.page.expression_filter.setCurrentIndex(self.page.expression_filter.findData(3))
        self.app.processEvents()
        visible = [self.page.sample_list.item(i) for i in range(1100) if not self.page.sample_list.item(i).isHidden()]
        self.assertEqual(len(visible), 550)
        self.assertTrue(all(row.data(Qt.UserRole)['label'] == 3 for row in visible))
        last = self.page.sample_list.item(1099)
        self.page.sample_list.setCurrentItem(last); self.page.sample_list.scrollToItem(last)
        self.app.processEvents(); self.page._load_visible_thumbnails()
        self.assertFalse(last.icon().isNull())
        self.assertEqual(last.data(Qt.UserRole)['path'].name, '3_549.png')
        self.page.expression_filter.setCurrentIndex(0); self.app.processEvents()
        self.assertEqual(len(self.page.filtered_samples()), 1100)
        self.page.sample_list.setCurrentItem(last); self.page.sample_list.scrollToItem(last)
        self.app.processEvents(); self.page._load_visible_thumbnails()
        self.assertTrue(self.page.sample_list.visualItemRect(last).intersects(self.page.sample_list.viewport().rect()))


if __name__ == '__main__':
    unittest.main()
