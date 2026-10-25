import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image


class AuditRegressions(unittest.TestCase):
    def test_disabled_source_teacher_rejects_silently_ineffective_weights(self):
        from expression_deid.training.train import train_anonymizer
        for values in ({'consistency_weight': 1.0}, {'teacher_label_weight': .2}):
            with self.assertRaisesRegex(ValueError, '源图表情教师已禁用'):
                train_anonymizer(identity_checkpoint='unused',expression_checkpoint='unused',**values)

    def test_failed_video_warmup_closes_both_loaded_pipelines(self):
        from unittest.mock import Mock
        from expression_deid.runtime.video import run_video
        machine, display = Mock(), Mock()
        machine.predict_expression.side_effect = RuntimeError('warmup failed')
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'input.mp4'
            source.write_bytes(b'fixture')
            with patch('expression_deid.runtime.video._stream_pipelines',return_value=(machine,display)):
                with self.assertRaisesRegex(RuntimeError,'warmup failed'):
                    run_video(source,Path(folder)/'out.mp4')
        machine.close.assert_called_once()
        display.close.assert_called_once()

    def test_ppb_without_video_stream_closes_container(self):
        from unittest.mock import Mock
        from expression_deid.preprocessing.prepare_ppb import align_ppb_neutral
        container=Mock(); container.streams.video=[]
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'source'; source.mkdir()
            (source/'P01-ND.mp4').write_bytes(b'fixture')
            with patch('av.open',return_value=container):
                with self.assertRaisesRegex(ValueError,'没有视频流'):
                    align_ppb_neutral(source,Path(folder)/'output')
        container.close.assert_called_once()

    def test_tensor_input_helper(self):
        from expression_deid.models import tensor_from_input
        self.assertEqual(tuple(tensor_from_input(np.zeros((100, 100))).shape), (1, 1, 100, 100))
        with self.assertRaises(ValueError):
            tensor_from_input(np.zeros((1, 3, 100, 100)))

    def test_mixed_missing_evaluation_labels(self):
        from expression_deid.training.evaluate import summarize
        row = dict(true_expression=6, anonymized_expression=6, expression_anonymized_correct=True,
                   expression_original_correct=True, expression_consistent=None)
        result = summarize([row, {**row, 'true_expression': None}])
        self.assertNotIn('expression_consistency', result)
        self.assertEqual(result['expression_anonymized_by_class']['6']['samples'], 1)

    def test_invalid_normalization_is_rejected(self):
        from expression_deid.preprocessing.normalize import load_stats
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'stats.json'
            for std in (0, -1, float('nan')):
                path.write_text(json.dumps(dict(channels=1, space='gray_0_1', mean=.5, std=std)))
                with self.assertRaises(ValueError):
                    load_stats(path)

    def test_experiment_failure_reenables_button(self):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6.QtWidgets import QApplication
        from expression_deid.ui.desktop import ExperimentPage
        app = QApplication.instance() or QApplication([])
        with patch('expression_deid.ui.desktop.samples', return_value=[]):
            page = ExperimentPage('cpu')
        page.run_button.setEnabled(False)
        page._error('missing weights')
        self.assertTrue(page.run_button.isEnabled())
        page.close()

    def test_face_only_training_crop_preserves_gradient(self):
        import torch
        from expression_deid.training.train import _identity_face_only_differentiable, _identity_face_only_batch
        torch.manual_seed(8)
        source = torch.rand(2, 1, 100, 100, requires_grad=True)
        result = _identity_face_only_differentiable(source)
        expected = _identity_face_only_batch(source)
        self.assertLess(float((result-expected).abs().mean().detach()), 2/255)
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(source.grad).all())
        self.assertGreater(float(source.grad.abs().sum()), 0)
        self.assertEqual(float(source.grad[:, :, :12].abs().sum()), 0)

    def test_degenerate_alignment_rejected(self):
        from expression_deid.preprocessing.align import similarity_matrix
        for points in (np.zeros((5,2)), np.full((5,2), np.nan)):
            with self.assertRaises(ValueError): similarity_matrix(points, np.ones((5,2)))

    def test_ppb_fractional_video_metadata(self):
        from types import SimpleNamespace
        from expression_deid.preprocessing.prepare_ppb import _frame_count
        stream = SimpleNamespace(frames=0, duration=4, time_base=.1, average_rate=29.97)
        self.assertEqual(_frame_count(SimpleNamespace(duration=None), stream), 12)
        stream.duration = None
        self.assertEqual(_frame_count(SimpleNamespace(duration=400000), stream), 12)

    def test_local_edit_resource_failure_is_not_original(self):
        from expression_deid.runtime.pipeline import ExpressionDeidentifier, AnonymizationUnavailable
        pipeline = ExpressionDeidentifier.__new__(ExpressionDeidentifier)
        pipeline.component_bank = []
        with self.assertRaises(AnonymizationUnavailable):
            pipeline._component_anonymize(np.full((100,100), 90, dtype=np.uint8))

    def test_donor_search_is_independent_of_pixel_units(self):
        from expression_deid.runtime.pipeline import ExpressionDeidentifier
        pipeline = ExpressionDeidentifier.__new__(ExpressionDeidentifier)
        rng = np.random.default_rng(0)
        pipeline.component_bank = [(i, rng.integers(0,256,(100,100)).astype(np.float32)/255) for i in range(4)]
        pipeline.component_people = np.arange(4)
        pipeline.component_vectors = np.stack([pipeline._component_descriptor(image) for _, image in pipeline.component_bank])
        source = rng.integers(0,256,(100,100),dtype=np.uint8)
        np.testing.assert_array_equal(pipeline._select_component_donor(source), pipeline._select_component_donor(source.astype(np.float32)/255))

    def test_desktop_stop_retains_loading_worker_and_switches_latest_source(self):
        import threading
        import time
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6.QtCore import QThread, Signal
        from PySide6.QtWidgets import QApplication
        from expression_deid.ui.desktop import VideoPage
        app = QApplication.instance() or QApplication([])
        class SlowWorker(QThread):
            frame_ready = Signal(object, object, str, float, float, int)
            worker_error = Signal(str)
            ended = Signal()
            starts = []
            def __init__(self, source, device):
                super().__init__(); self.source=source; self.frame_pending=threading.Event()
            def stop(self): pass  # mimic a noninterruptible model load
            def run(self):
                self.starts.append(self.source)
                time.sleep(.15)
                self.worker_error.emit('intentional test failure')
        with patch('expression_deid.ui.desktop.VideoWorker', SlowWorker):
            page = VideoPage('cpu')
            page._start(('video','first'), 'first')
            original_worker = page.worker
            page._start(('video','second'), 'second')
            page._start(('video','third'), 'third')
            self.assertIs(page.worker, original_worker)
            deadline = time.monotonic()+3
            while page.worker is not None and time.monotonic() < deadline:
                app.processEvents(); time.sleep(.01)
            self.assertIsNone(page.worker)
            self.assertEqual(SlowWorker.starts, [('video','first'),('video','third')])
            self.assertIn('intentional test failure', page.status.text())
            page.close()

    def test_training_augmentation_obeys_numpy_seed(self):
        from expression_deid.preprocessing.normalize import expression_input, NormStats
        image = np.arange(10000, dtype=np.uint8).reshape(100,100)
        np.random.seed(5)
        first = expression_input(image, NormStats(.5,.2), augment=True, component_augment=True)
        np.random.seed(5)
        second = expression_input(image, NormStats(.5,.2), augment=True, component_augment=True)
        np.testing.assert_array_equal(first, second)

    def test_ferplus_rejects_unknown_majority_and_ties(self):
        import csv
        from expression_deid.training.data import FERPlusExpressionDataset
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with (root/'fer2013.csv').open('w', newline='') as handle:
                writer=csv.DictWriter(handle, fieldnames=['Usage','pixels']); writer.writeheader()
                writer.writerows([dict(Usage='Training', pixels=' '.join(['90']*2304))]*3)
            fields=['Usage','neutral','happiness','unknown','NF']
            with (root/'fer2013new.csv').open('w', newline='') as handle:
                writer=csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
                writer.writerows([dict(Usage='Training',neutral=5,happiness=0,unknown=8,NF=0),
                                  dict(Usage='Training',neutral=5,happiness=5,unknown=0,NF=0),
                                  dict(Usage='Training',neutral=6,happiness=2,unknown=2,NF=0)])
            dataset = FERPlusExpressionDataset('train', root=root)
            self.assertEqual(len(dataset),1)
            self.assertEqual(dataset.records[0]['label'],6)

    def test_enhanced_fer_label_order(self):
        import csv
        from expression_deid.training.data import FEREnhancedExpressionDataset
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'ferplus_enhanced_train.csv'
            with path.open('w',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=['emotion','pixels','quality_score'])
                writer.writeheader()
                writer.writerows(dict(emotion=i,pixels=' '.join(['90']*2304),quality_score=.9) for i in range(7))
            dataset=FEREnhancedExpressionDataset(root=Path(folder))
            self.assertEqual([r['label'] for r in dataset.records],[5,2,1,3,4,0,6])

    def test_fer_validation_pixels_are_excluded_from_training(self):
        import csv
        from expression_deid.training.data import FERPlusExpressionDataset
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with (root/'fer2013.csv').open('w',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=['Usage','pixels']);writer.writeheader()
                writer.writerows([dict(Usage='Training',pixels=' '.join(['90']*2304)),
                                  dict(Usage='PublicTest',pixels=' '.join(['90']*2304)),
                                  dict(Usage='Training',pixels=' '.join(['91']*2304))])
            with (root/'fer2013new.csv').open('w',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=['Usage','neutral']);writer.writeheader()
                writer.writerows(dict(Usage=usage,neutral=10) for usage in ['Training','PublicTest','Training'])
            dataset=FERPlusExpressionDataset(root=root)
            self.assertEqual(len(dataset),1)
            self.assertEqual(dataset.records[0]['source'],'fer0000002.png')

    def test_explicit_video_checkpoint_is_never_silently_replaced(self):
        from expression_deid.runtime.video import _stream_pipelines, run_video
        with patch('expression_deid.ui.opencv_demo.build_pipeline') as loader:
            with self.assertRaises(FileNotFoundError):
                _stream_pipelines('cpu','Z:/missing-expression.pt')
            loader.assert_not_called()
        with self.assertRaises(FileNotFoundError): run_video('Z:/absent.mp4','out.mp4')

    def test_invalid_checkpoint_is_rejected(self):
        import torch
        from expression_deid.models import load_checkpoint
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'broken.pt'
            for payload in ({}, {'kind':'expression','state_dict':{'bad':torch.tensor(float('nan'))}}):
                torch.save(payload,path)
                with self.assertRaises(ValueError): load_checkpoint(path)


class RuntimePixelBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        from expression_deid.ui.opencv_demo import build_pipeline
        from expression_deid.config import KMU_ALIGNED_DIR
        cls.pipeline=build_pipeline('cpu',use_gan=False)
        cls.gray=np.asarray(Image.open(next(KMU_ALIGNED_DIR.glob('*.png'))).convert('L'))

    @classmethod
    def tearDownClass(cls): cls.pipeline.close()

    def test_native_aligned_size_and_classifier_pixels(self):
        from expression_deid.preprocessing.align import template_for,warp_gray
        for size in (100,256):
            source=np.asarray(Image.fromarray(self.gray).resize((size,size)))
            with patch.object(self.pipeline,'predict_expression',wraps=self.pipeline.predict_expression) as spy:
                result=self.pipeline.process(source,aligned=True,diagnostics=False)
            self.assertEqual(spy.call_count,1)
            self.assertEqual(result['anonymized'].shape,source.shape)
            machine=self.pipeline._last_anonymized_aligned
            np.testing.assert_array_equal(spy.call_args.args[0],machine)
            np.testing.assert_array_equal(result['anonymized'][:8,:8],source[:8,:8])
            self.assertFalse(np.array_equal(machine,self.pipeline._resize(source)))
            self.assertTrue(result['background_preserved'])

    def test_raw_square_explicitly_detected_background_exact(self):
        from expression_deid.preprocessing.align import template_for
        class Detector:
            calls=0
            def detect(detector,rgb):
                detector.calls+=1
                size=rgb.shape[0]
                row=np.zeros(15); row[:4]=(0,0,size,size); row[4:14]=template_for(size).ravel(); row[14]=1
                return row[None]
        detector=Detector()
        with patch.object(self.pipeline,'detector',detector):
            result=self.pipeline.process(self.gray,aligned=False,diagnostics=False)
        self.assertEqual(detector.calls,1)
        np.testing.assert_array_equal(result['anonymized'][:8,:8],self.gray[:8,:8])

    def test_diagnostics_do_not_change_anonymous_expression_input(self):
        fake=dict(label=0,confidence=.8,embedding=[1.,0.],probabilities=[1.,0.])
        with patch.object(self.pipeline,'identity_model',object()),patch.object(self.pipeline,'predict_identity',return_value=fake):
            self.pipeline.process(self.gray,aligned=True,diagnostics=False)
            first=self.pipeline._last_anonymized_aligned.copy()
            with patch.object(self.pipeline,'predict_expression',wraps=self.pipeline.predict_expression) as spy:
                self.pipeline.process(self.gray,aligned=True,diagnostics=True)
            self.assertEqual(spy.call_count,1)
            np.testing.assert_array_equal(first,self.pipeline._last_anonymized_aligned)

    def test_validation_sources_are_excluded_from_local_donors(self):
        from expression_deid.config import KMU_ALIGNED_DIR
        from expression_deid.preprocessing.prepare import iter_kmu_aligned
        from expression_deid.training.data import split_kmu_sequence_tails
        items=list(iter_kmu_aligned(KMU_ALIGNED_DIR,include_review=False))
        train,valid=split_kmu_sequence_tails(items)
        pool={p.resolve() for p in self.pipeline.component_source_paths}
        self.assertEqual((len(train),len(valid)),(880,220))
        self.assertFalse(pool & {items[i][0].resolve() for i in valid})
        self.assertTrue({items[i][0].resolve() for i in train} <= pool)


if __name__ == '__main__':
    unittest.main()
