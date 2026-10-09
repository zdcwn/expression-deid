import unittest

import numpy as np

from expression_deid.temporal import TemporalExpressionFilter


def scores(label, confidence=0.90):
    values = np.full(7, (1.0 - confidence) / 6)
    values[label] = confidence
    return values


class TemporalExpressionTest(unittest.TestCase):
    def test_isolated_high_confidence_errors_do_not_change_expression(self):
        filt = TemporalExpressionFilter()
        reported = []
        for i in range(90):
            prediction = scores(i % 6) if i in (18, 39, 64) else scores(6)
            result = filt.update(prediction, i / 30)
            reported.append(result['label'])
        self.assertEqual(set(reported[4:]), {6})

    def test_real_change_is_reported_at_different_capture_rates(self):
        for fps in (15, 24, 60):
            with self.subTest(fps=fps):
                filt = TemporalExpressionFilter()
                switched = None
                for i in range(2 * fps):
                    timestamp = i / fps
                    result = filt.update(scores(6 if timestamp < 1 else 3), timestamp)
                    if timestamp >= 1 and result['label'] == 3 and switched is None:
                        switched = timestamp
                self.assertIsNotNone(switched)
                self.assertLessEqual(switched - 1, 0.6)
                self.assertGreaterEqual(switched - 1, 0.2)

    def test_uncertainty_is_not_forced_into_neutral_or_kept_forever(self):
        filt = TemporalExpressionFilter()
        for i in range(15):
            filt.update(scores(5), i / 30)
        for i in range(15, 60):
            result = filt.update(np.full(7, 1 / 7), i / 30)
        self.assertIsNone(result['label'])
        self.assertIsNone(result['confidence'])
        self.assertEqual(result['state'], 'uncertain')

    def test_startup_and_long_pause_need_fresh_evidence(self):
        filt = TemporalExpressionFilter()
        for i in range(10):
            result = filt.update(scores(6), i / 30)
        self.assertEqual(result['label'], 6)
        result = filt.update(scores(3), 3.0)
        self.assertIsNone(result['label'])
        result = filt.update(scores(3), 2.0)  # seek backwards
        self.assertIsNone(result['label'])
        filt.reset()
        self.assertIsNone(filt.update(scores(5), 0)['label'])

    def test_confidence_is_probability_of_reported_class(self):
        filt = TemporalExpressionFilter()
        for i in range(10):
            filt.update(scores(6, 0.60), i / 30)
        result = filt.update(scores(3), 10 / 30)
        self.assertEqual(result['label'], 6)
        self.assertAlmostEqual(result['confidence'], result['probabilities'][6])
        self.assertLess(result['confidence'], 0.60)
        self.assertAlmostEqual(sum(result['probabilities']), 1)

    def test_invalid_observation_is_rejected(self):
        filt = TemporalExpressionFilter()
        for values in (np.full(7, np.nan), [-1, 2], [0, 0], [[1, 0]]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                filt.update(values, 0)


class VideoTrackTest(unittest.TestCase):
    def test_roi_composite_matches_full_frame_and_keeps_background(self):
        from PIL import Image
        from expression_deid.inference import ExpressionDeidentifier
        pipeline = ExpressionDeidentifier.__new__(ExpressionDeidentifier)
        rng = np.random.default_rng(17)
        gray = rng.integers(0, 256, (240, 320), dtype=np.uint8)
        edited = rng.integers(0, 256, (100, 100), dtype=np.uint8)
        matrix = np.array([[1.2, .1, -55], [-.1, 1.2, -18]])
        affine = tuple(matrix.ravel())
        for box in ((20, 10, 160, 190), (-25, -30, 160, 190), (280, 200, 100, 90)):
            for mode in ('features', 'face', 'conceal'):
                with self.subTest(box=box, mode=mode):
                    full = np.asarray(Image.fromarray(edited).resize((256, 256), Image.Resampling.BILINEAR)
                                      .transform((320, 240), Image.Transform.AFFINE, affine,
                                                 resample=Image.Resampling.BILINEAR), dtype=np.float32)
                    alpha = np.asarray(pipeline._composite_mask(mode)
                                       .transform((320, 240), Image.Transform.AFFINE, affine,
                                                  resample=Image.Resampling.BILINEAR), dtype=np.float32) / 255
                    x, y, w, h = box
                    rect = np.zeros_like(gray, dtype=bool)
                    rect[max(0, y):min(240, y + h), max(0, x):min(320, x + w)] = True
                    alpha *= rect
                    expected = np.clip(np.rint(gray * (1 - alpha) + full * alpha), 0, 255).astype(np.uint8)
                    actual = pipeline._composite_component_edit(gray, edited, box, matrix, mask_mode=mode)
                    self.assertLessEqual(np.abs(actual.astype(int) - expected.astype(int)).max(), 1)
                    np.testing.assert_array_equal(actual[~rect], gray[~rect])

    def test_track_locks_donor_and_classifier_receives_only_edited_pixels(self):
        from expression_deid.align import template_for
        from expression_deid.video_demo import FastVideoRunner

        face = np.zeros(15)
        face[:4] = (20, 20, 120, 120)
        face[4:14] = (template_for(100) * 1.2 + 20).ravel()
        face[14] = 0.99

        class Detector:
            visible = True

            def detect(self, rgb):
                return np.array([face]) if self.visible else np.empty((0, 15))

        class Pipeline:
            detector = Detector()
            ganonymization_backend = None
            donor_calls = 0
            classified = []

            @staticmethod
            def _image_gray(rgb):
                return rgb[:, :, 0].copy()

            @staticmethod
            def _resize(gray):
                from PIL import Image
                return np.asarray(Image.fromarray(gray).resize((100, 100)))

            def _select_component_donor(self, source):
                self.donor_calls += 1
                return np.full((100, 100), self.donor_calls)

            @staticmethod
            def _machine_runtime_candidate(aligned, variant):
                return np.full((100, 100), 200, dtype=np.uint8)

            @staticmethod
            def _human_facing_candidate(aligned):
                return np.full((100, 100), 170, dtype=np.uint8)

            @staticmethod
            def _composite_component_edit(gray, edited, face, matrix, mask_mode):
                output = gray.copy()
                output[20:140, 20:140] = edited[0, 0]
                return output

            def predict_expression(self, image, aligned):
                self.classified.append(image.copy())
                return dict(name='neutral', name_zh='中性', confidence=.9, probabilities=scores(6).tolist())

        pipeline = Pipeline()
        runner = FastVideoRunner(pipeline, detect_interval=5)
        rgb = np.full((160, 160, 3), 80, dtype=np.uint8)
        for i in range(20):
            shown, info = runner.process_frame(rgb, i, timestamp=i / 30)
        self.assertEqual(pipeline.donor_calls, 1)
        self.assertEqual(len(pipeline.classified), 20)  # every frame still analyzed
        self.assertTrue(all(image.mean() > 150 for image in pipeline.classified))
        self.assertTrue(all(not np.array_equal(image, np.full((100, 100), 170)) for image in pipeline.classified))
        np.testing.assert_array_equal(shown[:20], rgb[:20, :, 0])
        self.assertEqual(info['expression_input_source'], 'anonymized_only')
        self.assertEqual(info['expression'], 'neutral')
        pipeline.detector.visible = False
        hidden, info = runner.process_frame(rgb, 20, timestamp=20 / 30)
        self.assertFalse(np.any(hidden))
        self.assertIsNone(info['expression'])
        self.assertIsNone(runner.expression_filter.label)
        pipeline.detector.visible = True
        _, info = runner.process_frame(rgb, 21, timestamp=21 / 30)
        self.assertEqual(pipeline.donor_calls, 2)
        self.assertEqual(info['expression'], 'uncertain')
        runner.process_frame(rgb, 0, timestamp=0)  # source restarted
        self.assertEqual(pipeline.donor_calls, 3)

        from unittest.mock import patch
        from expression_deid.inference import AnonymizationUnavailable
        previous_calls = len(pipeline.classified)
        with patch.object(pipeline, '_machine_runtime_candidate', side_effect=AnonymizationUnavailable('no resources')):
            hidden, info = runner.process_frame(rgb, 1, timestamp=1/30)
        self.assertEqual(hidden[80,80], 0)
        self.assertEqual(info['status'], 'anonymization_unavailable')
        self.assertEqual(len(pipeline.classified), previous_calls)

        from expression_deid.ganonymization_backend import FaceLandmarksUnavailable

        class Display:
            ganonymization_backend = object()
            fail = True

            def _human_facing_candidate(self, aligned):
                if self.fail:
                    raise FaceLandmarksUnavailable('temporary dense landmark failure')
                return np.full((100, 100), 170, dtype=np.uint8)

        display = Display()
        runner.display_pipeline, runner.display_interval = display, 1
        shown, info = runner.process_frame(rgb, 1, timestamp=1 / 30)
        self.assertEqual(info['display_status'], 'concealed_after_landmark_loss')
        self.assertEqual(shown[80, 80], 0)
        self.assertGreater(pipeline.classified[-1].mean(), 150)
        display.fail = False
        runner.process_frame(rgb, 2, timestamp=2 / 30)
        display.fail = True
        shown, info = runner.process_frame(rgb, 3, timestamp=3 / 30)
        self.assertEqual(info['display_status'], 'cached_after_landmark_loss')
        self.assertEqual(shown[80, 80], 170)
        shown, info = runner.process_frame(rgb, 4, timestamp=.7)
        self.assertEqual(info['display_status'], 'concealed_after_landmark_loss')
        self.assertEqual(shown[80, 80], 0)

    def test_large_face_track_jump_is_not_same_track(self):
        from expression_deid.video_demo import FastVideoRunner
        self.assertTrue(FastVideoRunner._same_track(np.array([10, 10, 100, 100]), np.array([13, 14, 100, 100])))
        self.assertFalse(FastVideoRunner._same_track(np.array([10, 10, 100, 100]), np.array([200, 10, 100, 100])))


if __name__ == '__main__':
    unittest.main()
