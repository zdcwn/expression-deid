import unittest

import numpy as np

from expression_deid.preprocessing.align import landmarks_ok, select_face, similarity_matrix, warp_gray
from expression_deid.labels import parse_kmu_stem
from expression_deid.preprocessing.normalize import NormStats, expression_input, normalize
from expression_deid.preprocessing.prepare import _mark_jitter


def _record(person, expression, frame, eye_x):
    return {
        "person": person,
        "expression": expression,
        "frame": frame,
        "status": "ok",
        "reason": "",
        "landmarks": [[eye_x, 100.0], [eye_x + 40.0, 100.0]],
    }


def _face(box, points, score):
    row = np.empty(15, np.float64)
    row[:4] = box
    row[4:14] = np.asarray(points, np.float64).reshape(-1)
    row[14] = score
    return row


def _frontal_points(origin_x, origin_y, scale=1.0):
    base = np.array(
        [
            [30.0, 40.0],
            [70.0, 42.0],
            [50.0, 60.0],
            [34.0, 78.0],
            [68.0, 80.0],
        ]
    )
    return base * scale + np.array([origin_x, origin_y])


class PreprocessTest(unittest.TestCase):
    def test_kmu_name(self):
        parsed = parse_kmu_stem("02_FE_s01_031")
        self.assertEqual(parsed["person"], "02")
        self.assertEqual(parsed["expression"], "FE")
        self.assertEqual(parsed["label"], 1)
        self.assertEqual(parsed["frame"], 31)
        self.assertEqual(parse_kmu_stem("not-a-name")["label"], -1)

    def test_similarity_recovers_scale_rotation(self):
        rng = np.random.default_rng(0)
        src = rng.normal(size=(5, 2))
        theta = 0.3
        rotation = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        dst = 1.8 * (src @ rotation.T) + np.array([12.0, -4.0])
        matrix = similarity_matrix(src, dst)
        predicted = src @ matrix[:, :2].T + matrix[:, 2]
        self.assertLess(np.abs(predicted - dst).max(), 1e-8)

    def test_warp_puts_dot_on_template(self):
        from expression_deid.preprocessing.align import template_for

        template = template_for(256)
        gray = np.zeros((400, 400), np.uint8)
        point = template[0] / 256.0 * 200.0 + np.array([40.0, 30.0])
        x, y = int(point[0]), int(point[1])
        gray[y - 1 : y + 2, x - 1 : x + 2] = 255
        landmarks = template / 256.0 * 200.0 + np.array([40.0, 30.0])
        warped, fit_error = warp_gray(gray, landmarks, 256)
        ys, xs = np.nonzero(warped > 10)
        self.assertLess(abs(xs.mean() - template[0, 0]), 1.5)
        self.assertLess(abs(ys.mean() - template[0, 1]), 1.5)
        self.assertLess(fit_error, 1e-6)

    def test_largest_face_wins_and_low_score_is_marked(self):
        small = _face((10, 10, 80, 100), _frontal_points(10, 10), 0.95)
        large = _face((100, 20, 160, 200), _frontal_points(110, 30, scale=2.0), 0.93)
        chosen, reason = select_face(np.stack([small, large]), (400, 500))
        self.assertEqual(reason, "")
        self.assertAlmostEqual(chosen[2], 160.0)
        weak = _face((10, 10, 80, 100), _frontal_points(10, 10), 0.6)
        chosen, reason = select_face(weak[None, :], (200, 200))
        self.assertEqual(reason, "score_below_0.8")
        swapped = _frontal_points(10, 10)
        swapped[0], swapped[1] = swapped[1].copy(), swapped[0].copy()
        self.assertFalse(landmarks_ok(_face((10, 10, 80, 100), swapped, 0.99), (200, 200)))

    def test_head_motion_is_kept_and_a_spike_is_marked(self):
        drift = [_record("01", "HA", frame, 100 + frame * 8) for frame in range(1, 9)]
        _mark_jitter(drift)
        self.assertTrue(all(record["status"] == "ok" for record in drift))
        spike = [_record("01", "HA", frame, 100.0) for frame in range(1, 6)]
        spike[2]["landmarks"] = [[200.0, 100.0], [240.0, 100.0]]
        _mark_jitter(spike)
        self.assertEqual([record["status"] for record in spike], ["ok", "ok", "jitter", "ok", "ok"])
        self.assertEqual(spike[2]["reason"], "eye_spike")

    def test_expression_input_shape_and_augment_switch(self):
        stats = NormStats(mean=0.5, std=0.2)
        gray = np.tile(np.linspace(0, 255, 32, dtype=np.uint8), (32, 1))
        plain = expression_input(gray, stats, augment=False)
        again = expression_input(gray, stats, augment=False)
        self.assertEqual(plain.shape, (1, 100, 100))
        self.assertTrue(np.array_equal(plain, again))
        flipped = expression_input(gray, stats, augment=True, rng=np.random.default_rng(1))
        self.assertFalse(np.array_equal(plain, flipped))
        component = expression_input(gray, stats, component_augment=True, rng=np.random.default_rng(1))
        self.assertEqual(component.shape, (1, 100, 100))
        self.assertFalse(np.array_equal(plain, component))
        values = normalize(np.array([[0, 255]], np.uint8), stats)
        self.assertAlmostEqual(float(values[0, 0]), (0.0 - 0.5) / 0.2, places=5)


if __name__ == "__main__":
    unittest.main()
