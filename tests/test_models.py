import tempfile
import unittest
import inspect
from pathlib import Path

import numpy as np

from expression_deid.data import split_indices, split_ppb_people
from expression_deid.evaluate import summarize
from expression_deid.inference import ExpressionDeidentifier
from expression_deid.train import _expression_preservation_losses
try:
    from expression_deid.models import AnonymizerUNet, ExpressionNet, IdentityNet, load_checkpoint, save_checkpoint
except RuntimeError as exc:  # PyTorch is optional for preprocessing users.
    raise unittest.SkipTest(str(exc))


class ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("PyTorch 未安装")
        cls.torch = torch

    def test_model_shapes_and_range(self):
        torch = self.torch
        x = torch.rand(2, 1, 100, 100)
        self.assertEqual(ExpressionNet()(x).shape, (2, 7))
        self.assertEqual(IdentityNet(3)(x).shape, (2, 3))
        output = AnonymizerUNet()(x)
        self.assertEqual(output.shape, x.shape)
        self.assertGreaterEqual(float(output.detach().min()), 0.0)
        self.assertLessEqual(float(output.detach().max()), 1.0)

    def test_checkpoint_roundtrip(self):
        model = ExpressionNet()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "expression.pt"
            save_checkpoint(model, path, kind="expression", epoch=2)
            loaded, payload = load_checkpoint(path)
            self.assertEqual(payload["kind"], "expression")
            for left, right in zip(model.parameters(), loaded.parameters()):
                self.assertTrue(np.array_equal(left.detach().numpy(), right.detach().numpy()))

    def test_split_is_deterministic_and_disjoint(self):
        train, valid = split_indices(20, 0.25, seed=4)
        train_again, valid_again = split_indices(20, 0.25, seed=4)
        self.assertEqual(train, train_again)
        self.assertEqual(valid, valid_again)
        self.assertEqual(set(train) & set(valid), set())
        self.assertEqual(len(train) + len(valid), 20)

    def test_joint_summary(self):
        rows = [
            {"expression_original_correct": True, "expression_anonymized_correct": True,
             "expression_consistent": True, "identity_cosine": 0.1,
             "identity_prediction_changed": True},
            {"expression_original_correct": True, "expression_anonymized_correct": False,
             "expression_consistent": False, "identity_cosine": 0.3,
             "identity_prediction_changed": False},
        ]
        result = summarize(rows)
        self.assertEqual(result["samples"], 2)
        self.assertEqual(result["expression_consistency"], 0.5)
        self.assertEqual(result["identity_prediction_change_rate"], 0.5)

    def test_expression_loss_prefers_original_prediction(self):
        torch = self.torch
        original = torch.tensor([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
        same = original.clone()
        different = torch.tensor([[0.0, 4.0, 0.0], [4.0, 0.0, 0.0]])
        same_loss = _expression_preservation_losses(original, same, torch.tensor([0, 1]))[0]
        different_loss = _expression_preservation_losses(original, different, torch.tensor([0, 1]))[0]
        self.assertLess(float(same_loss), float(different_loss))

    def test_reference_face_excludes_original_identity(self):
        first = np.zeros((100, 100), dtype=np.float32)
        second = np.ones((100, 100), dtype=np.float32)
        bank = {3: {1: [first], 2: [second]}}
        selected = ExpressionDeidentifier._reference_face(
            bank, expression_label=3, identity_label=0, gray=first * 255
        )
        self.assertTrue(np.array_equal(selected, second))

    def test_shared_confidence_cannot_increase_after_replacement(self):
        self.assertEqual(ExpressionDeidentifier._shared_confidence(0.42, 0.98), 0.42)
        self.assertEqual(ExpressionDeidentifier._shared_confidence(0.92, 0.71), 0.71)

    def test_identity_diagnostic_excludes_outer_hair_and_background(self):
        source = np.full((100, 100), 80, dtype=np.uint8)
        source[:12, :] = 255
        source[:, :12] = 255
        cropped = ExpressionDeidentifier._face_only_view(source)
        self.assertEqual(cropped.shape, (100, 100))
        self.assertLess(float(cropped[:8].mean()), 150.0)

    def test_runtime_expression_path_is_anonymized_only(self):
        source = inspect.getsource(ExpressionDeidentifier.process)
        self.assertIn('"expression_input_source": "anonymized_only"', source)
        self.assertIn('purpose="machine"', source)
        self.assertIn("_display_from_last_source", source)
        self.assertNotIn("predict_expression(original_face", source)
        self.assertNotIn("expression_label=", source)

    def test_privacy_first_summary_does_not_invent_consistency(self):
        result = summarize([{
            "expression_original_correct": None,
            "expression_anonymized_correct": True,
            "expression_consistent": None,
        }])
        self.assertEqual(result["expression_anonymized_accuracy"], 1.0)
        self.assertNotIn("expression_consistency", result)

    def test_python_video_runner_classifies_edited_face_only(self):
        from expression_deid.video_demo import FastVideoRunner

        source = inspect.getsource(FastVideoRunner.process_frame)
        self.assertIn("predict_expression(edited100", source)
        self.assertNotIn("predict_expression(gray", source)

    def test_identity_candidate_selection_is_before_expression_inference(self):
        source = inspect.getsource(ExpressionDeidentifier.select_anonymized_candidate)
        self.assertIn("predict_sface_similarity", source)
        self.assertNotIn("predict_expression", source)
        process_source = inspect.getsource(ExpressionDeidentifier.process)
        self.assertIn('"privacy_selection"', process_source)

    def test_ppb_split_keeps_participants_disjoint_and_json_safe(self):
        class Fake:
            items = [(None, p, {"person": p}) for p in ("01", "01", "02", "02", "03", "03", "04", "04", "05", "05")]
        train, valid, test, people = split_ppb_people(Fake(), seed=7)
        self.assertFalse(set(train) & set(valid))
        self.assertFalse(set(train) & set(test))
        self.assertFalse(set(valid) & set(test))
        self.assertEqual(len(set(people["train"]) | set(people["valid"]) | set(people["test"])), 5)
        import json
        json.dumps(people)


if __name__ == "__main__":
    unittest.main()
