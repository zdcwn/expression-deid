"""FER must remain the training primary, independently of report domains."""
import unittest
import numpy as np
from expression_deid.cli import build_parser
from expression_deid.training.expression_current import sampling_weights


class FERPrimaryTests(unittest.TestCase):
    def test_training_draws_respect_the_declared_fer_fraction(self):
        labels=np.array([0,0,0,3,3,6,0,3,6])
        domains=np.array([0,0,0,0,0,0,1,1,1])
        weights=sampling_weights(labels,domains,.85,1.2)
        self.assertAlmostEqual(float(weights[domains==0].sum()),.85)
        self.assertAlmostEqual(float(weights[domains==1].sum()),.15)
        with self.assertRaises(ValueError): sampling_weights(labels,domains,.5,1.2)

    def test_candidate_is_separate_from_the_deployed_model(self):
        args=build_parser().parse_args(['train-expression'])
        self.assertEqual(args.output,'models/training/expression_fer2013')
        self.assertEqual(args.cache_name,'anonymous-runtime.npy')

    def test_release_contains_the_selected_runtime_models(self):
        import json
        from expression_deid.config import ROOT
        from scripts.distribution.build_windows import release_resources
        files=release_resources()
        selected=json.loads((ROOT/'configs/runtime_models.json').read_text(encoding='utf-8'))
        for key in ('identity','anonymizer'):
            self.assertIn(selected[key]['file'],files)
        self.assertIn(selected['expression']['current'],files)
        self.assertFalse(any('expression_video_quality' in path for path in files))
        self.assertTrue(all((ROOT/path).is_file() for path in files))


if __name__=='__main__': unittest.main()
