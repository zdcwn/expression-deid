import hashlib
import json
from pathlib import Path
import subprocess
import unittest

import numpy as np
import torch

from expression_deid.config import ROOT
from expression_deid.runtime.restored_expression import MainExpressionPredictor


class RestoredExpressionTests(unittest.TestCase):
    def test_compact_probability_matches_main_checkpoint_directly(self):
        predictor=MainExpressionPredictor()
        gray=np.random.default_rng(4).integers(0,256,(3,100,100),dtype=np.uint8)
        actual=predictor.predict_anonymous_batch(gray,input_source='anonymized_only')
        with torch.inference_mode():
            x=torch.from_numpy(gray.astype(np.float32))[:,None]/255
            expected=(predictor.model((x-predictor.mean)/predictor.std)/predictor.temperature).softmax(1).numpy()
        np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-7)
        with self.assertRaises(ValueError): predictor.predict_anonymous_batch(gray,input_source='original')

    def test_selected_anonymizer_and_identity_have_not_changed(self):
        config=json.loads((ROOT/'configs/runtime_models.json').read_text(encoding='utf-8'))
        for name in ('anonymizer','identity'):
            spec=config[name]
            with (ROOT/spec['file']).open('rb') as handle:
                self.assertEqual(hashlib.file_digest(handle,'sha256').hexdigest(),spec['sha256'])

    def test_abandoned_experiments_are_not_available(self):
        from expression_deid.models import model_class
        for name in ['ConvNeXtExpressionNet','ConvNeXtLocalExpressionNet','SpatialTransferExpressionNet']:
            with self.assertRaises(ValueError): model_class(name)
        self.assertFalse((ROOT/'configs/expression_next_run.json').exists())

    def test_runtime_expression_is_the_restored_self_trained_asset(self):
        config=json.loads((ROOT/'configs/runtime_models.json').read_text(encoding='utf-8'))
        path=ROOT/config['expression']['current']
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),config['expression_sha256'])
        from expression_deid.models import load_checkpoint
        _,metadata=load_checkpoint(path)
        self.assertEqual(metadata['metrics']['recipe'],'runtime_domain_fer2013_expression')
        self.assertEqual(metadata['metrics']['datasets']['primary'],'fer2013')
        self.assertEqual(config['expression_training']['primary'],'fer2013')
        self.assertEqual(metadata['metrics']['anonymizer_sha256'],config['anonymizer']['sha256'])

    def test_display_backend_does_not_change_current_classifier_pixels(self):
        from expression_deid.ui.opencv_demo import build_pipeline
        pipeline=build_pipeline('cpu',use_gan=False,fast_mode=False)
        gray=np.random.default_rng(8).integers(0,256,(100,100),dtype=np.uint8)
        try:
            pipeline.anonymize(gray,aligned=True,purpose='machine')
            expected=pipeline._last_anonymized_aligned.copy()
            # A display generator must not switch the current model's
            # training domain to geometry-generated machine input.
            pipeline.ganonymization_backend=object()
            pipeline.anonymize(gray,aligned=True,purpose='machine')
            np.testing.assert_array_equal(pipeline._last_anonymized_aligned,expected)
        finally:
            pipeline.ganonymization_backend=None
            pipeline.close()


if __name__=='__main__': unittest.main()
