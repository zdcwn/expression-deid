import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from expression_deid.expression_quality import anonymous_tensor, QualityExpressionPredictor
from expression_deid.labels import EXPRESSION_NAMES


class FixedLogits(torch.nn.Module):
    def __init__(self, values):
        super().__init__(); self.register_buffer('values', torch.tensor(values, dtype=torch.float32))
    def forward(self, x): return self.values[None].expand(len(x), -1)


class SequenceLogits(torch.nn.Module):
    def __init__(self, rows):
        super().__init__(); self.register_buffer('rows', torch.tensor(rows, dtype=torch.float32))
    def forward(self, x): return self.rows[:len(x)]


class QualityExpressionTests(unittest.TestCase):
    def manifest(self, folder, **changes):
        asset=Path(folder)/'weights';asset.write_bytes(b'checked fixture')
        data=dict(schema=1, expression_input_source='anonymized_only', class_names=list(EXPRESSION_NAMES),
                  preprocessing='float_bilinear_260_imagenet_gray_v1', quality_weight=.5,
                  quality_temperature=2.,compact_temperature=1.,final_temperature=1.,compact_mean=.5,compact_std=.2,
                  quality_model={'file':'weights','sha256':hashlib.sha256(asset.read_bytes()).hexdigest()},
                  compact_model={'file':'weights','sha256':hashlib.sha256(asset.read_bytes()).hexdigest()})
        data.update(changes);path=Path(folder)/'manifest.json';path.write_text(json.dumps(data));return path

    def test_float_preprocessing_matches_opencv_and_mirror(self):
        import cv2
        gray=np.random.default_rng(12).integers(0,256,(100,100),dtype=np.uint8)
        views, x=anonymous_tensor(gray,device='cpu')
        np.testing.assert_array_equal(views[1,0].numpy(),np.fliplr(gray).astype(np.float32)/255)
        expected=cv2.resize(gray.astype(np.float32)/255,(260,260),interpolation=cv2.INTER_LINEAR)
        # CPU interpolation kernels differ by up to 0.002 gray levels here.
        # This bound is still below 0.003 of one uint8 level after normalization.
        np.testing.assert_allclose(x[0,0].numpy(),(expected-.485)/.229,atol=4e-5)
        self.assertEqual(tuple(x.shape),(2,3,260,260))

    def test_only_uint8_aligned_grayscale_is_accepted(self):
        for image in (np.zeros((100,100,3),np.uint8),np.zeros((99,100),np.uint8),np.full((100,100),np.nan)):
            with self.assertRaises(ValueError): anonymous_tensor(image,device='cpu')

    def test_manifest_contract_and_parameters_are_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            for changes in ({'class_names':list(reversed(EXPRESSION_NAMES))},{'quality_temperature':0},
                            {'compact_std':float('nan')},{'expression_input_source':'original'}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    QualityExpressionPredictor(manifest=self.manifest(folder,**changes))

    def test_modified_asset_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            path=self.manifest(folder);(Path(folder)/'weights').write_bytes(b'tampered')
            with patch('torch.jit.load') as load, self.assertRaisesRegex(ValueError,'校验失败'):
                QualityExpressionPredictor(manifest=path)
            load.assert_not_called()

    def test_source_pixels_are_rejected_and_probabilities_are_real(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch('torch.jit.load',return_value=FixedLogits([0,0,0,0,0,0,2])), patch('expression_deid.expression_quality.load_checkpoint',return_value=(FixedLogits([0,0,0,0,0,0,2]),{})):
                model=QualityExpressionPredictor(manifest=self.manifest(folder))
            image=np.full((100,100),100,np.uint8)
            with self.assertRaisesRegex(ValueError,'原始图像'): model.predict_anonymous(image,input_source='original')
            result=model.predict_anonymous(image,input_source='anonymized_only')
            a=torch.tensor([0.,0,0,0,0,0,2])
            expected=(a.softmax(0)+(a/2).softmax(0))/2
            np.testing.assert_allclose(result['probabilities'],expected.numpy(),atol=1e-7)
            self.assertEqual(result['name'],'neutral')
            self.assertAlmostEqual(result['confidence'],float(expected[6]),places=6)
            batch=model.predict_anonymous_batch(np.stack([image,image]),input_source='anonymized_only')
            np.testing.assert_allclose(batch[0],result['probabilities'],atol=1e-7)
            np.testing.assert_allclose(batch[1],result['probabilities'],atol=1e-7)
            from types import SimpleNamespace
            model.validate_pipeline(SimpleNamespace(ganonymization_backend=None,machine_privacy={}))
            for pipeline in (SimpleNamespace(ganonymization_backend=object(),machine_privacy={}),
                             SimpleNamespace(ganonymization_backend=None,machine_privacy={'blur':2})):
                with self.assertRaisesRegex(ValueError,'输入域不匹配'): model.validate_pipeline(pipeline)

    def test_nonfinite_model_output_stops_prediction(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch('torch.jit.load',return_value=FixedLogits([float('nan')]*7)), patch('expression_deid.expression_quality.load_checkpoint',return_value=(FixedLogits([0]*7),{})):
                model=QualityExpressionPredictor(manifest=self.manifest(folder))
            with self.assertRaisesRegex(RuntimeError,'输出无效'):
                model.predict_anonymous(np.zeros((100,100),np.uint8),input_source='anonymized_only')

    def test_batch_pairs_each_face_with_its_mirror(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch('torch.jit.load',return_value=FixedLogits([0]*7)), patch('expression_deid.expression_quality.load_checkpoint',return_value=(FixedLogits([0]*7),{})):
                model=QualityExpressionPredictor(manifest=self.manifest(folder))
            quality_rows=np.array([[3,0,0,0,0,0,0],[0,0,3,0,0,0,0],[0,0,0,0,0,3,0],[0,0,0,0,0,0,3]],dtype=np.float32)
            compact_rows=np.array([[0,3,0,0,0,0,0],[0,0,0,3,0,0,0],[0,0,0,0,3,0,0],[0,0,0,0,0,0,3]],dtype=np.float32)
            model.quality=SequenceLogits(quality_rows); model.compact=SequenceLogits(compact_rows)
            images=np.stack([np.zeros((100,100),np.uint8),np.full((100,100),255,np.uint8)])
            actual=model.predict_anonymous_batch(images,input_source='anonymized_only')
            qa=torch.softmax(torch.tensor(quality_rows)/2,1).numpy()
            qb=torch.softmax(torch.tensor(compact_rows),1).numpy()
            expected=.5*(qa[:2]+qa[2:])/2 + .5*(qb[:2]+qb[2:])/2
            expected=np.exp(np.log(expected)/1); expected/=expected.sum(1,keepdims=True)
            np.testing.assert_allclose(actual,expected,atol=1e-7)
            wrong=.5*(qa[::2]+qa[1::2])/2 + .5*(qb[::2]+qb[1::2])/2
            self.assertGreater(np.abs(actual-wrong).max(),.05)

    def test_training_rejects_raw_cache_and_partition_leakage(self):
        import importlib.util
        from expression_deid.config import ROOT
        spec=importlib.util.spec_from_file_location('quality_training_checks',ROOT/'scripts/train_video_quality.py')
        training=importlib.util.module_from_spec(spec);spec.loader.exec_module(training)
        fixture=dict(images=np.zeros((6,100,100),np.uint8), labels=np.array([0,0,6,0,0,6]),
                     domains=np.array(['raf','kmu','ppb']*2), splits=np.array(['train']*3+['valid']*3),
                     sources=np.array(['a','b','c','d','e','f']),privacy_source=np.array('anonymized_only'),
                     ppb_people=np.array(json.dumps({'train':['01'],'valid':['02'],'test':['03']})))
        training.validate_cache(fixture)
        for changes in ({'privacy_source':np.array('original')},
                        {'sources':np.array(['a','b','c','a','e','f'])},
                        {'ppb_people':np.array(json.dumps({'train':['01'],'valid':['01'],'test':['03']}))}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                training.validate_cache({**fixture,**changes})

if __name__=='__main__':unittest.main()
