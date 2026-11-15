import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from expression_deid.training.required_data import (
    FER_TO_PROJECT,grouped_split,load_policy,prepare_fer2013,verify_prepared,
)


class RequiredDataTests(unittest.TestCase):

    def test_raf_auxiliary_excludes_overlap_with_primary_images(self):
        from PIL import Image
        from expression_deid.training.required_data import _save_prepared,_pixel_hash,prepare_raf_auxiliary
        rng=np.random.default_rng(81)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); source=root/'raf'; primary=root/'primary'
            primary_images=[rng.integers(0,256,(100,100),dtype=np.uint8) for _ in range(3)]
            rows=[{'source':str(i),'label':0,'split':split,'group':_pixel_hash(primary_images[i]),
                   'pixel_sha256':_pixel_hash(primary_images[i])} for i,split in enumerate(['train','valid','test'])]
            _save_prepared(primary,primary_images,rows,{'dataset':'fer2013'})
            for split in ['train','valid']:
                for label in range(7):
                    directory=source/split/str(label); directory.mkdir(parents=True)
                    for i in range(3):
                        Image.fromarray(rng.integers(0,256,(100,100),dtype=np.uint8)).save(directory/(str(i)+'.png'))
            Image.fromarray(primary_images[0]).save(source/'train/0/overlap.png')
            metadata=prepare_raf_auxiliary(source,root/'prepared',primary)
            self.assertGreaterEqual(metadata['cross_primary_near_duplicates_removed'],1)
            records=[json.loads(s) for s in (root/'prepared/manifest.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertFalse(any('overlap' in r['source'] for r in records))
            self.assertEqual(set(r['split'] for r in records),{'train','valid','test'})

    def test_identity_input_is_independent_of_outside_oval_pixels(self):
        import torch
        from expression_deid.training.required_training import face_only
        y,x=torch.meshgrid(torch.arange(88),torch.arange(76),indexing='ij')
        inside=((x-37.5)/36)**2+((y-43.5)/44)**2<1
        a=torch.linspace(.2,.8,10000).reshape(1,1,100,100).clone(); b=a.clone()
        a[:,:,12:100,12:88][...,~inside]=0; b[:,:,12:100,12:88][...,~inside]=1
        self.assertTrue(torch.equal(face_only(a),face_only(b)))

    def test_policy_rejects_wrong_primary_and_excess_auxiliary(self):
        policy=load_policy()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'policy.json'
            for section,updates in [('expression',{'primary':'raf'}),('expression',{'auxiliary':['kmu','ppb-emo']}),
                                    ('expression',{'auxiliary':['raf-db']}),('expression',{'primary_sampling_fraction':.5})]:
                changed=copy.deepcopy(policy); changed[section].update(updates)
                path.write_text(json.dumps(changed),encoding='utf-8')
                with self.assertRaises(ValueError): load_policy(path)

    def test_fer_uses_original_labels_and_removes_heldout_duplicates(self):
        rng=np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); csv_path=root/'fer2013.csv'; rows=[]
            for usage in ['Training','PublicTest','PrivateTest']:
                for label in range(7):
                    pixels=rng.integers(0,256,2304,dtype=np.uint8)
                    rows.append({'Usage':usage,'emotion':label,'pixels':' '.join(map(str,pixels.tolist()))})
            rows.append({**rows[-1],'Usage':'Training'})
            with csv_path.open('w',newline='',encoding='utf-8') as handle:
                writer=csv.DictWriter(handle,fieldnames=['emotion','pixels','Usage']); writer.writeheader(); writer.writerows(rows)
            metadata=prepare_fer2013(csv_path,root/'prepared')
            self.assertEqual(metadata['cross_split_duplicates_removed'],1)
            records=[json.loads(line) for line in (root/'prepared/manifest.jsonl').read_text().splitlines()]
            for row in records: self.assertEqual(row['label'],FER_TO_PROJECT[row['original_label']])
            self.assertEqual(verify_prepared(root/'prepared')['samples'],21)
            records[0]['group']=records[-1]['group']
            (root/'prepared/manifest.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
            with self.assertRaisesRegex(ValueError,'multiple partitions'): verify_prepared(root/'prepared')

    def test_identity_near_duplicates_stay_in_one_partition(self):
        rng=np.random.default_rng(12); rows=[]
        for label in range(2):
            for i in range(12):
                code=int.from_bytes(rng.bytes(8),'big')
                rows.append({'label':label,'dhash':code,'pixel_sha256':f'{label}-{i}'})
                rows.append({'label':label,'dhash':code^1,'pixel_sha256':f'{label}-{i}-near'})
        result=grouped_split(rows,per_identity=True)
        for i in range(0,len(result),2): self.assertEqual(result[i]['split'],result[i+1]['split'])
        for label in range(2): self.assertEqual({r['split'] for r in result if r['label']==label},{'train','valid','test'})

    def test_retired_training_commands_are_absent(self):
        from expression_deid.cli import build_parser
        parser=build_parser()
        choices=next(action.choices for action in parser._actions if getattr(action,'choices',None))
        self.assertNotIn('train-expression-joint',choices)
        self.assertNotIn('train-expression-kmu',choices)
        self.assertIn('train-anonymizer',choices)

    def test_final_composite_is_the_only_expression_input(self):
        import torch
        from PIL import Image
        from unittest.mock import Mock
        from expression_deid.runtime.required_pipeline import RequiredPipeline
        class Edit(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.face_mask=torch.zeros(1,1,100,100)
                self.face_mask[:,:,30:70,30:70]=1
            def forward(self,x): return (x+self.face_mask*.1).clamp(0,1)
        pipeline=RequiredPipeline.__new__(RequiredPipeline)
        pipeline.device=torch.device('cpu'); pipeline.anonymizer=Edit(); pipeline.identity=None
        pipeline.predict_anonymous=Mock(return_value={'label':6,'confidence':.5})
        source=np.full((256,256),80,np.uint8)
        result=pipeline.process(source,aligned=True)
        pipeline.predict_anonymous.assert_called_once()
        actual=pipeline.predict_anonymous.call_args.args[0]
        expected=np.asarray(Image.fromarray(result['anonymized']).resize((100,100),Image.Resampling.BILINEAR))
        np.testing.assert_array_equal(actual,expected)
        self.assertFalse(np.array_equal(actual,np.full((100,100),80,np.uint8)))
        self.assertTrue(result['protected_pixels_unchanged'])
        self.assertEqual(pipeline.predict_anonymous.call_args.kwargs['input_source'],'anonymized_only')

    def test_failed_checkpoint_save_keeps_the_existing_best(self):
        from unittest.mock import patch
        from expression_deid.models import save_checkpoint_payload
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'best.pt'; target.write_bytes(b'previous-best')
            with patch('torch.save',side_effect=OSError('disk write failed')):
                with self.assertRaises(OSError): save_checkpoint_payload({'weights':1},target)
            self.assertEqual(target.read_bytes(),b'previous-best')
            self.assertEqual([p.name for p in Path(folder).iterdir()],['best.pt'])


    def test_cache_is_rebuilt_when_source_pixels_change(self):
        import torch
        from expression_deid.models import model_class,save_checkpoint
        from expression_deid.training.required_training import anonymous_cache
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); data=root/'data'; data.mkdir()
            np.save(data/'images.npy',np.full((1,100,100),80,np.uint8))
            (data/'manifest.jsonl').write_text('{}\n',encoding='utf-8')
            (data/'metadata.json').write_text('{"dataset":"fer2013"}',encoding='utf-8')
            checkpoint=root/'anonymizer.pt'
            save_checkpoint(model_class('MaskedAnonymizer')(),checkpoint,kind='anonymizer',
                            metrics={'datasets':{'primary':'celebahq'}})
            values,first=anonymous_cache(data,checkpoint,torch.device('cpu'))
            self.assertEqual(int(values[0,0,0]),80)
            values._mmap.close(); del values
            np.save(data/'images.npy',np.full((1,100,100),120,np.uint8))
            values,second=anonymous_cache(data,checkpoint,torch.device('cpu'))
            self.assertEqual(int(values[0,0,0]),120)
            self.assertNotEqual(first['source_images_sha256'],second['source_images_sha256'])
            values._mmap.close()

    def test_full_photo_evaluation_counts_detection_failures(self):
        from unittest.mock import patch,Mock
        from expression_deid.training.required_evaluate import evaluate_full_photos
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); data=root/'datasets/pins/prepared'; data.mkdir(parents=True)
            records=[{'source':str(i)+'.jpg','label':0,'split':'valid'} for i in range(2)]
            (data/'manifest.jsonl').write_text('\n'.join(json.dumps(r) for r in records),encoding='utf-8')
            pipeline=Mock()
            pipeline.process.side_effect=[{'identity_original':{'label':0},'identity_anonymized':{'label':1},
                                           'protected_pixels_unchanged':True},ValueError('No reliable face')]
            with patch('expression_deid.training.required_evaluate.ROOT',root), \
                 patch('expression_deid.runtime.required_pipeline.RequiredPipeline',return_value=pipeline):
                result=evaluate_full_photos('id','anon','expr',split='valid',device='cpu')
            self.assertEqual((result['selected'],result['processed'],result['failed']),(2,1,1))
            self.assertEqual(result['original_accuracy'],1)
            self.assertEqual(result['anonymized_accuracy'],0)
            pipeline.close.assert_called_once()
