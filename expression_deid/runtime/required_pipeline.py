"""Candidate inference for the required datasets, without source-expression access."""
from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from expression_deid.labels import EXPRESSION_NAMES,EXPRESSION_ZH
from expression_deid.models import load_checkpoint
from expression_deid.preprocessing.align import landmarks_of,select_face,similarity_matrix,template_for,warp_gray
from expression_deid.preprocessing.normalize import to_gray
from expression_deid.training.required_training import face_only,file_digest,normalize


class RequiredPipeline:
    def __init__(self,anonymizer_checkpoint,expression_checkpoint=None,identity_checkpoint=None,device='cpu'):
        self.device=torch.device('cuda' if device=='auto' and torch.cuda.is_available() else 'cpu' if device=='auto' else device)
        self.anonymizer,self.anonymizer_meta=load_checkpoint(anonymizer_checkpoint,device=self.device)
        from expression_deid.runtime.restored_expression import MainExpressionPredictor
        self.expression_predictor=MainExpressionPredictor(str(self.device),checkpoint=expression_checkpoint)
        self.expression=self.expression_predictor.model
        self.expression_meta=self.expression_predictor.metadata
        for payload,kind in [(self.anonymizer_meta,'anonymizer')]:
            if payload.get('kind')!=kind or payload['metrics'].get('smoke_only'):
                raise ValueError('Use a fully trained checkpoint of the correct module')
        if self.anonymizer_meta['metrics'].get('datasets',{}).get('primary')!='celebahq':
            raise ValueError('Use the CelebA-HQ anonymizer')
        spec=self.expression_meta.get('metrics',{})
        if spec.get('anonymizer_sha256') and spec['anonymizer_sha256']!=file_digest(anonymizer_checkpoint):
            raise ValueError('Expression checkpoint was fine-tuned with another anonymizer')
        self.identity=self.identity_meta=None
        if identity_checkpoint:
            self.identity,self.identity_meta=load_checkpoint(identity_checkpoint,device=self.device)
            if self.identity_meta['metrics'].get('datasets',{}).get('primary')!='pins':
                raise ValueError('Identity diagnostics require the Pins model')
            if self.anonymizer_meta['metrics']['identity_checkpoint_sha256']!=file_digest(identity_checkpoint):
                raise ValueError('Use the same project identity model as the anonymizer training critic')
        self.detector=None

    @staticmethod
    def composite(gray,original_face,anonymous_face,face_mask,matrix):
        inverse=cv2.invertAffineTransform(matrix)
        size=(gray.shape[1],gray.shape[0])
        difference=anonymous_face.astype(np.float32)-original_face.astype(np.float32)
        delta=cv2.warpAffine(difference,inverse,size,flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        coverage=cv2.warpAffine(face_mask.astype(np.float32),inverse,size,flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        output=gray.copy(); editable=coverage>0
        output[editable]=np.rint(gray[editable].astype(np.float32)+delta[editable]).clip(0,255).astype(np.uint8)
        return output,editable

    def predict_anonymous(self,anonymous,*,input_source):
        if input_source!='anonymized_only': raise ValueError('原图不可进入表情分类器')
        if not isinstance(anonymous,np.ndarray) or anonymous.dtype!=np.uint8 or anonymous.shape!=(100,100):
            raise ValueError('Expression inference requires a 100x100 uint8 anonymous grayscale face')
        return self.expression_predictor.predict_anonymous(anonymous,input_source=input_source)

    def process(self,source,*,aligned=False,diagnostics=False):
        if isinstance(source,(str,Path)):
            with Image.open(source) as photo: rgb=np.asarray(photo.convert('RGB'))
        else:
            value=np.asarray(source)
            rgb=np.repeat(value[:,:,None],3,axis=2) if value.ndim==2 else value[:,:,:3]
        gray=to_gray(Image.fromarray(rgb))
        if aligned:
            original_face=np.asarray(Image.fromarray(gray).resize((100,100),Image.Resampling.BILINEAR))
            matrix=np.asarray([[100/gray.shape[1],0,0],[0,100/gray.shape[0],0]],dtype=np.float64)
            landmarks=None
        else:
            if self.detector is None:
                from expression_deid.preprocessing.detect import YuNet
                self.detector=YuNet()
            face,reason=select_face(self.detector.detect(rgb),gray.shape)
            if face is None: raise ValueError('No reliable face: '+reason)
            landmarks=landmarks_of(face); original_face,_=warp_gray(gray,landmarks,size=100)
            matrix=similarity_matrix(landmarks,template_for(100))
        values=torch.as_tensor(original_face.copy(),device=self.device,dtype=torch.float32)[None,None]/255
        with torch.inference_mode():
            anonymous=self.anonymizer(values)[0,0].clamp(0,1).mul(255).round().to(torch.uint8).cpu().numpy()
        mask=self.anonymizer.face_mask[0,0].cpu().numpy()
        output,editable=self.composite(gray,original_face,anonymous,mask,matrix)
        # Classification reads pixels from the final composite, not an unseen
        # intermediate patch and never the original frame.
        if landmarks is None:
            expression_input=np.asarray(Image.fromarray(output).resize((100,100),Image.Resampling.BILINEAR))
        else: expression_input,_=warp_gray(output,landmarks,size=100)
        expression=self.predict_anonymous(expression_input,input_source='anonymized_only')
        result={'original':gray,'anonymized':output,'expression_anonymized':expression,
                'expression_input_source':'anonymized_only','protected_pixels_unchanged':bool(np.array_equal(gray[~editable],output[~editable]))}
        if diagnostics and self.identity is not None:
            stats=self.identity_meta['metrics']['normalization']
            batch=torch.as_tensor(np.stack([original_face,expression_input]),device=self.device,dtype=torch.float32)[:,None]/255
            with torch.inference_mode(): probs=self.identity(normalize(face_only(batch),stats)).float().softmax(1).cpu().numpy()
            result['identity_original']={'label':int(probs[0].argmax()),'confidence':float(probs[0].max())}
            result['identity_anonymized']={'label':int(probs[1].argmax()),'confidence':float(probs[1].max())}
        return result

    def close(self):
        self.detector=None
