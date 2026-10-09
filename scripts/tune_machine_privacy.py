"""Compare local privacy edits on training frames; no source expression logits."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch
from expression_deid.align import template_for, similarity_matrix, warp_gray
from expression_deid.app import build_pipeline
from expression_deid.identity_gallery import SFaceGallery
from expression_deid.normalize import load_gray


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ratio-preserving',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1); cv2.setNumThreads(1)
    cache = np.load(ROOT / 'outputs/realtime_training/anonymous_cache.npz')
    pipeline = build_pipeline('cuda', use_gan=False)
    gallery = SFaceGallery(ROOT / 'datasets/lfw-gallery', cv2.FaceRecognizerSF_create(str(ROOT / 'models/face_recognition_sface_2021dec.onnx'), ''))
    indices = []
    for label in range(6):
        available = np.flatnonzero((cache['domains'] == 'kmu') & (cache['splits'] == 'train') & (cache['labels'] == label))
        indices.extend(available[np.linspace(0, len(available)-1, 10, dtype=int)])
    sources = []
    for i in indices:
        gray = load_gray(ROOT / str(cache['sources'][i]))
        points = template_for(gray.shape[0]); aligned = warp_gray(gray, points)[0]
        pipeline._runtime_donor = pipeline._select_component_donor(pipeline._resize(aligned))
        sources.append((gray, points, aligned, pipeline._machine_runtime_candidate(aligned), int(cache['labels'][i])))
    profiles = [dict(eye_scale=eye, eye_shift=shift, nose_scale=nose, mouth_scale=mouth, texture=texture, blur=blur)
                for eye, shift, nose, mouth, texture, blur in ((.55,4,1.45,.75,8,.4),(.4,7,1.7,.6,14,.7),(.3,10,2,.45,20,1),(.25,12,2.2,.4,26,1.2))]
    if args.ratio_preserving:
        profiles=[dict(eye_scale=eye,eye_shift=shift,nose_scale=nose,mouth_scale=mouth,texture=texture,blur=blur,ratio_preserving=True)
                  for eye,shift,nose,mouth,texture,blur in ((.65,7,1.45,.8,8,.4),(.55,8,1.6,.75,12,.5),(.45,9,1.7,.7,14,.7))]
    results = []
    for mode in ('features', 'face'):
        for number, profile in enumerate(profiles):
            matches, cosines, correct = [], [], []
            for gray, points, aligned, base, label in sources:
                edited = pipeline._machine_local_privacy(base, profile)
                frame = pipeline._composite_component_edit(gray, edited, (0,0,*gray.shape[::-1]), similarity_matrix(points, template_for(256)), mask_mode=mode)
                anonymous = pipeline._resize(warp_gray(frame, points)[0])
                diagnostic = gallery.compare(pipeline._resize(gray), anonymous)
                matches.append(diagnostic['gallery_match_original']); cosines.append(diagnostic['gallery_source_cosine'])
                correct.append(pipeline.predict_expression(anonymous, aligned=True)['label'] == label)
            row = dict(profile={**profile,'mask_mode':mode}, samples=len(matches), source_match_rate=float(np.mean(matches)), mean_cosine=float(np.mean(cosines)), frozen_expression_accuracy=float(np.mean(correct)), split='kmu/train')
            results.append(row); print(json.dumps(row), flush=True)
    (ROOT / 'outputs/full_audit_20261008' / ('privacy_ratio_trials.json' if args.ratio_preserving else 'privacy_trials.json')).write_text(json.dumps(results, indent=2), encoding='utf-8')

if __name__ == '__main__': main()
