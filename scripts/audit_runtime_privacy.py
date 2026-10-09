"""Face-only privacy checks on the exact fast anonymous validation cache."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from PIL import Image
import torch
from expression_deid.identity_gallery import SFaceGallery
from expression_deid.inference import ExpressionDeidentifier
from expression_deid.models import load_checkpoint
from expression_deid.normalize import load_gray, load_stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', default=str(ROOT / 'outputs/realtime_training/anonymous_cache.npz'))
    p.add_argument('--output', default=str(ROOT / 'outputs/full_audit_20261008/fast_privacy.json'))
    args = p.parse_args(); torch.set_num_threads(1)
    data = np.load(args.cache)
    indices = np.flatnonzero((data['domains'] == 'kmu') & (data['splits'] == 'valid'))
    recognizer = cv2.FaceRecognizerSF_create(str(ROOT / 'models/face_recognition_sface_2021dec.onnx'), '')
    gallery = SFaceGallery(ROOT / 'datasets/lfw-gallery', recognizer)
    model, metadata = load_checkpoint(ROOT / 'models/identity_faceonly.pt', device='cuda')
    originals = np.stack([ExpressionDeidentifier._resize(load_gray(ROOT / str(data['sources'][i]))) for i in indices])
    anonymous = data['images'][indices]
    stats = load_stats()
    def identity(images):
        crops = np.stack([ExpressionDeidentifier._face_only_view(image) for image in images])
        result = []
        with torch.inference_mode():
            for batch in np.array_split(crops, max(1, int(np.ceil(len(crops) / 32)))):
                x = torch.from_numpy(batch.copy()).float()[:, None].cuda() / 255
                result.extend(model((x-stats.mean)/stats.std).argmax(1).cpu().tolist())
        return np.array(result)
    expected = np.array([metadata['metrics']['person_to_id'][Path(str(data['sources'][i])).stem.split('_')[0]] for i in indices])
    original_ids, anonymous_ids = identity(originals), identity(anonymous)
    rows = []
    for index, (source, edited) in enumerate(zip(originals, anonymous)):
        row = gallery.compare(source, edited)
        row.update(source=str(data['sources'][indices[index]]), original_id_correct=bool(original_ids[index] == expected[index]),
                   anonymous_id_correct=bool(anonymous_ids[index] == expected[index]))
        rows.append(row)
        if (index+1) % 40 == 0: print(f'privacy {index+1}/{len(indices)}', flush=True)
    summary = dict(samples=len(rows), gallery_size=1681, input='actual fast anonymous composite and realignment',
                   original_closed_set_accuracy=float((original_ids == expected).mean()),
                   anonymous_closed_set_accuracy=float((anonymous_ids == expected).mean()),
                   sface_source_match_rate=float(np.mean([r['gallery_match_original'] for r in rows])),
                   sface_mean_cosine=float(np.mean([r['gallery_source_cosine'] for r in rows])),
                   caveat='KMU participants overlap the anonymizer training identities; source frames are enrolled per query. This is a privacy regression, not unseen-person evidence.')
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(summary=summary, rows=rows), ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)

if __name__ == '__main__': main()
