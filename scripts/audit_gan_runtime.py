"""Balanced tail-frame regression on the current desktop image/GAN path."""
import json
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from expression_deid.desktop_ui_qt import diagnostic_pipeline
from expression_deid.prepare import iter_kmu_aligned
from expression_deid.config import KMU_ALIGNED_DIR
from expression_deid.data import split_kmu_sequence_tails


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--per-class',type=int,default=4,help='0 使用全部 KMU 验证尾帧')
    parser.add_argument('--output',default='outputs/full_audit_20261008/final_gan_runtime.json')
    args=parser.parse_args()
    if args.per_class<0: raise ValueError('per-class 必须非负')
    torch.set_num_threads(1)
    items=list(iter_kmu_aligned(KMU_ALIGNED_DIR,include_review=False))
    _,valid=split_kmu_sequence_tails(items)
    selected=[]
    for label in range(6):
        group=[items[i] for i in valid if items[i][1]==label]
        selected.extend(group[:args.per_class] if args.per_class else group)
    pipeline=diagnostic_pipeline('cuda')
    rows=[]
    try:
        for path,label,record in selected:
            result=pipeline.process(path,aligned=True,diagnostics=True)
            rows.append(dict(source=str(path.relative_to(ROOT)),label=label,
                predicted=result['expression_anonymized']['label'],confidence=result['expression_anonymized']['confidence'],
                gallery_match_original=result['gallery_match_original'],gallery_source_cosine=result['gallery_source_cosine'],
                display_gallery_match_original=result['display_gallery_match_original'],display_sface_cosine=result['display_sface_cosine'],
                background_preserved=result['background_preserved'], expression_input_source=result['expression_input_source']))
    finally:pipeline.close()
    summary=dict(samples=len(rows),expression_accuracy=float(np.mean([r['label']==r['predicted'] for r in rows])),
                 sface_source_match_rate=float(np.mean([r['gallery_match_original'] for r in rows])),
                 display_sface_source_match_rate=float(np.mean([r['display_gallery_match_original'] for r in rows])),
                 gallery_size=1681, privacy_selection=False,
                 scope=f'{len(rows)} KMU tail frames, inherited training people; current desktop experiment/GAN input, not fast video or full 1100 historical evaluation')
    (ROOT/args.output).write_text(json.dumps(dict(summary=summary,rows=rows),indent=2),encoding='utf-8')
    print(summary,flush=True)


if __name__=='__main__':main()
