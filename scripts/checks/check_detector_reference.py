"""Compare custom YuNet decoding with OpenCV on identical padded pixels."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from PIL import Image
from expression_deid.preprocessing.detect import YuNet, _letterbox_bgr
from expression_deid.config import MODEL_PATH, DETECT_SCORE, NMS_IOU, NMS_TOP_K


def main():
    custom=YuNet()
    reference=cv2.FaceDetectorYN_create(str(MODEL_PATH),'',(640,640),DETECT_SCORE,NMS_IOU,NMS_TOP_K)
    paths=list((ROOT/'datasets/kmu-fed').glob('*.jpg'))[:3]
    paths+=list((ROOT/'datasets/raf-db/valid/6').glob('*.jpg'))[:3]
    rows=[]
    for path in paths:
        with Image.open(path) as image: rgb=np.asarray(image.convert('RGB'))
        blob,scale=_letterbox_bgr(rgb)
        canvas=np.ascontiguousarray(blob[0].transpose(1,2,0).astype(np.uint8))
        _,expected=reference.detect(canvas)
        actual=custom.detect(rgb)
        expected=np.zeros((0,15)) if expected is None else expected.astype(np.float64)
        expected[:,:14]/=scale
        assert len(actual)==len(expected),(path,len(actual),len(expected))
        actual=actual[np.argsort(-actual[:,14])];expected=expected[np.argsort(-expected[:,14])]
        error=float(np.abs(actual[:,:14]-expected[:,:14]).max()) if len(actual) else 0.
        score_error=float(np.abs(actual[:,14]-expected[:,14]).max()) if len(actual) else 0.
        rows.append(dict(path=str(path.relative_to(ROOT)),faces=len(actual),max_coordinate_error=error,max_score_error=score_error))
        assert error<.1 and score_error<1e-4, rows[-1]
    assert len(rows)==6
    output = ROOT/'outputs/checks/detector_reference.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(rows,flush=True)


if __name__=='__main__':main()
